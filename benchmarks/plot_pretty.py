import os
import click
import pandas as pd
from matplotlib import pyplot as plt
import re
import numpy as np
from itertools import cycle


def get_all_data_frames( path: str, names: tuple[ str, ... ] ) -> dict[
    str, tuple[ list[ float ], list[ pd.DataFrame ] ] ]:
    results = { }
    for name in names:
        qpses = [ ]
        dfs = [ ]
        pattern = re.compile( rf"^{name}_output_(\d+(\.\d+)?)\.csv$" )
        for filename in os.listdir( path ):
            match = pattern.match( filename )
            if match:
                qps = float( match.group( 1 ) )
                q = round( qps, 1 )
                if qps > 8:
                    continue
                df = pd.read_csv( os.path.join( path, filename ) )
                qpses += [ q ]
                dfs += [ df ]
        dfs = [ x for _, x in sorted( zip( qpses, dfs ) ) ]
        qpses = sorted( qpses )
        results[ name ] = (qpses, dfs)
    return results


def merge_baseline( results: dict[ str, tuple[ list[ float ], list[ pd.DataFrame ] ] ], names: tuple[ str, ... ] ) -> \
        dict[ str, tuple[ list[ float ], list[ pd.DataFrame ] ] ]:
    assert 'baseline' in results
    b_qps_to_df = { len( df ): df for qps, df in zip( *results[ 'baseline' ] ) }
    for name in names:
        qpses, dfs = results[ name ]
        for i in range( len( qpses ) ):
            prev_rows = len( dfs[ i ] )
            if prev_rows in b_qps_to_df:
                df_b = b_qps_to_df[ prev_rows ]
            else:
                max_len = max( b_qps_to_df.keys( ) )
                df_b = b_qps_to_df[ max_len ]

            df_b = df_b[ [ 'prompt_tokens', 'generation_tokens', 'ttlt' ] ]
            df_b = df_b.rename( columns = { 'ttlt': 'base_execution_time' } )

            def check_group_variance( group ):
                if group[ 'base_execution_time' ].max( ) / group[ 'base_execution_time' ].min( ) > 1.2:
                    print( f"Warning: Group variance {group[ 'base_execution_time' ].max( )} <-> {group[ 'base_execution_time' ].min( )} exceeds threshold" )
                return group.mean( )

            df_b = df_b.groupby( [ 'prompt_tokens', 'generation_tokens' ] )[ df_b.columns ].apply( check_group_variance,
                                                                                                   include_groups = False ).reset_index(
                drop = True )

            new_df = pd.merge( dfs[ i ], df_b, on = [ 'prompt_tokens', 'generation_tokens' ], how = 'left' )
            if prev_rows != len( new_df ):
                print( f"Lost rows in {name}-{qpses[ i ]}:" )
                print( "Original rows not in merged result:" )
                print( dfs[ i ][ ~dfs[ i ].index.isin( new_df.index ) ] )
                print( "\nMerged result rows not in original:" )
                print( new_df[ ~new_df.index.isin( dfs[ i ].index ) ] )
                raise ValueError( f"Lost rows during merge {name}-{qpses[ i ]}: {prev_rows} -> {len( new_df )}" )
            dfs[ i ] = new_df
            dfs[ i ][ 'request_e2e_slowdown' ] = dfs[ i ][ 'ttlt' ] / dfs[ i ][ 'base_execution_time' ]
    return results


@click.command( )
@click.option( "--path", type = click.Path( exists = True, file_okay = False, dir_okay = True ), required = True )
@click.option( '--test_names', type = str, multiple = True, required = True )
@click.option( '--output_dir', type = str, default = "figures_pretty", required = True )
def main( path: str, test_names: tuple[ str, ... ], output_dir: str ):
    os.makedirs( output_dir, exist_ok = True )
    # #################################################################################################################
    results = get_all_data_frames( path, test_names + ('baseline',) )
    results = merge_baseline( results, test_names )

    plt.rcParams.update( {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif",
        "font.serif": "Times",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "figure.labelsize": 12,
        "figure.titlesize": 12,
        "hatch.linewidth": 0.5, } )

    pretty_lbl = { "dynamo": "NVIDIA Dynamo", "llq": "LLQ", "roundrobin": "Round Robin", "hra": "AIScheduler", }
    colors = cycle( [ 'C3', 'C0', 'C2' ] )
    # fsize = (6.75, 3.5)
    fsize = (4, 3.5)
    # ftitle = rf"ShareGPT + Reasoning, {n} $\times$ NVIDIA A10 + vLLM, meta-llama/Meta-Llama-3-8B-Instruct"
    ftitle = rf"Cloud Setup with 4 $\times$ NVIDIA A10 + vLLM"

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = fsize )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'ttlt' ].mean( ) for df in results[ name ][ 1 ] ] )
        ax.semilogy( qpses,
                     stack_results,
                     marker = "s",
                     markersize = 3,
                     label = pretty_lbl[ name ],
                     color = next( colors ) )

    # ax.set_xlim( left = 0 )
    # ax.set_ylim( bottom = 0 )
    ax.set_yticks( [ 20, 30, 40, 60, 80, 100 ], labels = [ '20s', '30s', '40s', '60s', '80s', '100s' ] )
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Queries Per Second" )
    ax.set_ylabel( f"Average Response Time" )
    ax.set_title( ftitle )
    ax.legend( )

    fig.tight_layout( )
    fig.savefig( f"figures_pretty/request_e2e_time_avg.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_time_avg.pdf", dpi = 300, transparent = True )
    plt.close( fig )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = fsize )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'ttft' ].mean( ) for df in results[ name ][ 1 ] ] )
        ax.plot( qpses,
                 stack_results,
                 marker = "s",
                 markersize = 3,
                 label = pretty_lbl[ name ],
                 color = next( colors ) )

    # ax.set_xlim( left = 0 )
    # ax.set_ylim( bottom = -1, top = 20 )
    ax.set_yticks([0, 20, 40, 60, 80], labels = [ '0s', '20s', '40s', '60s', '80s' ])
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Queries Per Second" )
    ax.set_ylabel( f"Average Time to First Token" )
    ax.set_title( ftitle )
    ax.legend( )

    fig.tight_layout( )
    fig.savefig( f"figures_pretty/prefill_e2e_time_avg.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/prefill_e2e_time_avg.pdf", dpi = 300, transparent = True )
    plt.close( fig )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = fsize )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'request_e2e_slowdown' ].mean( ) for df in results[ name ][ 1 ] ] )
        print(qpses, stack_results)
        ax.plot( qpses,
                 stack_results,
                 marker = "s",
                 markersize = 3,
                 label = pretty_lbl[ name ],
                 color = next( colors ) )

    # ax.set_xlim( left = 0 )
    # ax.set_yticks([20, 30, 40, 60, 80, 100], labels = [ '20s', '30s', '40s', '60s', '80s', '100s' ])
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Queries Per Second" )
    ax.set_ylabel( f"Average Slowdown" )
    ax.set_title( ftitle )
    ax.legend( )

    ax.annotate( f"",
        xy = (4.75, 0.25),
        xytext = (4.75, 0.65),
        arrowprops = dict( arrowstyle = "->", facecolor = "black", lw = 1.5 ),
        xycoords = ("data", "axes fraction"),
        textcoords = ("data", "axes fraction"), )

    ax.text( 4.85, 0.48, "Better", ha = "left", va = "center", fontsize = 11, transform = ax.get_xaxis_transform( ) )

    fig.tight_layout( )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_avg.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_avg.pdf", dpi = 300, transparent = True )

    ax.set_ylim( bottom = -0.5, top = 11 )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_avg_zoom.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_avg_zoom.pdf", dpi = 300, transparent = True )
    plt.close( fig )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = fsize )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'request_e2e_slowdown' ].quantile( 0.95 ) for df in results[ name ][ 1 ] ] )
        ax.plot( qpses,
                 stack_results,
                 marker = "s",
                 markersize = 3,
                 label = pretty_lbl[ name ],
                 color = next( colors ) )

    # ax.set_xlim( left = 0 )
    # ax.set_yticks([20, 30, 40, 60, 80, 100], labels = [ '20s', '30s', '40s', '60s', '80s', '100s' ])
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Queries Per Second" )
    ax.set_ylabel( f"P95 of Slowdown Rate" )
    ax.set_title( ftitle )
    ax.legend( )

    fig.tight_layout( )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p95.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p95.pdf", dpi = 300, transparent = True )

    ax.set_ylim( bottom = -0.5, top = 11 )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p95_zoom.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p95_zoom.pdf", dpi = 300, transparent = True )
    plt.close( fig )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = fsize )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'request_e2e_slowdown' ].quantile( 0.99 ) for df in results[ name ][ 1 ] ] )
        ax.plot( qpses,
                 stack_results,
                 marker = "s",
                 markersize = 3,
                 label = pretty_lbl[ name ],
                 color = next( colors ) )

    # ax.set_xlim( left = 0 )
    # ax.set_ylim( top=11 )
    # ax.set_yticks([20, 30, 40, 60, 80, 100], labels = [ '20s', '30s', '40s', '60s', '80s', '100s' ])
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Queries Per Second" )
    ax.set_ylabel( f"P99 of Slowdown Rate" )
    ax.set_title( ftitle )
    ax.legend( )

    fig.tight_layout( )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p99.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p99.pdf", dpi = 300, transparent = True )

    ax.set_ylim( bottom = -1, top = 20 )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p99_zoom.png", dpi = 300, transparent = True )
    fig.savefig( f"figures_pretty/request_e2e_slowdown_p99_zoom.pdf", dpi = 300, transparent = True )
    plt.close( fig )

    # #################################################################################################################


if __name__ == "__main__":
    main( )
