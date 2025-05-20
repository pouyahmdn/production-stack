import os
import click
import pandas as pd
from matplotlib import pyplot as plt
import re
import numpy as np


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
                if group[ 'base_execution_time' ].max( )/group[ 'base_execution_time' ].min( ) > 1.2:
                    print( f"Warning: Group variance {group[ 'base_execution_time' ].max( )} <-> {group[ 'base_execution_time' ].min( )} exceeds threshold" )
                return group.mean()

            df_b = df_b.groupby( [ 'prompt_tokens', 'generation_tokens' ] )[ df_b.columns ].apply( check_group_variance,
                                                                                                   include_groups = False ).reset_index(
                drop = True )

            new_df = pd.merge( dfs[ i ], df_b, on = [ 'prompt_tokens', 'generation_tokens' ], how = 'left' )
            if prev_rows != len( new_df ):
                print(f"Lost rows in {name}-{qpses[i]}:")
                print("Original rows not in merged result:")
                print(dfs[i][~dfs[i].index.isin(new_df.index)])
                print("\nMerged result rows not in original:")
                print(new_df[~new_df.index.isin(dfs[i].index)])
                raise ValueError( f"Lost rows during merge {name}-{qpses[i]}: {prev_rows} -> {len( new_df )}" )
            dfs[ i ] = new_df
            dfs[ i ][ 'request_e2e_slowdown' ] = dfs[ i ][ 'ttlt' ] / dfs[ i ][ 'base_execution_time' ]
    return results


@click.command( )
@click.option( "--path", type = click.Path( exists = True, file_okay = False, dir_okay = True ), required = True )
@click.option( '--test_names', type = str, multiple = True, required = True )
@click.option( '--output_dir', type = str, default = "figures", required = True )
def main( path: str, test_names: tuple[ str, ... ], output_dir: str ):
    os.makedirs( output_dir, exist_ok = True )
    # #################################################################################################################

    results = get_all_data_frames( path, test_names + ('baseline',) )
    results = merge_baseline( results, test_names )

    # Turn text.usetex off if you don't have a local LaTeX installation.
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

    for key, lbl in [ ('ttlt', 'Average Response Time (s)'), ('ttft', "Average Time to First Token (s)"),
                      ('generation_time', 'Average Generation Time (s)'),
                      ('request_e2e_slowdown', 'Request Slowdown Rate'),
                      ('prompt_tokens', "Average Number of Prompt Tokens"),
                      ('generation_tokens', 'Average Number of Generation Tokens'), ('question_id', 'Average Round') ]:
        fig, ax = plt.subplots( 1, 1, figsize = (6.75, 3.5) )
        for name in test_names:
            qpses = np.array( results[ name ][ 0 ] )
            stack_results = np.array( [ df[ key ].mean( ) for df in results[ name ][ 1 ] ] )
            ax.plot( qpses, stack_results, marker = "s", linewidth = 2, markersize = 5, label = name )

        # ax.set_xlim( left = 0 )
        # ax.set_ylim( bottom = 0 )
        ax.spines[ "right" ].set_visible( False )
        ax.spines[ "top" ].set_visible( False )
        ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
        ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
        ax.grid( True, alpha = 0.3 )
        ax.set_xlabel( "Targeted Queries Per Second" )
        ax.set_ylabel( lbl )
        ax.set_title( rf"1 round, ShareGPT, inflation in/out 5\%x10 (throttle at 4096/4096 tokens)" )
        plt.legend( loc = "best" )
        plt.savefig( f"{output_dir}/{key}.png", dpi = 300 )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = (6.75, 3.5) )
    mq = 0
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ len( df ) / (df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( )) for df in
                                    results[ name ][ 1 ] ] )
        ax.plot( qpses, stack_results, marker = "s", linewidth = 2, markersize = 5, label = name )

        mq = max( mq, max( qpses ) )

    ax.plot( [ 0, mq ], [ 0, mq ], '--', alpha = 0.3, color = 'k' )

    ax.set_xlim( left = 0 )
    # ax.set_ylim( bottom = 0 )
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Targeted Queries Per Second" )
    ax.set_ylabel( "True Queries Per Second" )
    ax.set_title( rf"1 round, ShareGPT, inflation in/out 5\%x10 (throttle at 4096/4096 tokens)" )
    plt.legend( loc = "best" )
    plt.savefig( f"{output_dir}/qps.png", dpi = 300 )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = (6.75, 3.5) )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'generation_tokens' ].sum( ) / (
                df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( )) for df in results[ name ][ 1 ] ] )
        ax.plot( qpses, stack_results, marker = "s", linewidth = 2, markersize = 5, label = name )

    ax.set_xlim( left = 0 )
    # ax.set_ylim( bottom = 0 )
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Targeted Queries Per Second" )
    ax.set_ylabel( "Generation Throughput (BUYER BEWARE)" )
    ax.set_title( rf"1 round, ShareGPT, inflation in/out 5\%x10 (throttle at 4096/4096 tokens)" )
    plt.legend( loc = "best" )
    plt.savefig( f"{output_dir}/out_thr.png", dpi = 300 )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = (6.75, 3.5) )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'prompt_tokens' ].sum( ) / (
                df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( )) for df in results[ name ][ 1 ] ] )
        ax.plot( qpses, stack_results, marker = "s", linewidth = 2, markersize = 5, label = name )

    ax.set_xlim( left = 0 )
    # ax.set_ylim( bottom = 0 )
    ax.spines[ "right" ].set_visible( False )
    ax.spines[ "top" ].set_visible( False )
    ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
    ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
    ax.grid( True, alpha = 0.3 )
    ax.set_xlabel( "Targeted Queries Per Second" )
    ax.set_ylabel( "Prefill Throughput (BUYER BEWARE)" )
    ax.set_title( rf"1 round, ShareGPT, inflation in/out 5\%x10 (throttle at 4096/4096 tokens)" )
    plt.legend( loc = "best" )
    plt.savefig( f"{output_dir}/in_thr.png", dpi = 300 )

    # #################################################################################################################


if __name__ == "__main__":
    main( )
