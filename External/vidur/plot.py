import os
import click
import pandas as pd
from matplotlib import pyplot as plt
import re
import numpy as np


def get_all_data_frames( path: str, names: list[ str ] ) -> dict[ str, tuple[ list[ float ], list[ pd.DataFrame ] ] ]:
    results = { }
    for name in names:
        qpses = [ ]
        dfs = [ ]
        pattern = re.compile( rf"^{name}_(\d+(\.\d+)?)$" )
        for subdir in os.listdir( path ):
            match = pattern.match( subdir )
            if match:
                dirs_in_subdir = os.listdir( f"{path}/{subdir}" )
                assert len( dirs_in_subdir ) == 1
                qps = float( match.group( 1 ) )
                df = pd.read_csv( f"{path}/{subdir}/{dirs_in_subdir[ 0 ]}/request_metrics.csv" )
                df[ 'decode_e2e_time' ] = df[ 'request_e2e_time' ] - df[ 'prefill_e2e_time' ]
                df[ 'arrival_time' ] = df[ 'request_inter_arrival_delay' ].cumsum( )
                q = round( qps, 1 )
                qpses += [ q ]
                dfs += [ df ]
        dfs = [ x for _, x in sorted( zip( qpses, dfs ) ) ]
        qpses = sorted( qpses )
        results[ name ] = (qpses, dfs)
    return results


@click.command( )
@click.option( "--path", type = click.Path( exists = True, file_okay = False, dir_okay = True ), required = True )
@click.option( '--test_names', type = str, multiple = True, required = True )
def main( path: str, test_names: list[ str ] ):
    os.makedirs( "figures", exist_ok = True )
    # #################################################################################################################

    results = get_all_data_frames( path, test_names )

    for key, lbl in [ ('request_e2e_time', 'Average Response Time (s)'),
                      ('prefill_e2e_time', "Average Time to First Token (s)"),
                      ('decode_e2e_time', 'Average Generation Time (s)'),
                      ('request_num_prefill_tokens', "Average Number of Prompt Tokens"),
                      ('request_num_decode_tokens', 'Average Number of Generation Tokens'), ]:
        fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )
        for name in test_names:
            qpses = np.array( results[ name ][ 0 ] )
            stack_results = np.array( [ df[ key ].mean( ) for df in results[ name ][ 1 ] ] )
            ax.plot( qpses, stack_results, marker = "s", linewidth = 2, markersize = 5, label = name )

        ax.set_xlim( left = 0 )
        # ax.set_ylim( bottom = 0 )
        ax.spines[ "right" ].set_visible( False )
        ax.spines[ "top" ].set_visible( False )
        ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
        ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
        ax.grid( True, alpha = 0.3 )
        ax.set_xlabel( "Targeted Queries Per Second" )
        ax.set_ylabel( lbl )
        ax.set_title( f"1 round, ShareGPT, inflation in/out 5%x10 (throttle at 4096/4096 tokens)" )
        plt.legend( loc = "best" )
        plt.savefig( f"figures/{key}.png", dpi = 300 )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )
    mq = 0
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ len( df ) / (df[ 'arrival_time' ].max( ) - df[ 'arrival_time' ].min( )) for df in
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
    ax.set_title( f"1 round, ShareGPT, inflation in/out 5%x10 (throttle at 4096/4096 tokens)" )
    plt.legend( loc = "best" )
    plt.savefig( "figures/qps.png", dpi = 300 )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'request_num_decode_tokens' ].sum( ) / (
                df[ 'arrival_time' ].max( ) - df[ 'arrival_time' ].min( )) for df in results[ name ][ 1 ] ] )
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
    ax.set_title( f"1 round, ShareGPT, inflation in/out 5%x10 (throttle at 4096/4096 tokens)" )
    plt.legend( loc = "best" )
    plt.savefig( "figures/out_thr.png", dpi = 300 )

    # #################################################################################################################

    fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )
    for name in test_names:
        qpses = np.array( results[ name ][ 0 ] )
        stack_results = np.array( [ df[ 'request_num_prefill_tokens' ].sum( ) / (
                df[ 'arrival_time' ].max( ) - df[ 'arrival_time' ].min( )) for df in results[ name ][ 1 ] ] )
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
    ax.set_title( f"1 round, ShareGPT, inflation in/out 5%x10 (throttle at 4096/4096 tokens)" )
    plt.legend( loc = "best" )
    plt.savefig( "figures/in_thr.png", dpi = 300 )

    # #################################################################################################################


if __name__ == "__main__":
    main( )
