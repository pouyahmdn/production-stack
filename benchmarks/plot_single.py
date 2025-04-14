import os
import click
import pandas as pd
from matplotlib import pyplot as plt
import re
import numpy as np


@click.command( )
@click.option( "--path", type = click.Path( exists = True, file_okay = False, dir_okay = True ), required = True )
@click.option( '--test_names', type = str, multiple = True, required = True )
def main( path: str, test_names: list[ str ] ):
    os.makedirs( "figures", exist_ok = True )
    # #################################################################################################################

    for key, lbl in [ ('ttlt', 'Average Response Time (s)'), ('ttft', "Average Time to First Token (s)"),
                      ('generation_time', 'Average Generation Time (s)'),
                      ('prompt_tokens', "Average Number of Prompt Tokens"),
                      ('generation_tokens', 'Average Number of Generation Tokens'), ('question_id', 'Average Round') ]:
        fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )
        for name in test_names:
            qpses = [ ]
            stack_results = [ ]

            pattern = re.compile( rf"^{name}_output_(\d+(\.\d+)?)\.csv$" )

            for filename in os.listdir( path ):
                match = pattern.match( filename )
                if match:
                    qps = float( match.group( 1 ) )
                    df = pd.read_csv( os.path.join( path, filename ) )
                    q = round( qps, 1 )
                    qpses += [ q ]
                    print( name,
                           qps,
                           len( df ),
                           df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( ),
                           df[ 'prompt_tokens' ].mean( ),
                           df[ 'generation_tokens' ].mean( ),
                           df[ 'user_id' ].nunique( ),
                           df[ 'question_id' ].mean( ) )
                    stack_results.append( df[ key ].mean( ) )

            stack_results = np.array( stack_results )
            qpses = np.array( qpses )
            stack_results = stack_results[ np.argsort( qpses ) ]
            qpses = qpses[ np.argsort( qpses ) ]
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
        qpses = [ ]
        stack_results = [ ]

        pattern = re.compile( rf"^{name}_output_(\d+(\.\d+)?)\.csv$" )

        for filename in os.listdir( path ):
            match = pattern.match( filename )
            if match:
                qps = float( match.group( 1 ) )
                df = pd.read_csv( os.path.join( path, filename ) )
                q = round( qps, 1 )
                qpses += [ q ]
                stack_results.append( len( df ) / (df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( )) )

        stack_results = np.array( stack_results )
        qpses = np.array( qpses )
        stack_results = stack_results[ np.argsort( qpses ) ]
        qpses = qpses[ np.argsort( qpses ) ]
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
        qpses = [ ]
        stack_results = [ ]

        pattern = re.compile( rf"^{name}_output_(\d+(\.\d+)?)\.csv$" )

        for filename in os.listdir( path ):
            match = pattern.match( filename )
            if match:
                qps = float( match.group( 1 ) )
                df = pd.read_csv( os.path.join( path, filename ) )
                q = round( qps, 1 )
                qpses += [ q ]
                stack_results.append(
                    df[ 'generation_tokens' ].sum( ) / (df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( )) )

        stack_results = np.array( stack_results )
        qpses = np.array( qpses )
        stack_results = stack_results[ np.argsort( qpses ) ]
        qpses = qpses[ np.argsort( qpses ) ]
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
        qpses = [ ]
        stack_results = [ ]

        pattern = re.compile( rf"^{name}_output_(\d+(\.\d+)?)\.csv$" )

        for filename in os.listdir( path ):
            match = pattern.match( filename )
            if match:
                qps = float( match.group( 1 ) )
                df = pd.read_csv( os.path.join( path, filename ) )
                q = round( qps, 1 )
                qpses += [ q ]
                stack_results.append(
                    df[ 'prompt_tokens' ].sum( ) / (df[ 'launch_time' ].max( ) - df[ 'launch_time' ].min( )) )

        stack_results = np.array( stack_results )
        qpses = np.array( qpses )
        stack_results = stack_results[ np.argsort( qpses ) ]
        qpses = qpses[ np.argsort( qpses ) ]
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
