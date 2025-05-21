import os
import click
import pandas as pd
from matplotlib import pyplot as plt
import re
import numpy as np


def get_all_data_frames_vidur( path: str, names: list[ str ] ) -> dict[
    str, tuple[ list[ float ], list[ pd.DataFrame ] ] ]:
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
                if qps >= 12.0:
                    continue
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


def get_all_data_frames_ps( path: str, names: list[ str ] ) -> dict[
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
                if qps >= 12.0:
                    continue
                df = pd.read_csv( os.path.join( path, filename ) )
                df.rename( columns = {
                    'launch_time': 'arrival_time',
                    'ttlt': 'request_e2e_time',
                    'ttft': 'prefill_e2e_time',
                    'generation_time': 'decode_e2e_time',
                    'prompt_tokens': "request_num_prefill_tokens",
                    'generation_tokens': 'request_num_decode_tokens', }, inplace=True )
                qpses += [ q ]
                dfs += [ df ]
        dfs = [ x for _, x in sorted( zip( qpses, dfs ) ) ]
        qpses = sorted( qpses )
        results[ name ] = (qpses, dfs)
    return results


@click.command( )
@click.option( "--path_vidur", type = click.Path( exists = True, file_okay = False, dir_okay = True ), required = True )
@click.option( '--test_names_vidur', type = str, multiple = True, required = True )
@click.option( "--path_ps", type = click.Path( exists = True, file_okay = False, dir_okay = True ), required = True )
@click.option( '--test_names_ps', type = str, multiple = True, required = True )
@click.option( '--label_names', type = str, multiple = True, required = True )
def main( path_vidur: str, test_names_vidur: list[ str ], path_ps: str, test_names_ps: list[ str ], label_names: list[str] ):
    os.makedirs( "figures", exist_ok = True )
    # #################################################################################################################

    res = {}
    res['vidur'] = get_all_data_frames_vidur( path_vidur, test_names_vidur )
    res['ps'] = get_all_data_frames_ps( path_ps, test_names_ps )

    for key, lbl in [ ('request_e2e_time', 'Average Response Time (s)'),
                      ('prefill_e2e_time', "Average Time to First Token (s)"),
                      ('decode_e2e_time', 'Average Generation Time (s)'),
                      ('request_num_prefill_tokens', "Average Number of Prompt Tokens"),
                      ('request_num_decode_tokens', 'Average Number of Generation Tokens'), ]:
        fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )
        groups = []
        for name_vidur, name_ps, true_name in zip(test_names_vidur, test_names_ps, label_names):
            for qps_vidur, df_vidur, qps_ps, df_ps in zip(res['vidur'][name_vidur][0], res['vidur'][name_vidur][1], res['ps'][name_ps][0], res['ps'][name_ps][1]):
                assert qps_ps == qps_vidur
                groups.append((true_name, qps_ps, df_vidur[ key ], df_ps[key]))
        groups = sorted( groups, key = lambda x: x[1] )

        wd = 1
        dist = 4
        bars = sum(([g[2].mean(), g[3].mean()] for g in groups), [])
        x_s = sum(([2 * i * wd + i * dist, (2 * i+1) * wd + i * dist] for i, _ in enumerate(groups)), [])

        ax.bar(x_s[0::2], bars[0::2], width = wd, color='C1', edgecolor='k', label='Vidur', alpha=0.7)
        ax.bar(x_s[1::2], bars[1::2], width = wd, color='C2', edgecolor='k', label='Production Stack', alpha=0.7)

        grp_names = [f"{g[0]}, QPS {g[1]}" for g in groups]
        x_s = [(2 * i + 0.5) * wd + i * dist for i, _ in enumerate(groups)]

        ax.set_xticks(x_s)
        ax.set_xticklabels( grp_names, rotation=20 )

        ax.spines[ "right" ].set_visible( False )
        ax.spines[ "top" ].set_visible( False )
        ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
        ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
        ax.grid( True, alpha = 0.3 )
        ax.set_ylabel( lbl )
        ax.set_title( f"1 round, ShareGPT, inflation in/out 5%x10 (throttle at 4096/4096 tokens)" )
        plt.legend( loc = "best" )
        plt.savefig( f"figures/compare_{key}_avg.png", dpi = 300 )

        fig, ax = plt.subplots( 1, 1, figsize = (10, 5) )

        for i, (name, qps, df_vidur, df_ps) in enumerate(groups):
            ax.semilogx(np.sort(df_ps), np.linspace(0, 100, len(df_ps)), color=f'C{i}', linestyle='-', label=f'{name}, {qps}, Production Stack')
            ax.semilogx(np.sort(df_vidur), np.linspace(0, 100, len(df_vidur)), color=f'C{i}', linestyle='--', label=f'{name}, {qps}, Vidur')

        ax.spines[ "right" ].set_visible( False )
        ax.spines[ "top" ].set_visible( False )
        ax.plot( 1, 0, ">k", transform = ax.transAxes, clip_on = False )
        ax.plot( 0, 1, "^k", transform = ax.transAxes, clip_on = False )
        ax.grid( True, alpha = 0.3 )
        ax.set_xlabel( lbl )
        ax.set_ylabel( "Percentile (%)" )
        ax.set_title( f"1 round, ShareGPT, inflation in/out 5%x10 (throttle at 4096/4096 tokens)" )
        plt.legend( loc = "best", fontsize=6 )
        plt.savefig( f"figures/compare_{key}_cdf.png", dpi = 300 )

    # #################################################################################################################


if __name__ == "__main__":
    main( )
