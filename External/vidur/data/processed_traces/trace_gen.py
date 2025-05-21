import argparse
import json
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class WorkloadConfig:
    # Max length of the answer in one round
    max_output_len: int

    # Max length of the prompt in one round
    max_input_len: int

    # Overall QPS
    qps: float

    # Rate of inflation
    input_irate: float
    output_irate: float

    # Multiplier of inflation
    input_imult: int
    output_imult: int


class UserSessionManager:

    def __init__( self, workload_config: WorkloadConfig ):
        self.workload_config = workload_config

        assert workload_config.qps > 0

        self.gap_between_users = 1 / workload_config.qps

        self.rng = np.random.RandomState( seed = 192 )
        self.sigma = 2.0
        self.mu = np.log( self.gap_between_users ) - self.sigma ** 2 / 2
        self.gap_gen = lambda: self.rng.lognormal( mean = self.mu, sigma = self.sigma )

        print( f"Expected gap between users: {self.gap_between_users} secs." )

        self.user_id = 0
        self.trace = [ ]
        self.next_gap = None

        self._load_sharegpt_data( )

    def _load_sharegpt_data( self ):
        with open( "ShareGPT.json", "r", encoding = "utf-8" ) as file:
            self.sharegpt_data = json.load( file )
        orig_len = len( self.sharegpt_data )
        self.sharegpt_data = [ d for d in self.sharegpt_data if d[ "num_round" ] >= 2 ]
        print( f"There are {len( self.sharegpt_data )}/{orig_len} dataset entries with 2 rounds." )
        self.sharegpt_data = [ d for d in self.sharegpt_data if all(
            a[ "num_tokens" ] <= self.workload_config.max_input_len for a in d[ 'conversations' ][ ::2 ] ) ]
        print( f"There are {len( self.sharegpt_data )}/{orig_len} dataset entries with acceptable input size." )
        self.sharegpt_data = [ d for d in self.sharegpt_data if all(
            a[ "num_tokens" ] <= self.workload_config.max_output_len for a in d[ 'conversations' ][ 1::2 ] ) ]
        print( f"There are {len( self.sharegpt_data )}/{orig_len} dataset entries with acceptable output size." )
        rng = np.random.RandomState( seed = 151 )
        for q in self.sharegpt_data:
            for i, d in enumerate( q[ 'conversations' ] ):
                if i % 2 == 0:
                    if rng.random( ) < self.workload_config.input_irate:
                        max_mult = self.workload_config.max_input_len // d[ 'num_tokens' ]
                        max_mult = min( self.workload_config.input_imult, max_mult )
                        max_mult = max( max_mult, 1 )
                        assert max_mult > 0
                        d[ 'num_tokens' ] *= max_mult
                        assert d[ 'num_tokens' ] <= self.workload_config.max_input_len
                        d[ 'value' ] *= max_mult

                    d[ 'num_tokens' ] += 42
                    d[ 'value' ] = ('You are a knowledgeable, efficient, and direct AI assistant. '
                                    'Provide concise answers, focusing on the key information needed. '
                                    'Offer suggestions tactfully when appropriate to improve outcomes. '
                                    'Engage in productive collaboration with the user.\n\n') + d[ 'value' ]
                else:
                    if rng.random( ) < self.workload_config.output_irate:
                        d[ 'num_tokens' ] *= self.workload_config.output_imult
                        d[ 'num_tokens' ] = min( d[ 'num_tokens' ], self.workload_config.max_output_len )
                        d[ 'value' ] *= self.workload_config.output_imult
        rng.shuffle( self.sharegpt_data )

    def step( self ):
        self.trace.append( (self.get_next_time( ),
                            self.sharegpt_data[ self.user_id ][ 'conversations' ][ 0 ][ 'num_tokens' ],
                            self.sharegpt_data[ self.user_id ][ 'conversations' ][ 1 ][ 'num_tokens' ]) )
        self.next_gap = self.gap_gen( )
        self.user_id += 1
        self.user_id %= len( self.sharegpt_data )

    def get_next_time( self ) -> float:
        if len( self.trace ) == 0:
            return 0
        else:
            return self.trace[ -1 ][ 0 ] + self.next_gap

    def summary( self ) -> pd.DataFrame:
        df = pd.DataFrame( self.trace, columns = [ 'arrived_at', 'num_prefill_tokens', 'num_decode_tokens' ] )
        return df


def parse_arguments( ) -> argparse.Namespace:
    parser = argparse.ArgumentParser( description = "Parse benchmark configurations." )

    parser.add_argument( "--max-input-len", type = int, required = True, help = "Max length of prompts" )
    parser.add_argument( "--max-output-len", type = int, required = True, help = "Max length of responses" )
    parser.add_argument( "--qps", type = float, required = True, help = "Overall QPS" )
    parser.add_argument( "--time", type = int, required = True, help = "The time to run the simulation in seconds" )
    parser.add_argument( "--input-inflate-rate", type = float, required = True, help = "Input rate of inflation" )
    parser.add_argument( "--output-inflate-rate", type = float, required = True, help = "Output rate of inflation" )
    parser.add_argument( "--input-inflate-mult", type = int, required = True, help = "Input inflation multiplier" )
    parser.add_argument( "--output-inflate-mult", type = int, required = True, help = "Output inflation multiplier" )
    parser.add_argument( "--output", type = str, required = True, help = "The output file name for the trace csv" )

    args = parser.parse_args( )

    assert args.input_inflate_rate >= 0
    assert args.output_inflate_rate >= 0
    assert 1 >= args.input_inflate_rate + args.output_inflate_rate

    return args


def main( ):
    args = parse_arguments( )

    workload_config = WorkloadConfig( max_input_len = args.max_input_len,
                                      max_output_len = args.max_output_len,
                                      qps = args.qps,
                                      input_irate = args.input_inflate_rate,
                                      output_irate = args.output_inflate_rate,
                                      input_imult = args.input_inflate_mult,
                                      output_imult = args.output_inflate_mult, )

    manager = UserSessionManager( workload_config )

    while manager.get_next_time( ) < args.time:
        manager.step( )

    manager.summary( ).to_csv( args.output, index = False )
    print( f"{len( manager.trace )} requests in {args.time}, equivalent to {len( manager.trace ) / args.time:.2f} QPS" )


if __name__ == "__main__":
    main( )
