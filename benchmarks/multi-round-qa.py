import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import openai
import pandas as pd
import numpy as np

from utils import AsyncLoopWrapper, init_logger

logger = init_logger( __name__, logging.INFO )


@dataclass
class WorkloadConfig:
    # Time gap between LLM response and next user request
    user_lag: float

    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Max length of the answer in one round
    max_output_len: int

    # Max length of the prompt in one round
    max_input_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Overall QPS
    qps: float

    # Model name
    model: str

    # Let response be exactly output_len
    ignore_eos: bool

    # Rate of inflation
    input_irate: float
    output_irate: float

    # Multiplier of inflation
    input_imult: int
    output_imult: int


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Max length of the answer in one round
    max_output_len: int

    # Max length of the prompt in one round
    max_input_len: int

    # Gap between two requests
    gap_between_requests: float

    # Num rounds
    num_rounds: int

    # Let response be exactly output_len
    ignore_eos: bool

    @staticmethod
    def new_user_config( user_id: int, workload_config: WorkloadConfig ) -> "UserConfig":
        return UserConfig( user_id = user_id,
                           system_prompt_len = workload_config.system_prompt_len,
                           user_info_len = workload_config.user_info_len,
                           max_input_len = workload_config.max_input_len,
                           max_output_len = workload_config.max_output_len,
                           gap_between_requests = workload_config.user_lag,
                           num_rounds = workload_config.num_rounds,
                           ignore_eos = workload_config.ignore_eos, )


class ChatHistory:

    def __init__( self, ):
        self.history = [ ]

    def on_user_query( self, query: str ):
        if len( self.history ) == 0:
            self.history.append( { "role": "user", "content": query } )
        else:
            assert self.history[ -1 ][ "role" ] == "assistant", "Expect system response"
            self.history.append( { "role": "user", "content": query } )

    def on_system_response( self, response: str ):
        assert len( self.history ) > 0, "Expect user query"
        assert self.history[ -1 ][ "role" ] == "user", "Expect user query"
        self.history.append( { "role": "assistant", "content": response } )

    def get_messages_for_openai( self ):
        return self.history

    def __len__( self ):
        return len( self.history )


@dataclass
class Response:
    body: str
    ttft: float
    ttlt: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float
    itl: list[ float ]


class RequestExecutor:

    def __init__( self, base_url: str, api_key: str, model: str ):
        self.client = openai.AsyncOpenAI( api_key = api_key, base_url = base_url )
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop( )
        self.request_history = [ ]

    async def _async_launch_request( self, messages, max_tokens, ignore_eos: bool, extra_headers = None ):
        start_time = time.time( )
        words = ""

        response = await self.client.chat.completions.create( messages = messages,
                                                              model = self.model,
                                                              temperature = 0,
                                                              stream = True,
                                                              max_tokens = max_tokens,
                                                              stream_options = { "include_usage": True },
                                                              extra_headers = extra_headers,
                                                              extra_body = { 'ignore_eos': ignore_eos },
                                                              timeout = 1000000000)

        itl = [ ]
        first_token_time = None
        # last_token_time = None
        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[ 0 ].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time( )
                # if last_token_time is None:
                #     last_token_time = time.time( )
                # else:
                #     new_token_time = time.time( )
                #     itl.append( new_token_time - last_token_time )
                #     last_token_time = new_token_time
                words += chunk_message
        time_end = time.time( )
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens

        return Response( body = words,
                         ttft = first_token_time - start_time,
                         ttlt = time_end - start_time,
                         generation_time = time_end - first_token_time,
                         prompt_tokens = tokens_prefill,
                         generation_tokens = tokens_out,
                         launch_time = start_time,
                         finish_time = time_end,
                         itl = itl, )

    def launch_request( self,
                        chat_history: ChatHistory,
                        max_tokens: int,
                        ignore_eos: bool,
                        finish_callback,
                        extra_headers = None, ):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai( )
        real_callback = lambda x: finish_callback( x.result( ) )
        future = asyncio.run_coroutine_threadsafe( self._async_launch_request( messages,
                                                                               max_tokens,
                                                                               ignore_eos,
                                                                               extra_headers ), self.loop )
        future.add_done_callback( real_callback )


class UserSession:

    def __init__( self, user_config: UserConfig, use_sharegpt = False, sharegpt_data = None ):
        self.user_config = user_config
        self.last_request_time = None
        self.last_response_time = None
        self.chat_history = ChatHistory( )
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data

        self.has_unfinished_request = False
        self.last_unfinished_log = 0

        self.prompt_lengths = [ ]
        self.generation_lengths = [ ]
        self.ttfts = [ ]
        self.ttlts = [ ]
        self.generation_times = [ ]
        self.launch_times = [ ]
        self.finish_times = [ ]
        self.itls = [ ]

        self.finished = False
        self.next_gap = self.user_config.gap_between_requests

    def _update_result( self, response: Response ):
        self.prompt_lengths.append( response.prompt_tokens )
        self.generation_lengths.append( response.generation_tokens )
        self.ttfts.append( response.ttft )
        self.ttlts.append( response.ttlt )
        self.generation_times.append( response.generation_time )
        self.launch_times.append( response.launch_time )
        self.finish_times.append( response.finish_time )
        self.itls.append( response.itl )

    def _build_system_prompt( self, sharegpt: bool ):
        if not sharegpt:
            dummy_text_sys = " ".join( [ "hi" ] * self.user_config.system_prompt_len )
            dummy_text_user = " ".join( [ "hi" ] * self.user_config.user_info_len )
            system_prompt = (
                    f"Hi, here's some system prompt: {dummy_text_sys}." + f"For user {self.user_config.user_id}, " + f"here are some other context: {dummy_text_user}.")
        else:
            system_prompt = ("You are a knowledgeable, efficient, and direct AI assistant. "
                             "Provide concise answers, focusing on the key information needed. "
                             "Offer suggestions tactfully when appropriate to improve outcomes. "
                             "Engage in productive collaboration with the user.\n\n")
        return system_prompt

    def _build_new_question( self, sharegpt: bool ):
        if not sharegpt:
            prompt = (
                    f"Here's question #{self.question_id}: can you tell me " + "a new long story with a happy ending?")
            num_tokens = 0  # For non-sharegpt questions, we don't track tokens
        else:
            prompt = self.sharegpt_data[ "conversations" ][ 2 * self.question_id ][ "value" ]
            num_tokens = self.sharegpt_data[ "conversations" ][ 2 * self.question_id ][ 'num_tokens' ]
            assert num_tokens <= self.user_config.max_input_len
        self.question_id += 1
        return prompt, num_tokens

    def _get_max_tokens( self, sharegpt: bool ):
        if not sharegpt:
            max_tokens = self.user_config.max_output_len
        else:
            prev_q_id = self.question_id - 1
            max_tokens = self.sharegpt_data[ "conversations" ][ 2 * prev_q_id + 1 ][ "num_tokens" ]
            max_tokens = min( max_tokens, self.user_config.max_output_len )
        return max_tokens

    def _launch_new_request( self, timestamp: float, request_executor: RequestExecutor ):
        prompt, question_tokens = self._build_new_question( sharegpt = self.use_sharegpt )
        if len( self.chat_history ) == 0:
            prompt = self._build_system_prompt( sharegpt = self.use_sharegpt ) + prompt
            total_tokens = 42 + question_tokens  # 42 tokens for system prompt
        else:
            total_tokens = 0
        self.chat_history.on_user_query( prompt )
        logger.debug( f"User {self.user_config.user_id} issues request {self.question_id}" )
        max_tokens = self._get_max_tokens( sharegpt = self.use_sharegpt )
        request_executor.launch_request( self.chat_history,
                                         max_tokens,
                                         self.user_config.ignore_eos,
                                         self._on_request_finished,
                                         extra_headers = { 
                                             "x-user-id": str( self.user_config.user_id ),
                                             "x-prefill-tokens": str(total_tokens)
                                         }, )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    def _on_request_finished( self, response: Response ):
        self.chat_history.on_system_response( response.body )
        self.has_unfinished_request = False
        self.last_response_time = response.finish_time
        logger.debug( f"User {self.user_config.user_id} finished one request. "
                      f"Prompt tokens: {response.prompt_tokens}, "
                      f"generation tokens: {response.generation_tokens}" )
        self._update_result( response )

    def set_internal_state( self, offset: float, timestamp: float ):
        """Tell the session is the 'offset' seconds after the start"""
        assert len( self.chat_history ) == 0, ("Internal state should be set " "before the first request")

        num_passed_questions = int( offset / self.user_config.gap_between_requests ) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        logger.debug( f"Set internal state for user {self.user_config.user_id}, "
                      f"question_id: {self.question_id}, "
                      f"last_request_time: {self.last_request_time}" )

    def step( self, timestamp: float, request_executor: RequestExecutor ):
        if self.question_id >= self.user_config.num_rounds and not self.has_unfinished_request:
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request( timestamp, request_executor )
            return

        if not self.has_unfinished_request and self.last_response_time + self.user_config.gap_between_requests < timestamp:
            self._launch_new_request( timestamp, request_executor )
            return

    def summary( self ) -> pd.DataFrame:
        df = pd.DataFrame( )
        df[ "prompt_tokens" ] = self.prompt_lengths
        df[ "generation_tokens" ] = self.generation_lengths
        df[ "ttft" ] = self.ttfts
        df[ "ttlt" ] = self.ttlts
        df[ "generation_time" ] = self.generation_times
        df[ "user_id" ] = self.user_config.user_id
        df[ "question_id" ] = range( 1, len( self.prompt_lengths ) + 1 )
        df[ "launch_time" ] = self.launch_times
        df[ "finish_time" ] = self.finish_times
        df[ "itls" ] = self.itls
        return df


class UserSessionManager:

    def __init__( self, workload_config: WorkloadConfig, init_user_id = 0, use_sharegpt = False ):
        self.workload_config = workload_config
        self.sessions = [ ]

        assert workload_config.qps > 0
        assert workload_config.num_rounds > 0

        self.gap_between_users = workload_config.num_rounds / workload_config.qps

        self.rng = np.random.RandomState( seed = 192 )
        self.sigma = 2.0
        self.mu = np.log( self.gap_between_users ) - self.sigma ** 2 / 2
        self.gap_gen = lambda: self.rng.lognormal( mean = self.mu, sigma = self.sigma )
        self.next_gap = self.gap_gen( )

        logger.info( f"Expected gap between users: {self.gap_between_users} secs.\n"
                     f"Gap between user response and request: {workload_config.user_lag} secs.\n" )

        self.user_id = init_user_id
        self.last_user_join = 0
        self.session_summaries = [ ]
        self.start_time = None

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data( )

    def _load_sharegpt_data( self ):
        with open( "ShareGPT.json", "r", encoding = "utf-8" ) as file:
            self.sharegpt_data = json.load( file )
        orig_len = len( self.sharegpt_data )
        self.sharegpt_data = [ d for d in self.sharegpt_data if d[ "num_round" ] >= 2 * self.workload_config.num_rounds ]
        self.sharegpt_data = [ d for d in self.sharegpt_data if all( a[ "num_tokens" ] <= self.workload_config.max_input_len for a in d[ 'conversations' ][ ::2 ] ) ]
        self.sharegpt_data = [ d for d in self.sharegpt_data if all( a[ "num_tokens" ] <= self.workload_config.max_output_len for a in d[ 'conversations' ][ 1::2 ] ) ]
        logger.info( f"There are {len( self.sharegpt_data )}/{orig_len} dataset entries with {self.workload_config.num_rounds} rounds." )
        rng = np.random.RandomState( seed = 151 )
        for q in self.sharegpt_data:
            for i, d in enumerate( q[ 'conversations' ] ):
                if i % 2 == 0:
                    if rng.random( ) < self.workload_config.input_irate:
                        max_mult = self.workload_config.max_input_len // d[ 'num_tokens' ]
                        max_mult = min( self.workload_config.input_imult, max_mult )
                        max_mult = max( max_mult, 1 )
                        d[ 'num_tokens' ] *= max_mult
                        assert d[ 'num_tokens' ] <= self.workload_config.max_input_len
                        d[ 'value' ] *= max_mult
                else:
                    if rng.random( ) < self.workload_config.output_irate:
                        d[ 'num_tokens' ] *= self.workload_config.output_imult
                        d[ 'value' ] *= self.workload_config.output_imult
        rng.shuffle( self.sharegpt_data )

    def _create_user_session( self ):
        self.user_id += 1
        user_config = UserConfig.new_user_config( self.user_id, self.workload_config )
        if self.use_sharegpt:
            user_session = UserSession( user_config, self.use_sharegpt, self.sharegpt_data[ self.user_id % len( self.sharegpt_data ) ] )
        else:
            user_session = UserSession( user_config, self.use_sharegpt )
        self.sessions.append( user_session )
        return user_session

    def _remove_finished_sessions( self ):
        sessions_to_remove = [ s for s in self.sessions if s.finished ]
        if len( sessions_to_remove ) > 0:
            logger.info( f"Removing {len( sessions_to_remove )} finished sessions, now "
                         f"active users: {len( self.sessions ) - len( sessions_to_remove )}" )
            for session in sessions_to_remove:
                self.session_summaries.append( session.summary( ) )
        self.sessions = [ s for s in self.sessions if not s.finished ]

    def step( self, timestamp: float, executor: RequestExecutor ) -> float:
        if self.start_time is None:
            self.start_time = timestamp

            self._create_user_session( )
            self.last_user_join = timestamp
            self.next_gap = self.gap_gen( )
            logger.info( f"Joined a new user {self.user_id}, "
                         f"now active users: {len( self.sessions )}" )

        if timestamp - self.last_user_join > self.next_gap:
            self._create_user_session( )
            self.last_user_join += self.next_gap
            self.next_gap = self.gap_gen( )
            logger.info( f"Joined a new user {self.user_id}, "
                         f"now active users: {len( self.sessions )}" )

        for session in self.sessions:
            session.step( timestamp, executor )

        self._remove_finished_sessions( )

        return self.next_gap

    @staticmethod
    def process_summary( df: pd.DataFrame,
                         start_time: Optional[ float ] = None,
                         end_time: Optional[ float ] = None,
                         pending_queries: int = 0,
                         qps: Optional[ float ] = None, ):
        if start_time and end_time:
            launched_queries = len( df.query( f"{start_time} <= launch_time <= {end_time}" ) )
            df = df.query( f"{start_time} <= finish_time <= {end_time}" )
        else:
            launched_queries = len( df )

        logger.debug( f"Launched queries: {launched_queries}, "
                      f"pending queries: {pending_queries}, "
                      f"finished queries: {len( df )}" )

        if qps is None:
            qps = 0.0

        if start_time is None:
            start_time = df[ "launch_time" ].min( )
        if end_time is None:
            end_time = df[ "finish_time" ].max( )
        total_time = end_time - start_time

        total_requests = launched_queries + pending_queries
        _qps = total_requests / total_time

        total_finished_requests = len( df )
        finished_qps = total_finished_requests / total_time

        total_prompt_tokens = df[ "prompt_tokens" ].sum( )
        total_generation_tokens = df[ "generation_tokens" ].sum( )
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_generation_speed_per_request = (df[ "generation_tokens" ] / df[ "generation_time" ]).mean( )
        average_ttft = df[ "ttft" ].mean( )
        logger.info( "Calculating performance summary" )
        print( "\n" )
        print( "==================== Performance summary ======================" )
        print( f"  \033[33mQPS: \033[32m{qps:.4f} reqs/s\033[0m\n" )

        print( f"  \033[33mProcessing speed: "
               f"\033[32m{finished_qps:.4f} reqs/s\033[0m\n" )

        print( f"  \033[33mRequests on-the-fly: {pending_queries}\033[0m\n" )

        print( "  \033[33mInput tokens per second: "
               f"\033[32m{average_prefill_speed:.4f} tokens/s\033[0m\n" )

        print( "  \033[33mOutput tokens per second: "
               f"\033[32m{average_generation_speed:.4f} tokens/s\033[0m\n" )

        print( "  \033[33mAverage generation throughput (per request): "
               f"\033[32m{average_generation_speed_per_request:.4f} "
               "tokens/req/s\033[0m\n" )

        print( f"  \033[33mAverage TTFT: \033[32m{average_ttft:.4f}s\033[0m\n" )

        print( f"Time range: {start_time} - {end_time} ({total_time:.2f}s)" )

        print( "===============================================================" )
        print( "\n" )
        return df

    def summary( self, start_time: float, end_time: float ) -> pd.DataFrame:
        if len( self.session_summaries ) == 0 and len( self.sessions ) == 0:
            return pd.DataFrame( )

        df = pd.concat( [ s for s in self.session_summaries ] + [ s.summary( ) for s in self.sessions ] )
        pending_queries = len( [ s for s in self.sessions if s.has_unfinished_request ] )
        start_time = max( self.start_time, start_time )
        end_time = min( end_time, df[ "finish_time" ].max( ) )
        qps = self.workload_config.qps

        df = UserSessionManager.process_summary( df, start_time, end_time, pending_queries, qps )
        return df


def parse_arguments( ) -> argparse.Namespace:
    parser = argparse.ArgumentParser( description = "Parse benchmark configurations." )

    parser.add_argument( "--sharegpt", action = "store_true", help = "Whether to use ShareGPT dataset" )
    parser.add_argument( "--max-input-len", type = int, required = True, help = "Max length of prompts", )
    parser.add_argument( "--max-output-len", type = int, required = True, help = "Max length of responses", )
    parser.add_argument( "--ignore-eos",
                         action = "store_true",
                         help = "Force response lengths to be exactly asnwer-len or shareGPT response lengths" )
    parser.add_argument( "--num-rounds", type = int, required = True, help = "Number of rounds in the conversation", )
    parser.add_argument( "--user-lag",
                         type = float,
                         required = True,
                         help = "Gap between LLM response and next user request", )
    parser.add_argument( "--qps", type = float, required = True, help = "Overall QPS" )
    parser.add_argument( "--model", type = str, required = True, help = "Model name" )
    parser.add_argument( "--base-url", type = str, required = True, help = "Base URL of the serving engine endpoint", )

    parser.add_argument( "--time", type = int, default = None, help = "The time to run the simulation in seconds", )
    parser.add_argument( "--shared-system-prompt",
                         type = int,
                         default = None,
                         help = "Length of the shared system prompt (tokens); this is ignored if using sharegpt", )
    parser.add_argument( "--user-history-prompt",
                         type = int,
                         default = None,
                         help = "Length of the user-specific history prompt (tokens); this is ignored if using sharegpt", )
    parser.add_argument( "--output",
                         type = str,
                         default = "summary.csv",
                         help = "The output file name (ended with csv or txt) "
                                "for the summary csv and txt", )
    parser.add_argument( "--init-user-id", type = int, default = 0, help = "The initial user id to start with" )
    parser.add_argument( "--log-interval",
                         type = int,
                         default = 30,
                         help = "The time between two summary loggings in seconds", )

    parser.add_argument( "--input-inflate-rate", type = float, default = 0, help = "Input rate of inflation", )
    parser.add_argument( "--output-inflate-rate", type = float, default = 0, help = "Output rate of inflation", )
    parser.add_argument( "--input-inflate-mult", type = int, default = 1, help = "Input inflation multiplier", )
    parser.add_argument( "--output-inflate-mult", type = int, default = 1, help = "Output inflation multiplier", )

    parser.add_argument( "--verbose", action = "store_true", help = "Whether to enable verbose logging" )
    args = parser.parse_args( )

    if not args.sharegpt:
        assert args.user_history_prompt is not None, "Must provide --user-history-prompt if not using ShareGPT"
        assert args.shared_system_prompt is not None, "Must provide --shared-system-prompt if not using ShareGPT"
        assert args.user_history_prompt + args.shared_system_prompt <= args.max_input_len

    assert args.input_inflate_rate >= 0
    assert args.output_inflate_rate >= 0
    assert 1 >= args.input_inflate_rate + args.output_inflate_rate

    if args.input_inflate_mult > 1 and args.input_inflate_rate > 0:
        assert args.sharegpt
    if args.output_inflate_mult > 1 and args.output_inflate_rate > 0:
        assert args.sharegpt
        assert args.ignore_eos

    return args


def parse_process_summary( ):
    parser = argparse.ArgumentParser( description = "Parse benchmark configurations.", add_help = False )

    parser.add_argument( "--process-summary", type = str, default = None )

    args, _ = parser.parse_known_args( )
    return args


def process_output( filename ):
    logger.warning( f"Processing the existing summary file {filename}"
                    ", ignoring all the other arguments" )
    UserSessionManager.process_summary( pd.read_csv( filename ), pending_queries = 0 )


def main( ):
    args = parse_process_summary( )
    if args.process_summary:
        process_output( args.process_summary )
        return

    args = parse_arguments( )
    if args.verbose:
        global logger
        logger = init_logger( __name__, level = logging.DEBUG )

    max_step_interval = 0.01
    min_step_interval = 0.001

    executor = RequestExecutor( base_url = args.base_url, api_key = "EMPTY", model = args.model )

    workload_config = WorkloadConfig( user_lag = args.user_lag,
                                      system_prompt_len = args.shared_system_prompt,
                                      user_info_len = args.user_history_prompt,
                                      max_input_len = args.max_input_len,
                                      max_output_len = args.max_output_len,
                                      num_rounds = args.num_rounds,
                                      qps = args.qps,
                                      model = args.model,
                                      ignore_eos = args.ignore_eos,
                                      input_irate = args.input_inflate_rate,
                                      output_irate = args.output_inflate_rate,
                                      input_imult = args.input_inflate_mult,
                                      output_imult = args.output_inflate_mult, )

    manager = UserSessionManager( workload_config, init_user_id = args.init_user_id, use_sharegpt = args.sharegpt )

    num_steps = 0
    start_time = time.time( )
    last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            next_t = manager.step( time.time( ), executor ) + time.time( )

            if time.time( ) - last_summary_time > args.log_interval:
                manager.summary( last_summary_time, time.time( ) )
                last_summary_time = time.time( )

            time.sleep( max( min( max_step_interval, next_t - time.time( ) - 0.005 ), min_step_interval ) )

            if args.time is not None and time.time( ) - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info( "Interrupted, waiting for the final result" )

    AsyncLoopWrapper.StopLoop( )

    logger.info( f"Finished benchmarking, dumping summary to {args.output}" )
    summary = manager.summary( 0, time.time( ) )
    summary.to_csv( args.output, index = False )


if __name__ == "__main__":
    main( )
