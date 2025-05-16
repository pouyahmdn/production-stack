from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple
import math

from vllm_router.log import init_logger

# Constants for vLLM engine configuration
BLOCK_SIZE = 16  # Block size of vLLM engines
TOTAL_NUMBER_OF_BLOCKS = 2756  # Total number of blocks available on vLLM engines with A10 GPUs
DECODE_TO_PREFILL_RATIO = 0.6   # avg decode/prompt tokens
SAFETY_FRACTION = 0.03  # keep last 3 % blocks free

logger = init_logger( __name__ )


class SingletonMeta( type ):
    _instances = { }

    def __call__( cls, *args, **kwargs ):
        if cls not in cls._instances:
            instance = super( ).__call__( *args, **kwargs )
            cls._instances[ cls ] = instance
        return cls._instances[ cls ]


@dataclass
class RequestStats:
    # Number of queries per second
    qps: float
    # Average time-to-first-token (TTFT) in seconds
    ttft: float
    # Total number of requests during prefilling
    in_prefill_requests: int
    # Total number of requests during decoding
    in_decoding_requests: int
    #
    ts_prefill_enqueue: list[ float ]
    #
    ts_decoding_enqueue: list[ float ]
    # Total number of requests finished
    finished_requests: int
    # How long the engine has been serving requests (uptime)
    uptime: int
    # Average decoding length (time from first token to completion)
    avg_decoding_length: float
    # Average overall latency (from request arrival to completion)
    avg_latency: float
    # Average inter-token latency (if available; default -1 if not computed)
    avg_itl: float
    # Number of swapped requests (moved from GPU to CPU)
    num_swapped_requests: int


class MovingAverageMonitor:
    """
    Monitors the average of values in a sliding window.
    """

    def __init__( self, sliding_window_size: float ):
        self.sliding_window_size = sliding_window_size
        self.timestamps: Deque[ float ] = deque( )
        self.values: Deque[ float ] = deque( )

    def update( self, timestamp: float, value: float ):
        """
        Update the throughput monitor with a new timestamp

        Args:
            timestamp: The timestamp of the data point.
            value: The value of the data point.

        This method adds the new data point to the sliding window and
        removes any data point that is older than the sliding window size.
        """
        self.timestamps.append( timestamp )
        self.values.append( value )
        while self.timestamps and self.timestamps[ 0 ] < timestamp - self.sliding_window_size:
            self.timestamps.popleft( )
            self.values.popleft( )

    def update_no_value( self, timestamp: float ):
        """
        Update the throughput monitor with a new timestamp with no value
        """
        while len( self.timestamps ) > 0 and self.timestamps[ 0 ] < timestamp - self.sliding_window_size:
            self.timestamps.popleft( )
            self.values.popleft( )

    def get_average( self ) -> float:
        return sum( self.values ) / len( self.values ) if self.values else -1

    def get_sum( self ) -> float:
        return sum( self.values )


class RequestStatsMonitor( metaclass = SingletonMeta ):
    """
    Monitors the request statistics of all serving engines.
    """

    # NOTE (ApostaC): Currently, QPS is calculated based on the number of
    # arrived requests in the sliding window, but the inter_token_latency and
    # ttft are calculated based on the number of completed requests in the
    # sliding window.
    def __init__( self, sliding_window_size: float = None ):
        if hasattr( self, "_initialized" ):
            return
        if sliding_window_size is None:
            raise ValueError( "RequestStatsMonitor must be initialized with sliding_window_size" )
        self.sliding_window_size = sliding_window_size

        self.qps_monitors: Dict[ str, MovingAverageMonitor ] = { }
        self.ttft_monitors: Dict[ str, MovingAverageMonitor ] = { }
        self.latency_monitors: Dict[ str, MovingAverageMonitor ] = { }
        self.decoding_length_monitors: Dict[ str, MovingAverageMonitor ] = { }

        # The time when the request is coming request_id -> timestamp
        self.request_arrival_time: Dict[str, float] = { }
        # Record time when first token is received: (engine_url, request_id) -> timestamp
        self.first_token_time: Dict[ Tuple[ str, str ], float ] = { }

        # Number of requests in different stages (from the start of the router)
        self.in_prefill_requests_ids: Dict[ str, set[ str ] ] = { }
        self.in_decoding_requests_ids: Dict[ str, set[ str ] ] = { }
        self.finished_requests: Dict[ str, int ] = { }
        # Track tokens for each request in decoding: engine_url -> {request_id -> token_count}
        self.request_decode_tokens: Dict[ str, Dict[ str, int ] ] = { }
        # Track prefill tokens for each request: engine_url -> {request_id -> token_count}
        self.request_prefill_tokens: Dict[ str, Dict[ str, int ] ] = { }

        # Counter for swapped requests
        self.swapped_requests: Dict[ str, int ] = { }

        self.first_query_time: float = None
        self._initialized = True

    def on_request_arrival(self, request_id: str, timestamp: float):
        """
        Record the initial arrival of a request before it starts processing.

        Args:
            request_id: The global request ID
            timestamp: the timestamp when the request arrived
        """
        self.request_arrival_time[request_id] = timestamp

        if self.first_query_time is None:
            self.first_query_time = timestamp

    def on_request_start(self, engine_url: str, request_id: str, timestamp: float):
        """
        Tell the monitor that a new request has been sent to the engine.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: the timestamp when the request was sent to the engine
        """

        if engine_url not in self.qps_monitors:
            self.qps_monitors[engine_url] = MovingAverageMonitor(self.sliding_window_size)
        self.qps_monitors[engine_url].update(timestamp, 1)

    def on_request_routed(self, engine_url: str, request_id: str, prefill_tokens: int):
        """
        Update the prefill tokens tracking after HRA (Head-Room Admission) decision.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            prefill_tokens: number of tokens in the prefill phase
        """
        # Initialize prefill tokens tracking
        if engine_url not in self.request_prefill_tokens:
            self.request_prefill_tokens[engine_url] = {}
        self.request_prefill_tokens[engine_url][request_id] = prefill_tokens
        logger.debug(f"Initialized prefill token count for request {request_id} on {engine_url}: {prefill_tokens} tokens")

        if engine_url not in self.in_prefill_requests_ids:
            self.in_prefill_requests_ids[engine_url] = set()
        self.in_prefill_requests_ids[engine_url].add(request_id)

    def on_request_kill( self, engine_url: str, request_id: str ):
        if request_id in self.request_arrival_time:
            logger.debug( f"Kill request for {request_id} removed request_arrival_time entry..." )
            self.in_prefill_requests_ids[ engine_url ].discard( request_id )
            del self.request_arrival_time[request_id]
        if (engine_url, request_id) in self.first_token_time:
            logger.debug( f"Kill request for ({engine_url}, {request_id}) removed first_token_time entry..." )
            self.in_decoding_requests_ids[ engine_url ].discard( request_id )
            del self.first_token_time[(engine_url, request_id)]
        if engine_url in self.request_decode_tokens and request_id in self.request_decode_tokens[engine_url]:
            logger.debug( f"Kill request for ({engine_url}, {request_id}) removed request_decode_tokens entry..." )
            del self.request_decode_tokens[engine_url][request_id]
            # Clean up empty engine dict if no more requests
            if not self.request_decode_tokens[engine_url]:
                del self.request_decode_tokens[engine_url]
        if engine_url in self.request_prefill_tokens and request_id in self.request_prefill_tokens[engine_url]:
            logger.debug( f"Kill request for ({engine_url}, {request_id}) removed request_prefill_tokens entry..." )
            del self.request_prefill_tokens[engine_url][request_id]
            # Clean up empty engine dict if no more requests
            if not self.request_prefill_tokens[engine_url]:
                del self.request_prefill_tokens[engine_url]

    def on_request_response( self, engine_url: str, request_id: str, timestamp: float, is_first_token: bool = True):
        """
        Tell the monitor that a response token has been received for a request.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: The timestamp when the response token was received
        """
        # Initialize or increment token count for this request
        if engine_url not in self.request_decode_tokens:
            self.request_decode_tokens[engine_url] = {}
        if request_id not in self.request_decode_tokens[engine_url]:
            self.request_decode_tokens[engine_url][request_id] = 0
        self.request_decode_tokens[engine_url][request_id] += 1
        logger.debug(f"Updated token count for request {request_id} on {engine_url}: {self.request_decode_tokens[engine_url][request_id]} tokens")
        if not is_first_token:
            return
        
        if request_id not in self.request_arrival_time:
            logger.debug( f"Something weird happened; we needed {request_id} in request_arrival_time but it wasn't there" )
            self.on_request_kill(engine_url, request_id)
            return

        self.in_prefill_requests_ids[ engine_url ].discard( request_id )

        if engine_url not in self.in_decoding_requests_ids:
            self.in_decoding_requests_ids[ engine_url ] = set( )
        self.in_decoding_requests_ids[ engine_url ].add( request_id )

        # Record first token time (do not pop so we can compute overall latency later)
        self.first_token_time[ (engine_url, request_id) ] = timestamp

        # Update TTFT as time from request start to first token
        if engine_url not in self.ttft_monitors:
            self.ttft_monitors[ engine_url ] = MovingAverageMonitor( self.sliding_window_size )
        ttft = timestamp - self.request_arrival_time[request_id]
        self.ttft_monitors[ engine_url ].update( timestamp, ttft )

    def on_request_complete( self, engine_url: str, request_id: str, timestamp: float ):
        """
        Tell the monitor that a request has been completed.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: The timestamp when the request was completed
        """
        if request_id not in self.request_arrival_time:
            logger.debug( f"Something weird happened; we needed {request_id} in request_arrival_time but it wasn't there" )
            self.on_request_kill(engine_url, request_id)
            return
        if (engine_url, request_id) not in self.first_token_time:
            logger.debug( f"Something weird happened; we needed ({engine_url}, {request_id}) in first_token_time but it wasn't there" )
            self.on_request_kill(engine_url, request_id)
            return

        self.in_decoding_requests_ids[ engine_url ].discard( request_id )

        if engine_url not in self.finished_requests:
            self.finished_requests[ engine_url ] = 0
        self.finished_requests[ engine_url ] += 1

        if engine_url not in self.latency_monitors:
            self.latency_monitors[ engine_url ] = MovingAverageMonitor( self.sliding_window_size )
        lat = timestamp - self.request_arrival_time[request_id]
        self.latency_monitors[ engine_url ].update( timestamp, lat )

        if engine_url not in self.decoding_length_monitors:
            self.decoding_length_monitors[ engine_url ] = MovingAverageMonitor( self.sliding_window_size )
        dec_lat = timestamp - self.first_token_time[ (engine_url, request_id) ]
        self.decoding_length_monitors[ engine_url ].update( timestamp, dec_lat )

        # Log final token counts before cleanup
        if engine_url in self.request_decode_tokens and request_id in self.request_decode_tokens[engine_url]:
            logger.info(f"Request {request_id} on {engine_url} completed with {self.request_decode_tokens[engine_url][request_id]} decode tokens")
            del self.request_decode_tokens[engine_url][request_id]
            # Clean up empty engine dict if no more requests
            if not self.request_decode_tokens[engine_url]:
                del self.request_decode_tokens[engine_url]

        if engine_url in self.request_prefill_tokens and request_id in self.request_prefill_tokens[engine_url]:
            logger.info(f"Request {request_id} on {engine_url} completed with {self.request_prefill_tokens[engine_url][request_id]} prefill tokens")
            del self.request_prefill_tokens[engine_url][request_id]
            # Clean up empty engine dict if no more requests
            if not self.request_prefill_tokens[engine_url]:
                del self.request_prefill_tokens[engine_url]

        del self.request_arrival_time[request_id]
        del self.first_token_time[ (engine_url, request_id) ]

    def on_request_swapped( self, engine_url: str, request_id: str, timestamp: float ):
        # This function should be called if a request is determined to be swapped from GPU to CPU.
        """
        Tell the monitor that a request has been swapped from GPU to CPU.

        Args:
            engine_url: The URL of the serving engine
            request_id: The global request ID
            timestamp: The timestamp when the request was swapped
        """
        if engine_url not in self.swapped_requests:
            self.swapped_requests[ engine_url ] = 0
        self.swapped_requests[ engine_url ] += 1

    def get_request_stats( self, current_time: float ) -> Dict[ str, RequestStats ]:
        """
        Get the request statistics for each serving engine

        Args:
            current_time: The current timestamp in seconds

        Returns:
            A dictionary where the key is the serving engine URL and the value
            is the request statistics for that engine.
            The TTFT and inter token latency will be -1 if there is no requests
            finished in the sliding window.
        """
        ret = { }
        urls = set( self.in_prefill_requests_ids.keys( ) ).union( set( self.in_decoding_requests_ids.keys( ) ) )
        for engine_url in urls:
            if engine_url not in self.qps_monitors:
                qps = -1
            else:
                # Update the monitors
                self.qps_monitors[ engine_url ].update_no_value( current_time )
                qps = self.qps_monitors[ engine_url ].get_sum( ) / self.sliding_window_size

            if engine_url not in self.ttft_monitors:
                ttft = -1
            else:
                # Update the monitors
                self.ttft_monitors[ engine_url ].update_no_value( current_time )
                ttft = self.ttft_monitors[ engine_url ].get_average( )

            in_prefill = len(self.in_prefill_requests_ids.get( engine_url, set( ) ))
            in_decoding = len(self.in_decoding_requests_ids.get( engine_url, set( ) ))
            finished = self.finished_requests.get( engine_url, 0 )

            in_prefill_ts_s = [ current_time - self.request_arrival_time[r] for r in
                                self.in_prefill_requests_ids.get( engine_url, set( ) ) ]
            in_decode_ts_s = [ current_time - self.first_token_time[ (engine_url, r) ] for r in
                               self.in_decoding_requests_ids.get( engine_url, set( ) ) ]

            if engine_url in self.decoding_length_monitors:
                self.decoding_length_monitors[ engine_url ].update_no_value( current_time )
                avg_dec_len = self.decoding_length_monitors[ engine_url ].get_average( )
            else:
                avg_dec_len = -1

            if engine_url in self.latency_monitors:
                self.latency_monitors[ engine_url ].update_no_value( current_time )
                avg_lat = self.latency_monitors[ engine_url ].get_average( )
            else:
                avg_lat = -1

            # For avg_itl, if not computed, default to -1.
            avg_itl_val = -1

            if engine_url in self.swapped_requests:
                swapped = self.swapped_requests[ engine_url ]
            else:
                swapped = 0

            ret[ engine_url ] = RequestStats( qps = qps,
                ttft = ttft,
                in_prefill_requests = in_prefill,
                ts_prefill_enqueue = in_prefill_ts_s,
                in_decoding_requests = in_decoding,
                ts_decoding_enqueue = in_decode_ts_s,
                finished_requests = finished,
                uptime = (current_time - self.first_query_time if self.first_query_time else 0),
                avg_decoding_length = avg_dec_len,
                avg_latency = avg_lat,
                avg_itl = avg_itl_val,
                num_swapped_requests = swapped, )
        return ret

    def estimate_allocated_blocks(self, engine_url: str) -> int:
        """
        Estimate the total number of blocks currently allocated for all requests in decoding phase.
        For each request, total tokens = prefill_tokens + decode_tokens
        Allocated blocks = ceil(total_tokens / BLOCK_SIZE)

        Args:
            engine_url: The URL of the serving engine

        Returns:
            The total number of blocks allocated across all requests in decoding phase for this engine
        """
        if engine_url not in self.request_decode_tokens or engine_url not in self.in_decoding_requests_ids:
            return 0
        
        total_blocks = 0
        
        # Iterate over requests in request_decode_tokens and verify they're in decoding phase
        for request_id, decode_tokens in self.request_decode_tokens[engine_url].items():
            assert request_id in self.in_decoding_requests_ids[engine_url], \
                f"Request {request_id} has decode tokens but is not in decoding phase"
                
            # Get prefill tokens
            prefill_tokens = self.request_prefill_tokens.get(engine_url, {}).get(request_id, 0)
            # Calculate total tokens and blocks
            total_tokens = prefill_tokens + decode_tokens
            blocks = math.ceil(total_tokens / BLOCK_SIZE)
            total_blocks += blocks
            
        return total_blocks

    def estimate_pending_reserved_blocks(self, engine_url: str) -> int:
        """
        Estimate the number of blocks that need to be reserved for pending requests in prefill phase.
        For each pending request, we reserve blocks based on:
        - Prefill tokens
        - Expected decode tokens (using DECODE_TO_PREFILL_RATIO)
        
        Args:
            engine_url: The URL of the serving engine
            
        Returns:
            The total number of blocks that need to be reserved for pending requests
        """
        if engine_url not in self.request_prefill_tokens:
            return 0
            
        # Sum all prefill tokens for pending requests
        total_prefill_tokens = sum(
            tokens for tokens in self.request_prefill_tokens[engine_url].values()
        )
        
        # Calculate total expected tokens including decode phase
        total_expected_tokens = total_prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO)
        
        # Calculate total blocks needed
        return math.ceil(total_expected_tokens / BLOCK_SIZE)


def initialize_request_stats_monitor( sliding_window_size: float ):
    return RequestStatsMonitor( sliding_window_size )


def get_request_stats_monitor( ):
    return RequestStatsMonitor( )
