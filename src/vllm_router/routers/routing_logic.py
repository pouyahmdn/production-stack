import abc
import enum
from typing import Dict, List

from fastapi import Request
from uhashring import HashRing

from vllm_router.log import init_logger
from vllm_router.service_discovery import EndpointInfo
from vllm_router.stats.engine_stats import EngineStats
from vllm_router.stats.request_stats import RequestStats
from vllm_router.utils import SingletonABCMeta

logger = init_logger( __name__ )


class RoutingLogic( str, enum.Enum ):
    ROUND_ROBIN = "roundrobin"
    SESSION_BASED = "session"
    LEAST_LOADED = "llq"
    CUSTOM_LOGIC = "custom"
    HRA = "hra"


class RoutingInterface( metaclass = SingletonABCMeta ):
    @abc.abstractmethod
    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request,
                       request_id: str,
                       num_prefill_tokens: int ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_id (str): The ID of the request
            num_prefill_tokens (int): Number of prefill tokens in the request
        """
        raise NotImplementedError


class RoundRobinRouter( RoutingInterface ):
    # TODO (ApostaC): when available engines in the endpoints changes, the
    # algorithm may not be "perfectly" round-robin.
    def __init__( self ):
        if hasattr( self, "_initialized" ):
            return
        self.req_id = 0
        self._initialized = True

    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request,
                       request_id: str,
                       num_prefill_tokens: int ) -> str:
        """
        Route the request to the appropriate engine URL using a simple
        round-robin algorithm

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_id (str): The ID of the request
            num_prefill_tokens (int): Number of prefill tokens in the request
        """
        len_engines = len( endpoints )
        chosen = sorted( endpoints, key = lambda e: e.url )[ self.req_id % len_engines ]
        self.req_id += 1
        monitor = get_request_stats_monitor()
        monitor.on_request_routed(chosen.url, request_id, num_prefill_tokens)
        return chosen.url


class SessionRouter( RoutingInterface ):
    """
    Route the request to the appropriate engine URL based on the session key
    in the request headers
    """

    def __init__( self, session_key: str = None ):
        if hasattr( self, "_initialized" ):
            return
        if session_key is None:
            raise ValueError( "SessionRouter must be initialized with a session_key" )
        self.session_key = session_key
        self.hash_ring = HashRing( )
        self._initialized = True

    def _qps_routing( self, endpoints: List[ EndpointInfo ], request_stats: Dict[ str, RequestStats ] ) -> str:
        """
        Route the request to the appropriate engine URL based on the QPS of
        each engine

        Args:
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
        """
        lowest_qps = float( "inf" )
        ret = None
        for info in endpoints:
            url = info.url
            if url not in request_stats:
                return url  # This engine does not have any requests
            request_stat = request_stats[ url ]
            if request_stat.qps < lowest_qps:
                lowest_qps = request_stat.qps
                ret = url
        return ret

    def _update_hash_ring( self, endpoints: List[ "EndpointInfo" ] ):
        """
        Update the hash ring with the current list of endpoints.
        """
        # Extract endpoint URLs
        endpoint_urls = [ endpoint.url for endpoint in endpoints ]

        # Get the current nodes in the hash ring
        current_nodes = set( self.hash_ring.get_nodes( ) )

        # Convert the new endpoint URLs to a set for easy comparison
        new_nodes = set( endpoint_urls )

        # Remove nodes that are no longer in the list
        for node in current_nodes - new_nodes:
            self.hash_ring.remove_node( node )

        # Add new nodes that are not already in the hash ring
        for node in new_nodes - current_nodes:
            self.hash_ring.add_node( node )

    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request,
                       request_id: str,
                       num_prefill_tokens: int ) -> str:
        """
        Route the request to the appropriate engine URL by the 'session id' in
        the request headers.
        If there is no session id in the request header, it will pick a server
        with lowest qps

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_id (str): The ID of the request
            num_prefill_tokens (int): Number of prefill tokens in the request
        """
        session_id = request.headers.get( self.session_key, None )
        logger.debug( f"Got session id: {session_id}" )

        # Update the hash ring with the current list of endpoints
        self._update_hash_ring( endpoints )

        if session_id is None:
            # Route based on QPS if no session ID is present
            url = self._qps_routing( endpoints, request_stats )
        else:
            # Use the hash ring to get the endpoint for the session ID
            url = self.hash_ring.get_node( session_id )

        monitor = get_request_stats_monitor()
        monitor.on_request_routed(url, request_id, num_prefill_tokens)
        return url


class LeastLoadedRouter( RoutingInterface ):

    def __init__( self ):
        if hasattr( self, "_initialized" ):
            return
        self._initialized = True

    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request,
                       request_id: str,
                       num_prefill_tokens: int ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_id (str): The ID of the request
            num_prefill_tokens (int): Number of prefill tokens in the request
        """

        def estimate_work( url: str ) -> float:
            if url not in request_stats:
                return 0
            else:
                if len( request_stats[ url ].ts_prefill_enqueue ) != request_stats[ url ].in_prefill_requests:
                    logger.debug( f"{url}, {len( request_stats[ url ].ts_prefill_enqueue )}, {request_stats[ url ].in_prefill_requests}" )
                if len( request_stats[ url ].ts_decoding_enqueue ) != request_stats[ url ].in_decoding_requests:
                    logger.debug( f"{url}, {len( request_stats[ url ].ts_decoding_enqueue )}, {request_stats[ url ].in_decoding_requests}" )
                return request_stats[ url ].in_prefill_requests + request_stats[ url ].in_decoding_requests

        lowest_work = float( "inf" )
        ret = None
        for info in endpoints:
            endpoint_work = estimate_work( info.url )
            if endpoint_work < lowest_work:
                lowest_work = endpoint_work
                ret = info.url
        monitor = get_request_stats_monitor()
        monitor.on_request_routed(ret, request_id, num_prefill_tokens)
        return ret


# ---------------------------------------------------------------------------
# Head-Room Admission (HRA) Router
# ---------------------------------------------------------------------------

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from math import ceil

from vllm_router.stats.request_stats import (
    BLOCK_SIZE,
    TOTAL_NUMBER_OF_BLOCKS,
    DECODE_TO_PREFILL_RATIO,
    SAFETY_FRACTION,
    get_request_stats_monitor,
)


@dataclass(order=True)
class _QueuedRequest:
    """Internal helper structure for queued admission-controlled requests."""

    sort_index: int = field(init=False, repr=False)
    prefill_tokens: int
    arrived_at: float
    request: Request
    endpoints: List[EndpointInfo]
    future: asyncio.Future
    request_id: str

    def __post_init__(self):
        # Sorting priority: by prefill tokens, then FIFO arrival time.
        self.sort_index = (self.prefill_tokens, self.arrived_at)


class HRARouter(RoutingInterface):
    """Memory-aware router that implements Head-Room Admission control (HRA).

    The router maintains an internal queue for requests that cannot be
    immediately admitted to any backend replica.  When memory becomes
    available (detected via `on_request_complete`) the queued requests are
    re-evaluated.
    """

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._queue: list[_QueuedRequest] = []
        self._initialized = True

    # ---------------------------------------------------------------------
    # Public API expected by the router core
    # ---------------------------------------------------------------------

    def route_request(
        self,
        endpoints: List[EndpointInfo],
        engine_stats: Dict[str, EngineStats],  # Unused but kept for interface
        request_stats: Dict[str, RequestStats],  # Unused but kept for interface
        request: Request,
        request_id: str,
        num_prefill_tokens: int,
    ) -> str:
        """Either returns a backend URL immediately or waits until admission."""

        future: asyncio.Future = asyncio.get_event_loop().create_future()

        queued_req = _QueuedRequest(
            prefill_tokens=num_prefill_tokens,
            arrived_at=time.time(),
            request=request,
            endpoints=endpoints,
            future=future,
            request_id=request_id,
        )
        self._queue.append(queued_req)
        # Keep queue ordered according to SJF (prefill tokens, then FIFO).
        self._queue.sort()
        self._try_schedule()

        return future

    def on_request_complete(self, engine_url: str):
        """Hook called when a request finishes on *engine_url*.

        We re-run scheduling to see if queued requests can now be admitted.
        """

        # We do not need the *engine_url* explicitly here, but keeping the
        # signature future-proof (could be used for smarter triggers).
        self._try_schedule()

    # ------------------------------------------------------------------
    # Internal helpers (callers must hold self._lock)
    # ------------------------------------------------------------------

    def _try_schedule(self):
        if not self._queue:
            return

        monitor = get_request_stats_monitor()
        current_time = time.time()
        req_stats_snapshot = monitor.get_request_stats(current_time)

        # Pre-compute per-replica values that will be updated speculatively.
        replica_urls = {
            url for qr in self._queue for url in (ep.url for ep in qr.endpoints)
        }

        allocated_blocks: Dict[str, int] = {
            url: monitor.estimate_allocated_blocks(url) for url in replica_urls
        }
        pending_reserved_blocks: Dict[str, int] = {
            url: monitor.estimate_pending_reserved_blocks(url)
            for url in replica_urls
        }
        queue_lengths: Dict[str, int] = {
            url: (
                req_stats_snapshot[url].in_prefill_requests
                + req_stats_snapshot[url].in_decoding_requests
            )
            if url in req_stats_snapshot
            else 0
            for url in replica_urls
        }

        min_free_blocks = int(TOTAL_NUMBER_OF_BLOCKS * SAFETY_FRACTION)

        idx = 0
        while idx < len(self._queue):
            qr = self._queue[idx]

            # Calculate pessimistic block demand for this request.
            req_blocks = ceil(
                qr.prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO) / BLOCK_SIZE
            )

            admissible: list[str] = []
            for ep in qr.endpoints:
                url = ep.url
                projected_usage = allocated_blocks[url] + pending_reserved_blocks[url] + req_blocks
                free_after = TOTAL_NUMBER_OF_BLOCKS - projected_usage
                if free_after >= min_free_blocks:
                    admissible.append(url)

            if not admissible:
                # Oldest unschedulable request blocks younger ones to preserve
                # fairness; stop here.
                break

            # Choose replica with least queue len.
            target_url = min(
                admissible,
                key=lambda u: queue_lengths[u],
            )

            monitor.on_request_routed(target_url, qr.request_id, qr.prefill_tokens)

            # Commit placement: pop from queue, set future result, update local
            # projections so subsequent iterations see the effect.
            qr.future.set_result(target_url)
            self._queue.pop(idx)

            pending_reserved_blocks[target_url] += req_blocks
            queue_lengths[target_url] += 1
            # Do *not* increment idx – we just removed the current element.

        # Done; any remaining queued requests stay.



class CustomRouter( RoutingInterface ):

    def __init__( self ):
        if hasattr( self, "_initialized" ):
            return
        self._initialized = True

    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request,
                       request_id: str,
                       num_prefill_tokens: int ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
            request_id (str): The ID of the request
            num_prefill_tokens (int): Number of prefill tokens in the request
        """

        def estimate_work( url: str ) -> float:
            if url not in request_stats:
                logger.debug( f"{url}, None" )
                return 0
            else:
                ireq = request_stats[ url ].ts_prefill_enqueue
                oreq = request_stats[ url ].ts_decoding_enqueue
                len_ireq = len(ireq)
                len_oreq = len(oreq)
                avg_gen_lat = request_stats[ url ].avg_decoding_length
                avg_ttft = request_stats[ url ].ttft
                in_q_work = len( ireq ) * avg_gen_lat
                in_d_work = sum( max(tdiff, avg_gen_lat) for tdiff in oreq )
                logger.debug( f"{url}, {len_ireq}, {len_oreq}, {avg_ttft}, {avg_gen_lat}, {request_stats[ url ].qps}, {in_q_work}, {in_d_work}, {in_q_work + in_d_work}" )

                if avg_gen_lat < 0:
                    return request_stats[ url ].qps

                assert avg_gen_lat >= 0
                return in_q_work + in_d_work

        lowest_work = float( "inf" )
        ret = None
        for info in endpoints:
            endpoint_work = estimate_work( info.url )
            if endpoint_work < lowest_work:
                lowest_work = endpoint_work
                ret = info.url
        monitor = get_request_stats_monitor()
        monitor.on_request_routed(ret, request_id, num_prefill_tokens)
        return ret


# Instead of managing a global _global_router, we can define the initialization functions as:
def initialize_routing_logic( routing_logic: RoutingLogic, *args, **kwargs ) -> RoutingInterface:
    if routing_logic == RoutingLogic.ROUND_ROBIN:
        logger.info( "Initializing round-robin routing logic" )
        return RoundRobinRouter( )
    elif routing_logic == RoutingLogic.SESSION_BASED:
        logger.info( f"Initializing session-based routing logic with kwargs: {kwargs}" )
        return SessionRouter( kwargs.get( "session_key" ) )
    elif routing_logic == RoutingLogic.LEAST_LOADED:
        logger.info( f"Initializing LLQ routing logic" )
        return LeastLoadedRouter( )
    elif routing_logic == RoutingLogic.HRA:
        logger.info("Initializing HRA routing logic")
        return HRARouter()
    elif routing_logic == RoutingLogic.CUSTOM_LOGIC:
        logger.info( f"Initializing custom routing logic" )
        return CustomRouter( )
    else:
        raise ValueError( f"Invalid routing logic {routing_logic}" )


def reconfigure_routing_logic( routing_logic: RoutingLogic, *args, **kwargs ) -> RoutingInterface:
    # Remove the existing routers from the singleton registry
    for cls in (SessionRouter, RoundRobinRouter, LeastLoadedRouter, HRARouter, CustomRouter):
        if cls in SingletonABCMeta._instances:
            del SingletonABCMeta._instances[ cls ]
    return initialize_routing_logic( routing_logic, *args, **kwargs )


def get_routing_logic( ) -> RoutingInterface:
    # Look up in our singleton registry which router (if any) has been created.
    for cls in (SessionRouter, RoundRobinRouter, LeastLoadedRouter, HRARouter, CustomRouter):
        if cls in SingletonABCMeta._instances:
            return cls( )
    raise ValueError( "The global router has not been initialized" )
