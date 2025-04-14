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


class RoutingInterface( metaclass = SingletonABCMeta ):
    @abc.abstractmethod
    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request, ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
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
                       request: Request, ) -> str:
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
        """
        len_engines = len( endpoints )
        chosen = sorted( endpoints, key = lambda e: e.url )[ self.req_id % len_engines ]
        self.req_id += 1
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
                       request: Request, ) -> str:
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
                       request: Request, ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
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
        return ret


class CustomRouter( RoutingInterface ):

    def __init__( self ):
        if hasattr( self, "_initialized" ):
            return
        self._initialized = True

    def route_request( self,
                       endpoints: List[ EndpointInfo ],
                       engine_stats: Dict[ str, EngineStats ],
                       request_stats: Dict[ str, RequestStats ],
                       request: Request, ) -> str:
        """
        Route the request to the appropriate engine URL

        Args:
            endpoints (List[EndpointInfo]): The list of engine URLs
            engine_stats (Dict[str, EngineStats]): The engine stats indicating
                the 'physical' load of each engine
            request_stats (Dict[str, RequestStats]): The request stats
                indicating the request-level performance of each engine
            request (Request): The incoming request
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
    elif routing_logic == RoutingLogic.CUSTOM_LOGIC:
        logger.info( f"Initializing custom routing logic" )
        return CustomRouter( )
    else:
        raise ValueError( f"Invalid routing logic {routing_logic}" )


def reconfigure_routing_logic( routing_logic: RoutingLogic, *args, **kwargs ) -> RoutingInterface:
    # Remove the existing routers from the singleton registry
    for cls in (SessionRouter, RoundRobinRouter, LeastLoadedRouter, CustomRouter):
        if cls in SingletonABCMeta._instances:
            del SingletonABCMeta._instances[ cls ]
    return initialize_routing_logic( routing_logic, *args, **kwargs )


def get_routing_logic( ) -> RoutingInterface:
    # Look up in our singleton registry which router (if any) has been created.
    for cls in (SessionRouter, RoundRobinRouter, LeastLoadedRouter, CustomRouter):
        if cls in SingletonABCMeta._instances:
            return cls( )
    raise ValueError( "The global router has not been initialized" )
