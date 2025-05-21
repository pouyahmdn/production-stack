from typing import List
import os
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)

GS_LOG_HEADER = "time,replica_id,num_pending_requests,num_active_requests,num_allocated_blocks,num_blocks,memory_usage_percent"

class GlobalScheduleEvent(BaseEvent):
    outstanding: bool = False

    def __init__(self, time: float, callback: bool = False):
        super().__init__(time, EventType.GLOBAL_SCHEDULE)

        self._replica_set = []
        self._request_mapping = []

        # If a request finished, this event is actually a callback to redo global scheduling.
        # Since callbacks are scheduled immediately after a batch end, there is no point to having more than one
        # callback event at a time.
        self._callback = callback
        if self._callback:
            assert self.outstanding is False
            self.set_outstanding()

    @classmethod
    def clear_outstanding( cls ):
        cls.outstanding = False

    @classmethod
    def set_outstanding( cls ):
        cls.outstanding = True

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        
        base_path=metrics_store._config.output_dir
        file_path = base_path + '/gs_log.csv'
        if not os.path.isfile(file_path):
            with open(file_path, 'a') as f:
                print (GS_LOG_HEADER,file=f)
        
        for rs_id in scheduler._replica_schedulers:
            rs = scheduler.get_replica_scheduler(rs_id)
            with open(file_path, 'a') as f:
                print (
                 f"{self.time},{rs_id},{rs.num_pending_requests},{rs.num_active_requests},{rs.num_allocated_blocks},{rs._config.num_blocks},{rs.memory_usage_percent}",
                 file=f
                )
        
        self._replica_set = set()
        self._request_mapping = scheduler.schedule()

        for replica_id, request in self._request_mapping:
            self._replica_set.add(replica_id)
            scheduler.get_replica_scheduler(replica_id).add_request(request)

        if self._callback:
            assert self.outstanding
            self.clear_outstanding()

        return [
            ReplicaScheduleEvent(self.time, replica_id)
            for replica_id in self._replica_set
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_set": self._replica_set,
            "request_mapping": [
                (replica_id, request.id)
                for replica_id, request in self._request_mapping
            ],
        }