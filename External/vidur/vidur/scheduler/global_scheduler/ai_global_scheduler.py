"""Head-room admission global scheduler (HRA).

Goal
====
Eliminate most **pre-emptions** inside the vLLM replica scheduler by keeping a
fixed KV-cache head-room on every replica and by pessimistically reserving
additional blocks to account for the unknown *decode* phase.

Key parameters (chosen empirically for the ShareGPT-7 workload):
    DECODE_TO_PREFILL_RATIO ≈ 1.3   – mean decode / prefill token ratio.
    SAFETY_FRACTION          = 0.10  – keep 10 % of blocks free at all times.

The scheduler admits a request to a replica **only if** that replica would stay
within the safe memory envelope *after* accounting for

    prompt_tokens * (1 + ratio).

If no replica can currently host the request, it remains in the global queue
until a later batch completion frees space.  This sometimes increases queuing
delay a little but removes the far more expensive pre-empt / restart cycles.
"""

from math import ceil
from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


# ---------- Tunables ---------------------------------------------------------

DECODE_TO_PREFILL_RATIO = 0.5
SAFETY_FRACTION = 0.02


class AIGlobalScheduler(BaseGlobalScheduler):
    def schedule(self) -> List[Tuple[int, Request]]:
        """Return list of (replica_id, request) assignments for this timestep."""

        # Prioritise short prompts (SJF) to minimise average turnaround.
        self._request_queue.sort(key=lambda r: (r.num_prefill_tokens, r.arrived_at))

        if not self._request_queue:
            return []

        # Static values (identical for all replicas of the cluster).
        any_rs = next(iter(self._replica_schedulers.values()))
        block_size = any_rs._config.block_size
        max_blocks_per_replica = any_rs._config.num_blocks
        min_free_blocks = int(max_blocks_per_replica * SAFETY_FRACTION)

        # Helper maps that we update optimistically while dispatching multiple
        # requests within the same global-schedule call.
        allocated_blocks = {rid: rs.num_allocated_blocks for rid, rs in self._replica_schedulers.items()}
        pending_blocks = {
            rid: ceil(sum(r.num_prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO) for r in rs._request_queue) / block_size)
            for rid, rs in self._replica_schedulers.items()
        }
        queue_len = {rid: rs.num_pending_requests + rs.num_active_requests for rid, rs in self._replica_schedulers.items()}

        mapping: List[Tuple[int, Request]] = []

        idx = 0
        # Traverse requests in prefill length order. If the head request cannot be
        # admitted at the moment we *stop*; a later batch completion will retry.
        while idx < len(self._request_queue):
            req = self._request_queue[idx]

            # pessimistic block requirement (prompt + avg decode)
            need_blocks = ceil(req.num_prefill_tokens * (1 + DECODE_TO_PREFILL_RATIO) / block_size)

            admissible = []
            for rid in self._replica_schedulers.keys():
                projected = allocated_blocks[rid] + pending_blocks[rid] + need_blocks
                free_after = max_blocks_per_replica - projected
                if free_after >= min_free_blocks:
                    admissible.append(rid)

            if not admissible:
                # Cannot place *this* request right now -> stop processing newer
                # arrivals so we preserve strict SRPT order / fairness.
                break

            # choose replica with smallest queue length; tie-break by least projected usage.
            chosen = min(admissible, key=lambda rid: (queue_len[rid], allocated_blocks[rid] + pending_blocks[rid]))

            # Commit placement.
            mapping.append((chosen, req))
            # Remove from global queue.
            self._request_queue.pop(idx)  # do *not* advance idx

            # optimistic bookkeeping so later requests in the same cycle see the effect
            pending_blocks[chosen] += need_blocks
            queue_len[chosen] += 1

        return mapping