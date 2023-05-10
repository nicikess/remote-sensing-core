from typing import Callable, Tuple, Optional

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State

from remote_sensing_core.constants import SEASON_S2_INDEX

class SeasonS2Transform(Operation):
    def generate_code(self) -> Callable:

        def season_s2_transform(row, *args):
            # access season s2 at index 0
            season_s2 = row[SEASON_S2_INDEX]
            return season_s2 * 255

        return season_s2_transform

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, None
