from dataclasses import replace
from typing import Tuple, Optional, Callable

# FFCV
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.state import State


class Convert(Operation):
    """Convert to target data type.

    Parameters
    ----------
    target_dtype: numpy.dtype or torch.dtype
        Target data type.
    """

    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def generate_code(self) -> Callable:
        target_dtype = self.target_dtype

        def convert(inp, *args):
            return inp.astype(target_dtype)

        convert.is_parallel = True
        return convert

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.target_dtype), None
