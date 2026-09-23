"""Fixed-order sparse inference for isolated, reproducible evaluation processes."""
from contextlib import contextmanager
from unittest.mock import patch

import torch
import warp as wp

from .full_brain import SparseMessage


@wp.kernel
def ordered_rows(ptr: wp.array(dtype=wp.int32), pre: wp.array(dtype=wp.int32),
                 values: wp.array(dtype=wp.float32), activity: wp.array2d(dtype=wp.float32),
                 result: wp.array2d(dtype=wp.float32)):
    row, batch = wp.tid()
    total = float(0.0)
    for edge in range(ptr[row], ptr[row+1]):
        total = total + values[edge]*activity[pre[edge], batch]
    result[row, batch] = total


def ordered_forward(ctx, values, activity, ptr, pre):
    if torch.is_grad_enabled():
        raise RuntimeError('Ordered CSR supports inference only')
    if values.dtype != torch.float32 or activity.dtype != torch.float32:
        raise TypeError('Ordered CSR requires float32')
    if ptr.dtype != torch.int32 or pre.dtype != torch.int32:
        raise TypeError('Ordered CSR requires int32 indices')
    result = torch.empty_like(activity)
    wp.launch(ordered_rows, dim=activity.shape,
              inputs=[wp.from_torch(ptr), wp.from_torch(pre), wp.from_torch(values), wp.from_torch(activity)],
              outputs=[wp.from_torch(result)], stream=wp.stream_from_torch(torch.cuda.current_stream()))
    return result


@contextmanager
def ordered_inference():
    wp.init()
    with torch.inference_mode(), patch.object(SparseMessage, 'forward', staticmethod(ordered_forward)):
        yield
