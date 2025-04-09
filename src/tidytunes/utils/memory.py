import gc

import torch


def is_oom_error(exception: BaseException) -> bool:
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
        or is_onnx_out_of_memory(exception)
    )


def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) >= 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


def is_cudnn_snafu(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) >= 1
        and (
            "CUDNN_STATUS_NOT_SUPPORTED" in exception.args[0]
            or "CUDNN_FE" in exception.args[0]
        )
    )


def is_cufft_snafu(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) >= 1
        and "cuFFT error: CUFFT_INTERNAL_ERROR" in exception.args[0]
    )


def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) >= 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def is_onnx_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "Failed to allocate memory" in exception.args[0]
    )


def garbage_collection_cuda() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.backends.cuda.cufft_plan_cache.clear()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            raise
