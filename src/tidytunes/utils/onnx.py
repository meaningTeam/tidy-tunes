from pathlib import Path

import onnxruntime as ort
import torch


def load_onnx_session(
    path: Path, device: torch.device, num_threads: int | None
) -> ort.InferenceSession:

    opts = ort.SessionOptions()
    if num_threads is not None:
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads

    providers = ["CPUExecutionProvider"]
    provider_options = None

    if device.type != "cpu":
        provider_options = [
            {"device_id": 0 if device.index is None else device.index},
            {},
        ]

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"] + providers
        elif "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = ["CoreMLExecutionProvider"] + providers
        else:
            raise NotImplementedError()

    infs = ort.InferenceSession(
        path, providers=providers, provider_options=provider_options, sess_options=opts
    )
    return infs
