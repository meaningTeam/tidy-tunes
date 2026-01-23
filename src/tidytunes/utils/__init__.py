from .asr import compute_wer
from .audio import Audio, Segment, collate_audios, decollate_audios, trim_audios
from .download import download_github
from .etc import (
    SpeculativeBatcher,
    batched,
    frame_labels_to_time_segments,
    partition,
    to_batches,
)
from .logging import setup_logger
from .tensors import masked_max, masked_mean, masked_std, sequence_mask
from .trace import TraceMixin
