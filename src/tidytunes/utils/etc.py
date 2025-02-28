import torch

from .audio import Audio, Segment


def partition(lst: list, by: list, other: list | None = None) -> tuple[list, list]:
    """
    Split input list `lst` into two lists according to `by`, i.e. a list of boolean
    values. If `other` is provided, the list corresponding to False values in `by`
    is made of elements of the `other` list and not the input list.
    """
    assert len(lst) == len(by)
    a = [item for cond, item in zip(by, lst) if cond]
    if other is None:
        other = lst
    b = [item for cond, item in zip(by, other) if not cond]
    return a, b


def frame_labels_to_time_segments(
    frame_labels: torch.Tensor,
    frame_shift: float,
    filter_with=None,
) -> list[Segment]:
    """
    Converts a 1-D tensor of frame labels to a list of Segments, each having
    `start` time in seconds, `duration` in seconds, and `symbol` with the
    corresponding label.

    Args:
        frame_labels (T,): Tensor with label indices.
        frame_shift (float): Frame shift, used to convert frame indices to time stamps.
        filter_with (callable, optional): Filtering function of output segments.
                                          Defaults to no filtering.

    Returns:
        list[Segment]: List of possibly filtered segments labeled with symbols.
    """
    assert frame_labels.ndim == 1

    if filter_with is None:
        filter_with = lambda x: True

    frame_labels, frame_count = frame_labels.unique_consecutive(return_counts=True)
    start_frame = frame_count.cumsum(dim=-1) - frame_count

    segments = []
    for symbol, fc, st in zip(
        frame_labels.tolist(), frame_count.tolist(), start_frame.tolist()
    ):
        item = Segment(
            start=round(st * frame_shift, 2),
            duration=round(fc * frame_shift, 2),
            symbol=symbol,
        )
        if filter_with(item):
            segments.append(item)

    return segments


def to_batches(audios: list[Audio], max_size: int, max_duration: float) -> list[list]:
    """
    Split input list `audios` into lists of length of at most `max_size`, but at
    least 1, while containing Audio objects with duration of at most `max_duration`
    (might be violated when a about to return only a single element).
    """
    assert max_size >= 1
    assert max_duration > 0.0

    batches, batch = [], []
    for audio in audios:

        if len(batch) == 0:
            batch.append(audio)
            continue

        total_duration = sum(a.duration for a in batch)
        if (len(batch) == max_size) or (total_duration + audio.duration > max_duration):
            batches.append(batch)
            batch = []

        batch.append(audio)

    if len(batch) > 0:
        batches.append(batch)

    return batches
