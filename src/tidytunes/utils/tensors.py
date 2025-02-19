import torch


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None):
    """Construct a mask from a tensor of lengths, padding is set to False."""
    if max_length is None:
        max_length = lengths.max()
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]


def collate_tensors(
    tensors: list[torch.Tensor],
    padding_value: int | float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an iterable of tensors (of possibly various first dimension, but other dimensions)
    into a single stacked tensor.

    Args:
        tensors (list[torch.Tensor]): A list of tensors to be collated.
        padding_value: (int or float, optional): The padding value inserted to make all tensors
        have the same length.

    Returns:
        1) a tensor with shape `(B, L, *)` where `B` is the number of input tensors,
        `L` is the largest found shape[0], and `*` is the rest of dimensions 2) a tensor with
        shape `(B,)` of the number of rows of the input matrices (their shape[0])
    """
    tensors = [
        t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in tensors
    ]
    padded = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=float(padding_value)
    )
    lens = torch.tensor([len(t) for t in tensors], device=tensors[0].device).long()
    return padded, lens


def masked_mean(
    x: torch.Tensor, x_mask: torch.Tensor | None = None, dim: int = -1
) -> torch.Tensor:
    """
    Calculate the mean value of each row of the input tensor `x` in the given dimension dim,
    while respecting `x_mask` with True on valid input positions and False otherwise.
    """
    if x_mask is not None:
        x[~x_mask] = 0
        x = x.sum(dim=dim) / (x_mask).sum(dim=dim)
    else:
        x = x.mean(dim=dim)
    return x


def masked_std(
    x: torch.Tensor,
    x_mask: torch.Tensor | None = None,
    dim: int = -1,
    mean: torch.Tensor = None,
) -> torch.Tensor:
    """
    Calculate the standard deviation of each row of the input tensor `x` in the given dimension dim,
    while respecting `x_mask` with True on valid input positions and False otherwise.
    """
    if x_mask is not None:
        if mean is None:
            mean = masked_mean(x, x_mask, dim)
        variance = ((x - mean.unsqueeze(dim)) ** 2 * x_mask).sum(dim=dim) / x_mask.sum(
            dim=dim
        )
        std_dev = torch.sqrt(variance)
    else:
        std_dev = x.std(dim=dim)
    return std_dev
