from abc import ABC, abstractmethod

import torch


class HasDummyInputs(ABC):
    """
    Abstract base class requiring implementation of `dummy_inputs` method
    to provide sample inputs for model tracing.
    """

    @abstractmethod
    def dummy_inputs(
        self, batch: int, device: str, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        pass


class TraceMixin(HasDummyInputs):
    """
    Mixin providing JIT tracing functionality for models that implement `HasDummyInputs`.
    """

    trace_atol: float = 1e-4
    trace_rtol: float = 1e-4

    def to_jit_trace(
        self,
        device: str = "cpu",
        batch_size: int = 2,
        dtype: torch.dtype = torch.float,
        check_trace: bool = True,
    ) -> torch.jit.ScriptModule:
        """
        Converts the model to a JIT-traced TorchScript module.

        Args:
            device (str): Device to place the model on.
            batch_size (int): Batch size for dummy inputs.
            dtype (torch.dtype): Data type for inputs.
            check_trace (bool): Whether to validate the traced model output.

        Returns:
            torch.jit.ScriptModule: The traced TorchScript model.
        """
        return to_jit_trace(
            self,
            device=device,
            batch_size=batch_size,
            dtype=dtype,
            atol=self.trace_atol,
            rtol=self.trace_rtol,
            check_trace=check_trace,
        )


def to_jit_trace(
    model: HasDummyInputs | torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 2,
    dtype: torch.dtype = torch.float,
    atol: float | None = None,
    rtol: float | None = None,
    dummy_inputs: tuple[torch.Tensor, ...] | None = None,
    check_trace: bool = True,
) -> torch.jit.ScriptModule:
    """
    Traces a given model with dummy inputs and return an executable that will be optimized using just-in-time compilation.

    Args:
        model (HasDummyInputs or torch.nn.Module): The model to be traced.
        device (str): The device to run tracing on.
        batch_size (int): The batch size for dummy inputs.
        dtype (torch.dtype): Data type for inputs.
        atol (float, optional): Absolute tolerance for trace verification.
        rtol (float, optional): Relative tolerance for trace verification.
        dummy_inputs (tuple[torch.Tensor, ...], optinal): Custom dummy inputs if provided.
        check_trace (bool): Whether to validate traced model output.

    Returns:
        torch.jit.ScriptModule: The traced TorchScript model.
    """
    torch.manual_seed(0)
    model.to(device)
    model.eval()

    if dummy_inputs is None:
        dummy_inputs = model.dummy_inputs(batch=batch_size, device=device, dtype=dtype)

    with torch.no_grad(), torch.amp.autocast(
        enabled=dtype in {torch.half, torch.bfloat16},
        dtype=dtype,
        device_type=device,
    ):
        torch.manual_seed(0)
        reference_outputs = model(*dummy_inputs)
        traced_model = torch.jit.trace(
            model, dummy_inputs, check_trace=False, strict=False
        )

    torch.manual_seed(0)
    traced_outputs = traced_model(*dummy_inputs)

    if check_trace:
        assert_tensors_close(reference_outputs, traced_outputs, atol=atol, rtol=rtol)

    return traced_model


def assert_tensors_close(
    tensors1,
    tensors2,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """
    Assert that two sets of tensors are numerically close.

    Args:
        tensors1: First iterable of tensors or a single tensor.
        tensors2: Second iterable of tensors or a single tensor.
        atol (float, optional): Absolute tolerance.
        rtol (float, optional): Relative tolerance.

    Raises:
        AssertionError: If the tensors are not close within the given tolerances.
    """
    if torch.is_tensor(tensors1):
        torch.testing.assert_close(tensors1, tensors2, atol=atol, rtol=rtol)
    else:
        assert len(tensors1) == len(tensors2), "Tensor lists must have the same length."
        for t1, t2 in zip(tensors1, tensors2):
            assert_tensors_close(t1, t2, atol=atol, rtol=rtol)
