from typing import Optional

import torch
from torch import Tensor

from fracdiff.fdiff import fdiff_coef as fdiff_coef_numpy


def fdiff_coef(d: float, window: int) -> Tensor:
    """Returns sequence of coefficients in fracdiff operator.

    Args:
        d (float): Order of differentiation.
        window (int): Number of terms.

    Returns:
        torch.Tensor

    Examples:
        >>> from fracdiff.torch import fdiff_coef
        >>>
        >>> fdiff_coef(0.5, 4)
        tensor([ 1.0000, -0.5000, -0.1250, -0.0625], dtype=torch.float64)
        >>> fdiff_coef(1.0, 4)
        tensor([ 1., -1.,  0., -0.], dtype=torch.float64)
        >>> fdiff_coef(1.5, 4)
        tensor([ 1.0000, -1.5000,  0.3750,  0.0625], dtype=torch.float64)
    """
    return torch.as_tensor(fdiff_coef_numpy(d, window))


def _prepare_boundary(
    input: Tensor,
    boundary: Optional[Tensor], dim: int
) -> Optional[Tensor]:
    """Ensures prepend/append tensors match input dtype, device, and expected shape."""
    if boundary is None:
        return None

    boundary = torch.as_tensor(boundary).to(input)
    if boundary.dim() == 0:
        size = list(input.size())
        size[dim] = 1
        return boundary.broadcast_to(torch.Size(size))
    return boundary


def fdiff(
    input: Tensor,
    n: float,
    dim: int = -1,
    prepend: Optional[Tensor] = None,
    append: Optional[Tensor] = None,
    window: int = 10,
    mode: str = "same",
) -> Tensor:
    r"""Computes the ``n``-th differentiation along the given dimension.

    This is an extension of :func:`torch.diff` to fractional differentiation.
    See :class:`fracdiff.torch.Fracdiff` for details.

    Note:
        For integer ``n``, the output is the same as :func:`torch.diff`
        and the parameters ``window`` and ``mode`` are ignored.

    Shape:
        - input: :math:`(N, *, L_{\mathrm{in}})`, where where :math:`*` means any
          number of additional dimensions.
        - output: :math:`(N, *, L_{\mathrm{out}})`, where :math:`L_{\mathrm{out}}`
          is given by :math:`L_{\mathrm{in}}` if `mode="same"` and
          :math:`L_{\mathrm{in}} - \mathrm{window} - 1` if `mode="valid"`.
          If `prepend` and/or `append` are provided, then :math:`L_{\mathrm{out}}`
          increases by the number of elements in each of these tensors.

    Examples:
        >>> from fracdiff.torch import fdiff
        ...
        >>> input = torch.tensor([1, 2, 4, 7, 0])
        >>> fdiff(input, 0.5, mode="same", window=3)
        tensor([ 1.0000,  1.5000,  2.8750,  4.7500, -4.0000])

        >>> fdiff(input, 0.5, mode="valid", window=3)
        tensor([ 2.8750,  4.7500, -4.0000])

        >>> fdiff(input, 0.5, mode="valid", window=3, prepend=[1, 1])
        tensor([ 0.3750,  1.3750,  2.8750,  4.7500, -4.0000])

        >>> input = torch.arange(10).reshape(2, 5)
        >>> fdiff(input, 0.5)
        tensor([[0.0000, 1.0000, 1.5000, 1.8750, 2.1875],
                [5.0000, 3.5000, 3.3750, 3.4375, 3.5547]])
    """
    # Calls torch.diff if n is an integer
    if float(n).is_integer():
        return input.diff(n=int(n), dim=dim, prepend=prepend, append=append)

    if not input.is_floating_point():
        input = input.to(torch.get_default_dtype())

    dim = dim if dim >= 0 else dim + input.dim()

    parts = [
        _prepare_boundary(input, prepend, dim),
        input,
        _prepare_boundary(input, append, dim)
    ]
    input = torch.cat([p for p in parts if p is not None], dim=dim)

    # 4. Dimension Shifting (Supports any dim by moving it to the end)
    # We transpose 'dim' to the last position, process, then transpose back.
    if dim != input.dim() - 1:
        input = input.transpose(dim, -1)

    orig_shape = input.shape
    # Flatten all dimensions except the last one for conv1d: (N, 1, L)
    input_reshaped = input.reshape(-1, 1, orig_shape[-1])

    # 5. Convolution Setup
    # Fractional diff is a causal filter; we pad the beginning (left)
    input_padded = torch.nn.functional.pad(input_reshaped, (window - 1, 0))

    # Weight shape for conv1d: (out_channels, in_channels, kernel_width)
    # We flip because convolution in PyTorch is cross-correlation
    weight = fdiff_coef(n, window).to(input).reshape(1, 1, -1).flip(-1)

    output = torch.nn.functional.conv1d(input_padded, weight)

    # 6. Mode Slicing
    if mode == "same":
        L_out = orig_shape[-1]
    elif mode == "valid":
        L_out = orig_shape[-1] - window + 1
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Slice the last L_out elements
    output = output[..., -L_out:]

    # 7. Reshape and Restore Dimension
    output = output.reshape(orig_shape[:-1] + (L_out,))
    if dim != input.dim() - 1:
        output = output.transpose(dim, -1)

    return output
