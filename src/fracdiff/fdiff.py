from functools import partial
from typing import Optional

import numpy as np

# found module but no type hints or library stubs
from scipy.special import binom  # type: ignore


def _prepare_boundary(
    a: np.ndarray,
    boundary: Optional[np.ndarray], axis: int
) -> Optional[np.ndarray]:
    """Handles the broadcasting and shape validation for prepend/append arrays."""
    if boundary is None:
        return None

    boundary = np.asanyarray(boundary)
    if boundary.ndim == 0:
        shape = list(a.shape)
        shape[axis] = 1
        return np.broadcast_to(boundary, tuple(shape))
    return boundary


def fdiff_coef(d: float, window: int) -> np.ndarray:
    """Returns sequence of coefficients in fracdiff operator.

    Parameters
    ----------
    d : float
        Order of differentiation.
    window : int
        Number of terms.

    Returns
    -------
    coef : numpy.array, shape (window,)
        Coefficients in fracdiff operator.

    Examples
    --------
    >>> fdiff_coef(0.5, 4)
    array([ 1.    , -0.5   , -0.125 , -0.0625])
    >>> fdiff_coef(1.0, 4)
    array([ 1., -1.,  0., -0.])
    >>> fdiff_coef(1.5, 4)
    array([ 1.    , -1.5   ,  0.375 ,  0.0625])
    """
    return (-1) ** np.arange(window) * binom(d, np.arange(window))


def fdiff(
    a: np.ndarray,
    n: float = 1.0,
    axis: int = -1,
    prepend: Optional[np.ndarray] = None,
    append: Optional[np.ndarray] = None,
    window: int = 10,
    mode: str = "same",
) -> np.ndarray:
    """Calculate the `n`-th differentiation along the given axis.

    Extention of ``numpy.diff`` to fractional differentiation.

    Parameters
    ----------
    a : array_like
        The input array.
    n : float, default=1.0
        The order of differentiation.
        If ``n`` is an integer, returns the same output with ``numpy.diff``.
    axis : int, default=-1
        The axis along which differentiation is performed, default is the last axis.
    prepend : array_like, optional
        Values to prepend to ``a`` along axis prior to performing the differentiation.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes.
        Otherwise the dimension and shape must match ``a`` except along axis.
    append : array_like, optional
        Values to append.
    window : int, default=10
        Number of observations to compute each element in the output.
    mode : {"same", "valid"}, default="same"
        "same" (default) :
            At the beginning of the time series,
            return elements where at least one coefficient of fracdiff is used.
            Output size along ``axis`` is :math:`L_{\\mathrm{in}}`
            where :math:`L_{\\mathrm{in}}` is the length of ``a`` along ``axis``
            (plus the lengths of ``append`` and ``prepend``).
            Boundary effects may be seen at the at the beginning of a time-series.
        "valid" :
            Return elements where all coefficients of fracdiff are used.
            Output size along ``axis`` is
            :math:`L_{\\mathrm{in}} - \\mathrm{window} + 1` where
            where :math:`L_{\\mathrm{in}}` is the length of ``a`` along ``axis``
            (plus the lengths of ``append`` and ``prepend``).
            Boundary effects are not seen.

    Returns
    -------
    fdiff : numpy.ndarray
        The fractional differentiation.
        The shape of the output is the same as ``a`` except along ``axis``.

    Examples
    --------
    This returns the same result with ``numpy.diff`` for integer `n`.

    >>> from fracdiff import fdiff
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> (np.diff(a) == fdiff(a)).all()
    True
    >>> (np.diff(a, 2) == fdiff(a, 2)).all()
    True

    This returns fractional differentiation for noninteger `n`.

    >>> fdiff(a, 0.5, window=3)
    array([ 1.   ,  1.5  ,  2.875,  4.75 , -4.   ])

    Mode "valid" returns elements for which all coefficients are convoluted.

    >>> fdiff(a, 0.5, window=3, mode="valid")
    array([ 2.875,  4.75 , -4.   ])
    >>> fdiff(a, 0.5, window=3, mode="valid", prepend=[1, 1])
    array([ 0.375,  1.375,  2.875,  4.75 , -4.   ])

    Differentiation along desired axis.

    >>> a = np.array([[  1,  3,  6, 10, 15],
    ...               [  0,  5,  6,  8, 11]])
    >>> fdiff(a, 0.5, window=3)
    array([[1.   , 2.5  , 4.375, 6.625, 9.25 ],
           [0.   , 5.   , 3.5  , 4.375, 6.25 ]])
    >>> fdiff(a, 0.5, window=3, axis=0)
    array([[ 1. ,  3. ,  6. , 10. , 15. ],
           [-0.5,  3.5,  3. ,  3. ,  3.5]])
    """
    if mode == "full":
        mode = "same"
        raise DeprecationWarning("mode 'full' was renamed to 'same'.")

    if isinstance(n, int) or n.is_integer():
        prepend = np._NoValue if prepend is None else prepend  # type: ignore
        append = np._NoValue if append is None else append  # type: ignore
        return np.diff(a, n=int(n), axis=axis, prepend=prepend, append=append)

    if float(n).is_integer():
        return np.diff(
            a, n=int(n), axis=axis,
            prepend=prepend if prepend is not None else np._NoValue,
            append=append if append is not None else np._NoValue
        )

    a = np.asanyarray(a)
    if a.ndim == 0:
        raise ValueError("diff requires input that is at least one dimensional")

    # Mypy complains:
    # fracdiff/fdiff.py:135: error: Module has no attribute "normalize_axis_index"
    axis = np.core.multiarray.normalize_axis_index(axis, a.ndim)  # type: ignore
    dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64

    parts = [
        _prepare_boundary(a, prepend, axis),
        a,
        _prepare_boundary(a, append, axis)
    ]
    a = np.concatenate([p for p in parts if p is not None], axis=axis)

    # 5. Core Fractional Differentiation Logic
    coef = fdiff_coef(n, window).astype(dtype)

    if mode == "valid":
        conv_func = partial(np.convolve, coef, mode="valid")
        return np.apply_along_axis(conv_func, axis, a)

    if mode == "same":
        conv_func = partial(np.convolve, coef, mode="full")
        out = np.apply_along_axis(conv_func, axis, a)
        # Use a dynamic slice to trim the 'full' convolution back to 'same' size
        indexer = [slice(None)] * out.ndim
        indexer[axis] = slice(0, a.shape[axis])
        return out[tuple(indexer)]

    raise ValueError(f"Invalid mode: {mode}")
