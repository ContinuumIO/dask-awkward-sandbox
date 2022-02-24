from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

import awkward._v2 as ak

from .core import (
    DaskAwkwardNotImplemented,
    TrivialPartitionwiseOp,
    pw_reduction_with_agg_to_scalar,
)

if TYPE_CHECKING:
    from .core import Array, Scalar

    LazyResult = Union[Array, Scalar]

__all__ = (
    "all",
    "any",
    "argmax",
    "argmin",
    "corr",
    "count",
    "count_nonzero",
    "covar",
    "linear_fit",
    "max",
    "mean",
    "min",
    "moment",
    "prod",
    "ptp",
    "softmax",
    "std",
    "sum",
    "var",
)

_count_trivial = TrivialPartitionwiseOp(ak.count, axis=1)
_count_nonzero_trivial = TrivialPartitionwiseOp(ak.count_nonzero, axis=1)
_min_trivial = TrivialPartitionwiseOp(ak.min, axis=1)
_max_trivial = TrivialPartitionwiseOp(ak.max, axis=1)
_sum_trivial = TrivialPartitionwiseOp(ak.sum, axis=1)
_std_trivial = TrivialPartitionwiseOp(ak.std, axis=1)


def all(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def any(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def argmax(array, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def argmin(array, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def corr(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented("TODO")


def count(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis == 1:
        return _count_trivial(
            array,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    elif axis is None:
        trivial_result = _count_trivial(
            array,
            axis=1,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
        return pw_reduction_with_agg_to_scalar(
            trivial_result,
            ak.sum,
            ak.sum,
        )
    elif axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be None or an integer.")


def count_nonzero(
    array, axis=None, keepdims=False, mask_identity=False, flatten_records=False
):
    if axis is not None and axis == 1:
        return _count_nonzero_trivial(
            array,
            axis=1,
            keepdims=False,
            mask_identity=False,
            flatten_records=False,
        )
    elif axis is None:
        trivial_result = _count_nonzero_trivial(
            array,
            axis=1,
            keepdims=False,
            mask_identity=False,
            flatten_records=False,
        )
        return pw_reduction_with_agg_to_scalar(
            trivial_result,
            ak.sum,
            ak.sum,
        )
    elif axis == 0 or axis == -1 * array.ndim:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be None or an integer.")


def covar(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented("TODO")


def linear_fit(
    x,
    y,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented("TODO")


def max(
    array,
    axis=None,
    keepdims=False,
    initial=None,
    mask_identity=True,
    flatten_records=False,
):
    return _min_or_max(
        ak.max,
        array,
        axis,
        keepdims=keepdims,
        initial=initial,
        mask_identity=mask_identity,
        flatten_records=flatten_records,
    )


def mean(
    x, weight=None, axis=None, keepdims=False, mask_identity=True, flatten_records=False
):
    raise DaskAwkwardNotImplemented("TODO")


def min(
    array,
    axis=None,
    keepdims=False,
    initial=None,
    mask_identity=True,
    flatten_records=False,
):
    return _min_or_max(
        ak.min,
        array,
        axis,
        keepdims=keepdims,
        initial=initial,
        mask_identity=mask_identity,
        flatten_records=flatten_records,
    )


def moment(
    x,
    n,
    weight=None,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented("TODO")


def prod(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def ptp(arr, axis=None, keepdims=False, mask_identity=True, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def softmax(x, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    raise DaskAwkwardNotImplemented("TODO")


def std(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    if weight is not None:
        raise DaskAwkwardNotImplemented("dak.std with weights is not supported yet.")

    if axis == 1:
        return _std_trivial(
            x,
            weight=weight,
            ddof=ddof,
            axis=axis,
            keepdims=keepdims,
            mask_identity=mask_identity,
            flatten_records=flatten_records,
        )
    raise DaskAwkwardNotImplemented("TODO")


def sum(array, axis=None, keepdims=False, mask_identity=False, flatten_records=False):
    if axis is not None and axis < 0:
        axis = array.ndim + axis + 1
    if axis == 1:
        return _sum_trivial(
            array, keepdims=False, mask_identity=False, flatten_records=False
        )
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(array, ak.sum, ak.sum)
    elif axis == 0:
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be none or an integer")


def var(
    x,
    weight=None,
    ddof=0,
    axis=None,
    keepdims=False,
    mask_identity=True,
    flatten_records=False,
):
    raise DaskAwkwardNotImplemented("TODO")


def _min_or_max(
    f: Callable,
    array: Array,
    axis: int | None = None,
    **kwargs: Any,
) -> LazyResult:
    # translate negative axis (array.ndim currently raises)
    if axis is not None and axis < 0 and array.ndim is not None:
        axis = array.ndim + axis + 1
    # get the correct trivial callable
    tf = _min_trivial if f == ak.min else _max_trivial
    # generate collection based on axis
    if axis == 1:
        return tf(array, axis=axis, **kwargs)
    elif axis is None:
        return pw_reduction_with_agg_to_scalar(array, f, f, **kwargs)
    elif array.ndim is not None and (axis == 0 or axis == -1 * array.ndim):
        raise DaskAwkwardNotImplemented(
            f"axis={axis} is not supported for this array yet."
        )
    else:
        raise ValueError("axis must be None or an integer.")
