from __future__ import annotations

import keyword
import logging
import math
import numbers
import operator
import sys
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from enum import IntEnum
from functools import cached_property, partial, wraps
from inspect import getattr_static
from numbers import Number
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, overload

import awkward as ak
import cachetools
import dask.config
import numpy as np
from awkward._do import remove_structure as ak_do_remove_structure
from awkward.highlevel import NDArrayOperatorsMixin, _dir_pattern
from awkward.typetracer import (
    MaybeNone,
    OneOf,
    TypeTracerArray,
    create_unknown_scalar,
    is_unknown_scalar,
)
from dask.base import (
    DaskMethodsMixin,
    dont_optimize,
    is_dask_collection,
    tokenize,
    unpack_collections,
)
from dask.blockwise import BlockwiseDep
from dask.blockwise import blockwise as dask_blockwise
from dask.context import globalmethod
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import IndexCallable
from dask.utils import OperatorMethodMixin as DaskOperatorMethodMixin
from dask.utils import funcname, is_arraylike, key_split

from dask_awkward.layers import (
    AwkwardBlockwiseLayer,
    AwkwardMaterializedLayer,
    _dask_uses_tasks,
)
from dask_awkward.lib.optimize import all_optimizations
from dask_awkward.utils import (
    ConcretizationTypeError,
    DaskAwkwardNotImplemented,
    IncompatiblePartitions,
    field_access_to_front,
    first,
    hyphenize,
    is_empty_slice,
)

if _dask_uses_tasks:
    from dask.blockwise import TaskRef

if TYPE_CHECKING:
    from awkward.contents.content import Content
    from awkward.forms.form import Form
    from awkward.types.arraytype import ArrayType
    from awkward.types.type import Type
    from dask.array.core import Array as DaskArray
    from dask.bag.core import Bag as DaskBag
    from dask.typing import Graph, Key, NestedKeys, PostComputeCallable
    from numpy.typing import DTypeLike


T = TypeVar("T")


log = logging.getLogger(__name__)


def _make_dask_descriptor(func: Callable) -> Callable[[T, type[T], Array], Any]:
    """Adapt a function accepting a `dask_array` into a dask-awkward descriptor
    that invokes and returns the user function when invoked.

    Parameters
    ----------
    func : Callable dask-awkward descriptor body

    Returns
    -------
    Callable
        The callable dask-awkward descriptor
    """

    def descriptor(instance: T, owner: type[T], dask_array: Array) -> Any:
        impl = func.__get__(instance, owner)
        return impl(dask_array)

    return descriptor


def _make_dask_method(func: Callable) -> Callable[[T, type[T], Array], Callable]:
    """Adapt a function accepting a `dask_array` and additional arguments into
    a dask-awkward descriptor that invokes and returns the bound user function.

    Parameters
    ----------
    func : Callable
        The dask-awkward descriptor body.

    Returns
    -------
    Callable
        The callable dask-awkward descriptor.
    """

    def descriptor(instance: T, owner: type[T], dask_array: Array) -> Any:
        def impl(*args, **kwargs):
            impl = func.__get__(instance, owner)
            return impl(dask_array, *args, **kwargs)

        return impl

    return descriptor


F = TypeVar("F", bound=Callable)
G = TypeVar("G", bound=Callable)


class _DaskProperty(property):
    """A property descriptor that exposes a `.dask` method for registering
    dask-awkward descriptor implementations.
    """

    _dask_get: Callable | None = None

    def dask(self, func: F) -> _DaskProperty:
        assert self._dask_get is None
        self._dask_get = _make_dask_descriptor(func)
        return self


def _adapt_naive_dask_get(func: Callable) -> Callable:
    """Adapt a non-dask-awkward user-defined descriptor function into
    a dask-awkward aware descriptor that invokes the original function.

    Parameters
    ----------
    func : Callable
        The non-dask-awkward descriptor body.

    Returns
    -------
    Callable
        The callable dask-awkward aware descriptor body.
    """

    def wrapper(self, dask_array, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


@overload
def dask_property(maybe_func: Callable, *, no_dispatch: bool = False) -> _DaskProperty:
    """An extension of Python's built-in `property` that supports registration
    of a dask getter via `.dask`.

    Parameters
    ----------
    maybe_func : Callable
        The property getter function.
    no_dispatch : bool
        If True, re-use the main getter function as the Dask implementation.

    Returns
    -------
    Callable
        The dask-awkward aware property descriptor
    """


@overload
def dask_property(
    maybe_func: None = None, *, no_dispatch: bool = False
) -> Callable[[Callable], _DaskProperty]:
    """An extension of Python's built-in `property` that supports registration
    of a dask getter via `.dask`.

    Parameters
    ----------
    maybe_func : Callable, optional
        The property getter function.
    no_dispatch : bool
        If True, re-use the main getter function as the Dask implementation.

    Returns
    -------
    Callable
        The callable dask-awkward aware property descriptor factory
    """
    ...


def dask_property(maybe_func=None, *, no_dispatch=False):
    """An extension of Python's built-in `property` that supports registration
    of a dask getter via `.dask`.

    Parameters
    ----------
    maybe_func : Callable, optional
        The property getter function.
    no_dispatch : bool
        If True, re-use the main getter function as the Dask implementation

    Returns
    -------
    Callable
        The callable dask-awkward aware descriptor factory or the descriptor itself
    """

    def dask_property_wrapper(func: Callable) -> _DaskProperty:
        prop = _DaskProperty(func)
        if no_dispatch:
            return prop.dask(_adapt_naive_dask_get(func))
        else:
            return prop

    if maybe_func is None:
        return dask_property_wrapper
    else:
        return dask_property_wrapper(maybe_func)


class _DaskMethod:
    _impl: Callable
    _dask_get: Callable | None = None

    def __init__(self, impl: Callable):
        self._impl = impl

    def __get__(
        self, instance: T | None, owner: type[T] | None = None
    ) -> _DaskMethod | Callable:
        if instance is None:
            return self

        return self._impl.__get__(instance, owner)

    def dask(self, func: Callable) -> _DaskMethod:
        self._dask_get = _make_dask_method(func)
        return self


@overload
def dask_method(maybe_func: F, *, no_dispatch: bool = False) -> _DaskMethod:
    """Decorate an instance method to provide a mechanism for overriding the
    implementation for dask-awkward arrays via `.dask`.

    Parameters
    ----------
    maybe_func : Callable
        The method implementation to decorate.
    no_dispatch : bool
        If True, re-use the main getter function as the Dask implementation

    Returns
    -------
    Callable
        The callable dask-awkward aware method.
    """


@overload
def dask_method(
    maybe_func: None = None, *, no_dispatch: bool = False
) -> Callable[[F], _DaskMethod]:
    """Decorate an instance method to provide a mechanism for overriding the
    implementation for dask-awkward arrays via `.dask`.

    Parameters
    ----------
    maybe_func : Callable, optional
        The method implementation to decorate.
    no_dispatch : bool
        If True, re-use the main getter function as the Dask implementation

    Returns
    -------
    Callable
        The callable dask-awkward aware method factory.
    """


def dask_method(maybe_func=None, *, no_dispatch=False):
    """Decorate an instance method to provide a mechanism for overriding the
    implementation for dask-awkward arrays via `.dask`.

    Parameters
    ----------
    maybe_func : Callable, optional
        The method implementation to decorate.
    no_dispatch : bool
        If True, re-use the main getter function as the Dask implementation

    Returns
    -------
    Callable
        The callable dask-awkward aware method.
    """

    def dask_method_wrapper(func: F) -> _DaskMethod:
        method = _DaskMethod(func)

        if no_dispatch:
            return method.dask(_adapt_naive_dask_get(func))
        else:
            return method

    if maybe_func is None:
        return dask_method_wrapper
    else:
        return dask_method_wrapper(maybe_func)


class Scalar(DaskMethodsMixin, DaskOperatorMethodMixin):
    """Single partition Dask collection representing a lazy Scalar.

    The class constructor is not intended for users. Instances of this
    class will be results from awkward operations.

    Within dask-awkward the ``new_scalar_object`` and
    ``new_known_scalar`` factory functions are used for creating new
    instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: Any | None = None,
        dtype: DTypeLike | None = None,
        known_value: Any | None = None,
    ) -> None:
        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(name, dsk, dependencies=())  # type: ignore
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        if meta is not None and dtype is None:
            self._meta = self._check_meta(meta)
            self._dtype = self._meta.layout.dtype
        elif meta is None and dtype is not None:
            self._meta = ak.Array(create_unknown_scalar(dtype))
            self._dtype = dtype
        else:
            ValueError("One (and only one) of dtype or meta can be defined.")
        self._known_value: Any | None = known_value

    def __dask_graph__(self) -> Graph:
        return self._dask

    def __dask_keys__(self) -> NestedKeys:
        return [self.key]

    def __dask_layers__(self) -> Sequence[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> Hashable:
        return self.name

    __dask_optimize__ = globalmethod(
        all_optimizations, key="awkward_scalar_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self) -> tuple[PostComputeCallable, tuple]:
        return first, ()

    def __dask_postpersist__(self):
        return self._rebuild, ()

    def _rebuild(self, dsk, *, rename=None):
        name = self._name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self._meta, self.known_value)

    def __reduce__(self):
        return (Scalar, (self.dask, self.name, None, self.dtype, self.known_value))

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def name(self) -> str:
        return self._name

    @property
    def key(self) -> Key:
        return (self._name, 0)

    def _check_meta(self, m):
        if isinstance(m, MaybeNone):
            return ak.Array(m.content)
        elif isinstance(m, ak.Array) and len(m) == 1:
            return m
        elif isinstance(m, OneOf) or is_unknown_scalar(m):
            if isinstance(m, TypeTracerArray):
                return ak.Array(m)
            else:
                return m
        raise TypeError(f"meta must be a typetracer, not a {type(m)}")

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def npartitions(self) -> int:
        """Scalar and Records are unpartitioned by definition."""
        return 1

    @property
    def fields(self) -> list[str]:
        return []

    @property
    def layout(self) -> Any:
        raise TypeError("Scalars do not have a layout.")

    @property
    def divisions(self) -> tuple[None, None]:
        """Scalar and Records do not have divisions by definition."""
        return (None, None)

    @staticmethod
    def from_known(s: Any, dtype: DTypeLike | None = None) -> Scalar:
        """Create a scalar from a known value."""
        return new_known_scalar(s, dtype=dtype)

    def __repr__(self) -> str:  # pragma: no cover
        return self.__str__()

    def __str__(self) -> str:
        if self.known_value is not None:
            return (
                f"dask.awkward<{key_split(self.name)}, "
                "type=Scalar, "
                f"dtype={self.dtype}, "
                f"known_value={self.known_value}>"
            )
        return f"dask.awkward<{key_split(self.name)}, type=Scalar, dtype={self.dtype}>"

    def __getitem__(self, where: Any) -> Any:
        msg = (
            "__getitem__ access on Scalars should be done after converting "
            "the Scalar collection to delayed with the to_delayed method."
        )
        raise NotImplementedError(msg)

    @property
    def known_value(self) -> Any | None:
        return self._known_value

    def to_delayed(self, optimize_graph: bool = True) -> Delayed:
        """Convert Scalar collection into a Delayed collection.

        Parameters
        ----------
        optimize_graph : bool
            If ``True`` optimize the existing task graph before
            converting to delayed.

        Returns
        -------
        Delayed
            Resulting Delayed collection object.

        """
        dsk = self.__dask_graph__()
        layer = self.__dask_layers__()[0]
        if optimize_graph:
            layer = f"delayed-{self.name}"
            dsk = self.__dask_optimize__(dsk, self.__dask_keys__())
            dsk = HighLevelGraph.from_collections(layer, dsk, dependencies=())
        return Delayed(self.key, dsk, layer=layer)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError  # pragma: no cover
        msg = (
            "Attribute access on Scalars should be done after converting "
            "the Scalar collection to delayed with the to_delayed method."
        )
        raise AttributeError(msg)

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        def f(self, other):
            name = f"{op.__name__}-{tokenize(self, other)}"
            deps = [self]
            plns = [self.name]
            if is_dask_collection(other):
                task = (op, self.key, *other.__dask_keys__())
                deps.append(other)
                if inv:
                    plns.insert(0, other.name)
                else:
                    plns.append(other.name)
            else:
                if inv:
                    task = (op, other, self.key)
                else:
                    task = (op, self.key, other)
            if inv:
                plns.reverse()
            graph = HighLevelGraph.from_collections(
                name,
                layer=AwkwardMaterializedLayer(
                    {(name, 0): task},
                    previous_layer_names=plns,
                    fn=op,
                ),
                dependencies=tuple(deps),
            )
            if isinstance(other, (Scalar, Array)):
                if inv:
                    meta = op(other._meta, self._meta)
                else:
                    meta = op(self._meta, other._meta)
            else:
                if inv:
                    meta = op(other, self._meta)
                else:
                    meta = op(self._meta, other)
            return new_scalar_object(graph, name, meta=meta)

        return f

    @classmethod
    def _get_unary_operator(cls, op, inv=False):
        def f(self):
            name = f"{op.__name__}-{tokenize(self)}"
            layer = AwkwardMaterializedLayer(
                {(name, 0): (op, self.key)},
                previous_layer_names=[self.name],
            )
            graph = HighLevelGraph.from_collections(
                name,
                layer,
                dependencies=(self,),
            )
            meta = op(self._meta)
            return new_scalar_object(graph, name, meta=meta)

        return f


def _promote_maybenones(op: Callable) -> Callable:
    """Wrap `op` function such that MaybeNone arguments are promoted.

    Typetracer graphs (i.e. what is run by our necessary buffers
    optimization) need `MaybeNone` results to be promoted to length 1
    typetracer arrays. MaybeNone objects don't support these ops, but
    arrays do.

    """

    @wraps(op)
    def f(*args):
        args = tuple(
            ak.Array(arg.content) if isinstance(arg, MaybeNone) else arg for arg in args
        )
        result = op(*args)
        return result

    return f


for op in [
    _promote_maybenones(operator.abs),
    _promote_maybenones(operator.neg),
    _promote_maybenones(operator.pos),
    _promote_maybenones(operator.invert),
    _promote_maybenones(operator.add),
    _promote_maybenones(operator.sub),
    _promote_maybenones(operator.mul),
    _promote_maybenones(operator.floordiv),
    _promote_maybenones(operator.truediv),
    _promote_maybenones(operator.mod),
    _promote_maybenones(operator.pow),
    _promote_maybenones(operator.and_),
    _promote_maybenones(operator.or_),
    _promote_maybenones(operator.xor),
    _promote_maybenones(operator.lshift),
    _promote_maybenones(operator.rshift),
    _promote_maybenones(operator.eq),
    _promote_maybenones(operator.ge),
    _promote_maybenones(operator.gt),
    _promote_maybenones(operator.ne),
    _promote_maybenones(operator.le),
    _promote_maybenones(operator.lt),
]:
    Scalar._bind_operator(op)


def new_scalar_object(
    dsk: HighLevelGraph,
    name: str,
    *,
    meta: Any | None = None,
    dtype: DTypeLike | None = None,
) -> Scalar:
    """Instantiate a new scalar collection.

    Parameters
    ----------
    dsk : HighLevelGraph
        Dask highlevel task graph.
    name : str
        Name for the collection.
    meta : Any
        Awkward typetracer metadata.

    Returns
    -------
    Scalar
        Resulting collection.

    """

    if meta is not None and dtype is None:
        pass
    elif meta is None and dtype is not None:
        meta = ak.Array(create_unknown_scalar(dtype))
    else:
        ValueError("One (and only one) of dtype or meta can be defined.")

    if isinstance(meta, MaybeNone):
        meta = ak.Array(meta.content)
    elif meta is not None:
        try:
            if ak.backend(meta) != "typetracer":
                raise TypeError(
                    f"meta Scalar must have a typetracer backend, not {ak.backend(meta)}"
                )
        except AttributeError:
            raise TypeError("meta Scalar must have a typetracer backend; check failed")

    return Scalar(dsk, name, meta, known_value=None)


def new_known_scalar(
    s: Any,
    dtype: DTypeLike | None = None,
    label: str | None = None,
) -> Scalar:
    """Instantiate a Scalar with a known value.

    Parameters
    ----------
    s : Any
        Python object.
    dtype : DTypeLike, optional
        NumPy dtype associated with the object, if undefined the dtype
        will be assigned via NumPy's interpretation of the object.
    label : str, optional
        Label for the task graph; if undefined "known-scalar" will be
        used.

    Returns
    -------
    Scalar
        Resulting collection.

    Examples
    --------
    >>> from dask_awkward.core import new_known_scalar
    >>> a = new_known_scalar(5, label="five")
    >>> a
    dask.awkward<five, type=Scalar, dtype=int64, known_value=5>
    >>> a.compute()
    5

    """
    label = label or "known-scalar"
    name = f"{label}-{tokenize(s)}"
    if dtype is None:
        if isinstance(s, (int, np.integer)):
            dtype = np.dtype(int)
        elif isinstance(s, (float, np.floating)):
            dtype = np.dtype(float)
        else:
            dtype = np.dtype(type(s))
    else:
        dtype = np.dtype(dtype)
    llg = AwkwardMaterializedLayer({(name, 0): s}, previous_layer_names=[])
    hlg = HighLevelGraph.from_collections(name, llg, dependencies=())
    return Scalar(
        hlg,
        name,
        dtype=dtype,
        known_value=s,
    )


class Record(Scalar):
    """Single partition Dask collection representing a lazy Awkward Record.

    The class constructor is not intended for users. Instances of this
    class will be results from awkward operations.

    Within dask-awkward the ``new_record_object`` factory function is
    used for creating new instances.

    """

    def __init__(self, dsk: HighLevelGraph, name: str, meta: Any | None = None) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._meta: ak.Record = self._check_meta(meta)

    def _check_meta(self, m: Any | None) -> Any | None:
        if not isinstance(m, ak.Record):
            raise TypeError(f"meta must be a Record typetracer object, not a {type(m)}")
        return m

    def __getitem__(self, where):
        token = tokenize(self, where)
        new_name = f"{where}-{token}"
        new_meta = self._meta[where]

        # first check for array type return
        if isinstance(new_meta, ak.Array):
            graphlayer = {(new_name, 0): (operator.getitem, self.key, where)}
            hlg = HighLevelGraph.from_collections(
                new_name,
                graphlayer,
                dependencies=[self],
            )
            return new_array_object(hlg, new_name, meta=new_meta, npartitions=1)

        # then check for scalar (or record) type
        graphlayer = {(new_name, 0): (operator.getitem, self.key, where)}
        hlg = HighLevelGraph.from_collections(
            new_name,
            graphlayer,
            dependencies=[self],
        )
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, new_name, meta=new_meta)
        else:
            return new_scalar_object(hlg, new_name, meta=new_meta)

    def __getattr__(self, attr):
        if attr not in (self.fields or []):
            raise AttributeError(f"{attr} not in fields.")
        try:
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def __str__(self) -> str:
        return f"dask.awkward<{key_split(self.name)}, type=Record>"

    def __reduce__(self):
        return (Record, (self.dask, self.name, self._meta))

    @property
    def fields(self) -> list[str]:
        if self._meta is None:
            raise TypeError("metadata is missing; cannot determine fields.")
        return getattr(self._meta, "fields", None) or []

    @property
    def layout(self) -> Any:
        return self._meta.layout

    def _ipython_key_completions_(self) -> list[str]:
        if self._meta is not None:
            return self._meta._ipython_key_completions_()
        return []

    def __dir__(self) -> list[str]:
        fields = [] if self._meta is None else self._meta._layout.fields
        return sorted(
            set(
                [x for x in dir(type(self)) if not x.startswith("_")]
                + dir(super())
                + [
                    x
                    for x in fields
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )


def new_record_object(dsk: HighLevelGraph, name: str, *, meta: Any) -> Record:
    """Instantiate a new record collection.

    Parameters
    ----------
    dsk : HighLevelGraph
        Dask high level graph.
    name : str
        Name for the collection.
    meta : Any
        Awkward typetracer as metadata

    Returns
    -------
    Record
        Resulting collection.

    """
    out = Record(dsk, name, meta)
    if meta.__doc__ != meta.__class__.__doc__:
        out.__doc__ = meta.__doc__
    if ak.backend(meta) != "typetracer":
        raise TypeError(
            f"meta Record must have a typetracer backend, not {ak.backend(meta)}"
        )
    return Record(dsk, name, meta)


def _is_numpy_or_cupy_like(arr: Any) -> bool:
    return (
        hasattr(arr, "ndim")
        and hasattr(arr, "shape")
        and isinstance(arr.shape, tuple)
        and hasattr(arr, "dtype")
    )


def _finalize_array(results: Sequence[Any]) -> Any:
    # special cases for length 1 results
    if len(results) == 1:
        np_like = _is_numpy_or_cupy_like(results[0])
        if isinstance(results[0], (int, ak.Array)) or np_like:  # type: ignore[unreachable]
            return results[0]

    # a sequence of arrays that need to be concatenated.
    elif any(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)

    # a sequence of scalars that are stored as np.ndarray(N) where N
    # is a number (i.e. shapeless numpy array)
    elif any(_is_numpy_or_cupy_like(r) for r in results) and any(
        r.shape == () for r in results
    ):
        return ak.Array(list(results))

    # in awkward < 2.5 we can get integers instead of np.array scalars
    elif isinstance(results, (tuple, list)) and all(
        isinstance(r, (int, np.integer)) for r in results
    ):
        return ak.Array(list(results))

    # sometimes all partition results will be None (some write-to-disk
    # operations)
    elif all(r is None for r in results):
        return None

    else:
        msg = (
            "Unexpected results of a computation.\n "
            f"results: {results}"
            f"type of first result: {type(results[0])}"
        )
        raise RuntimeError(msg)


dak_cache = cachetools.LRUCache(maxsize=1000)


class Array(DaskMethodsMixin, NDArrayOperatorsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`~dask_awkward.from_parquet`,
    :py:func:`~dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: ak.Array,
        divisions: tuple[int, ...] | tuple[None, ...],
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._divisions: tuple[int, ...] | tuple[None, ...] = divisions
        self._meta: ak.Array = meta

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> NestedKeys:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> Hashable:
        return self.name

    def __dask_postcompute__(self) -> tuple[Callable, tuple]:
        return _finalize_array, ()

    def __dask_postpersist__(self):
        return self._rebuild, ()

    __dask_optimize__ = globalmethod(
        all_optimizations, key="awkward_array_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(threaded_get)

    def __bool__(self):
        raise ConcretizationTypeError(f"The __bool__() method was called on {self!r}.")

    def __int__(self):
        raise ConcretizationTypeError(f"The __int__() method was called on {self!r}.")

    def __float__(self):
        raise ConcretizationTypeError(f"The __float__() method was called on {self!r}.")

    def __complex__(self):
        raise ConcretizationTypeError(
            f"The __complex__() method was called on {self!r}."
        )

    def __index__(self):
        raise ConcretizationTypeError(f"The __index__() method was called on {self!r}.")

    def __setitem__(self, where: Any, what: Any) -> None:
        if not (
            isinstance(where, str)
            or (isinstance(where, tuple) and all(isinstance(x, str) for x in where))
        ):
            raise TypeError("only fields may be assigned in-place (by field name)")

        if not isinstance(what, (Array, Number)):
            raise DaskAwkwardNotImplemented(
                "Supplying anything other than a dak.Array, or Number to __setitem__ is not yet available!"
            )

        from dask_awkward.lib.structure import with_field

        appended = with_field(self, what, where=where, behavior=self.behavior)

        self._meta = appended._meta
        self._dask = appended._dask
        self._name = appended._name

    def _rebuild(self, dsk, *, rename=None):
        name = self.name
        if rename:
            name = rename.get(name, name)
        return Array(dsk, name, self._meta, divisions=self.divisions)

    def reset_meta(self) -> None:
        """Assign an empty typetracer array as the collection metadata."""
        self._meta = empty_typetracer()

    def repartition(
        self,
        npartitions: int | None = None,
        divisions: tuple[int, ...] | None = None,
        rows_per_partition: int | None = None,
        one_to_n: int | None = None,
        n_to_one: int | None = None,
    ) -> Array:
        """Restructure the partitioning of the whole array

        Various schemes are possible, with one of the mutually exclusive
        optional arguments for each. Of these, the first three require
        knowledge of the number of rows in each existing partition, which
        will be eagerly computed if not already known, and some shuffling of
        data between partitions.

        - npartitions: split all the rows as evenly as possible into this
          many output partitions.
        - divisions: exact row count offsets of each output partition
        - rows_per_partition: each partition will have this many rows,
          except the last, which will have this number or fewer
        - one_to_n: each input partition becomes n output partitions
        - n_to_one: every n adjacent input partitions becomes one
          output partition. Note that exactly one output partition
          (npartitions=1) is a special case of this.
        """
        from dask_awkward.layers import AwkwardMaterializedLayer
        from dask_awkward.lib.structure import (
            repartition_layer,
            simple_repartition_layer,
        )

        if (
            sum(
                bool(_)
                for _ in (
                    npartitions,
                    divisions,
                    rows_per_partition,
                    one_to_n,
                    n_to_one,
                )
            )
            != 1
        ):
            raise ValueError("Please specify exactly one of the inputs")
        new_divisions: tuple[int, ...] = ()
        if npartitions and npartitions == 1:
            npartitions, n_to_one = None, self.npartitions
        if n_to_one or one_to_n:
            token = tokenize(self, n_to_one, one_to_n)
            key = f"repartition-{token}"
            new_layer_raw, new_divisions = simple_repartition_layer(
                self, n_to_one, one_to_n, key
            )
        else:
            if not self.known_divisions:
                self.eager_compute_divisions()
            nrows = self.defined_divisions[-1]
            if divisions:
                if divisions == self.divisions:
                    # noop
                    return self
                new_divisions = divisions
            elif npartitions:
                rows_per_partition = math.ceil(nrows / npartitions)
            if rows_per_partition:
                new_divs = list(range(0, nrows, rows_per_partition))
                new_divs.append(nrows)
                new_divisions = tuple(new_divs)
            token = tokenize(self, divisions)
            key = f"repartition-{token}"
            new_layer_raw = repartition_layer(self, key, new_divisions)

        new_layer = AwkwardMaterializedLayer(
            new_layer_raw,
            previous_layer_names=[self.name],
        )
        new_graph = HighLevelGraph.from_collections(
            key, new_layer, dependencies=(self,)
        )
        return new_array_object(
            new_graph,
            key,
            meta=self._meta,
            behavior=self.behavior,
            divisions=tuple(new_divisions),
        )

    def __len__(self) -> int:
        if not self.known_divisions:
            raise TypeError(
                "Cannot determine length of collection with unknown partition sizes without executing the graph.\n"
                "Use `dask_awkward.num(..., axis=0)` if you want a lazy Scalar of the length.\n"
                "If you want to eagerly compute the partition sizes to have the ability to call `len` on the collection"
                ", use `.eager_compute_divisions()` on the collection."
            )
        return self.divisions[-1]  # type: ignore

    def _shorttypestr(self, max: int = 10) -> str:
        return str(_type(self))[0:max]

    def _typestr(self, max: int = 0) -> str:
        tstr = str(_type(self))
        if max and len(tstr) > max:
            tstr = f"{tstr[0:max]} ... }}"
        return f"var * {tstr}"

    def __str__(self) -> str:
        return (
            f"dask.awkward<{key_split(self.name)}, "
            f"npartitions={self.npartitions}"
            ">"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.__str__()

    def __iter__(self):
        raise NotImplementedError(
            "Iteration over a Dask Awkward collection is not supported.\n"
            "A suggested alternative: define a function which iterates over\n"
            "an awkward array and use that function with map_partitions."
        )

    def _ipython_display_(self):
        return self._meta._ipython_display_()  # pragma: no cover

    def _ipython_canary_method_should_not_exist_(self):
        return self._meta._ipython_canary_method_should_not_exist_()  # pragma: no cover

    def _repr_mimebundle_(self):
        return self._meta._repr_mimebundle_()  # pragma: no cover

    def _ipython_key_completions_(self) -> list[str]:
        if self._meta is not None:
            return self._meta._ipython_key_completions_()
        return []

    def __dir__(self) -> list[str]:
        fields = [] if self._meta is None else self._meta._layout.fields
        return sorted(
            set(
                [x for x in dir(type(self)) if not x.startswith("_")]
                + dir(super())
                + [
                    x
                    for x in fields
                    if _dir_pattern.match(x) and not keyword.iskeyword(x)
                ]
            )
        )

    def __reduce__(self):
        return (Array, (self.dask, self.name, self._meta, self.divisions))

    @property
    def dask(self) -> HighLevelGraph:
        """High level task graph associated with the collection."""
        return self._dask

    @property
    def keys(self) -> NestedKeys:
        """Task graph keys."""
        return self.__dask_keys__()

    @property
    def name(self) -> str:
        """Name of the collection."""
        return self._name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        assert self._meta is not None
        return self._meta.ndim

    @property
    def divisions(self) -> tuple[int, ...] | tuple[None, ...]:
        """Location of the collections partition boundaries."""
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        """True if the divisions are known (absence of ``None`` in the tuple)."""
        return len(self.divisions) > 0 and None not in self.divisions

    @property
    def defined_divisions(self) -> tuple[int, ...]:
        if not self.known_divisions:
            raise ValueError("defined_divisions only works when divisions are known.")
        return self._divisions  # type: ignore

    @property
    def npartitions(self) -> int:
        """Total number of partitions."""
        return len(self.divisions) - 1

    @property
    def layout(self) -> Content:
        """awkward Array layout associated with the eventual computed result."""
        if self._meta is not None:
            return self._meta.layout
        raise ValueError("This collection's meta is None; unknown layout.")

    @property
    def attrs(self) -> Mapping[str, Any]:
        """awkward Array attrs dictionary."""
        if self._meta is not None:
            return MappingProxyType(self._meta.attrs)
        raise ValueError("This collection's meta is None; no attrs property available.")

    @property
    def behavior(self) -> Mapping | None:
        """awkward Array behavior dictionary."""
        if self._meta is not None:
            behavior = self._meta.behavior
            return None if behavior is None else MappingProxyType(behavior)
        raise ValueError(
            "This collection's meta is None; no behavior property available."
        )

    @property
    def mask(self) -> AwkwardMask:
        return AwkwardMask(self)

    @property
    def fields(self) -> list[str]:
        """Record field names (if any)."""
        return getattr(self._meta, "fields", None) or []

    @property
    def form(self) -> Form:
        """awkward Array form associated with the eventual computed result."""
        if self._meta is not None:
            return self._meta.layout.form
        raise ValueError("This collection's meta is None; unknown form.")

    @property
    def type(self) -> ArrayType:
        """awkward Array type associated with the eventual computed result."""
        t = ak.types.ArrayType(
            self._meta._layout.form.type,
            0,
            behavior=self._meta._behavior,
        )
        t._length = "??"
        return t

    @cached_property
    def keys_array(self) -> np.ndarray:
        """NumPy array of task graph keys."""
        return np.array(self.__dask_keys__(), dtype=object)

    def _partitions(self, index: Any) -> Array:
        # TODO: this produces a materialized layer, but could work like repartition() and slice()
        if not isinstance(index, tuple):
            index = (index,)
        token = tokenize(self, index)
        from dask.array.slicing import normalize_index

        raw = normalize_index(index, (self.npartitions,))
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in raw)  # type: ignore
        name = f"partitions-{token}"
        new_keys = self.keys_array[index].tolist()
        dsk = {(name, i): tuple(key) for i, key in enumerate(new_keys)}
        graph = HighLevelGraph.from_collections(
            name,
            AwkwardMaterializedLayer(dsk, previous_layer_names=[self.name]),
            dependencies=(self,),
        )

        # if a single partition was requested we trivially know the new divisions.
        if len(raw) == 1 and isinstance(raw[0], int) and self.known_divisions:
            # TODO: don't we always know the divisions?
            new_divisions = (
                0,
                self.divisions[raw[0] + 1] - self.divisions[raw[0]],  # type: ignore
            )
        # otherwise nullify the known divisions
        else:
            new_divisions = (None,) * (len(new_keys) + 1)  # type: ignore

        return new_array_object(
            graph, name, meta=self._meta, divisions=tuple(new_divisions)
        )

    @property
    def partitions(self) -> IndexCallable:
        """Get a specific partition or slice of partitions.

        Returns
        -------
        dask.utils.IndexCallable

        Examples
        --------
        >>> import dask_awkward as dak
        >>> import awkward as ak
        >>> aa = ak.Array([[1, 2, 3], [], [2]])
        >>> a = dak.from_awkward(aa, npartitions=3)
        >>> a
        dask.awkward<from-awkward, npartitions=3>
        >>> a.partitions[0]
        dask.awkward<partitions, npartitions=1>
        >>> a.partitions[0:2]
        dask.awkward<partitions, npartitions=2>
        >>> a.partitions[2].compute()
        <Array [[2]] type='1 * var * int64'>

        """
        return IndexCallable(self._partitions)

    def _getitem_trivial_map_partitions(
        self,
        where: Any,
        meta: Any | None = None,
        label: str | None = None,
    ) -> Any:
        if meta is None and self._meta is not None:
            if isinstance(where, tuple):
                metad = to_meta(where)
                meta = self._meta[metad]
            else:
                m = to_meta([where])[0]
                meta = self._meta[m]
        return _map_partitions(
            operator.getitem,
            self,
            where,
            meta=meta,
            output_divisions=1,
            label=label,
        )

    def _getitem_outer_bool_or_int_lazy_array(self, where):
        ba = where if isinstance(where, Array) else where[0]
        if partition_compatibility(self, ba) == PartitionCompatibility.NO:
            raise IncompatiblePartitions("getitem", self, ba)

        if isinstance(where, tuple):
            raise DaskAwkwardNotImplemented(
                "tuple style input boolean/int selection is not supported."
            )

        new_meta = self._meta[where._meta]
        return self._map_partitions(
            operator.getitem,
            where,
            meta=new_meta,
        )

    def _getitem_outer_str_or_list(
        self,
        where: str | list | tuple[Any, ...],
        label: str | None = None,
    ) -> Any:
        new_meta: Any | None = None
        if self._meta is not None:
            if isinstance(where, tuple):
                if not isinstance(where[0], (str, list)):
                    raise TypeError("Expected where[0] to be a string or list")
                metad = to_meta(where)
                new_meta = self._meta[metad]
            elif isinstance(where, (str, list)):
                new_meta = self._meta[where]
        return self._getitem_trivial_map_partitions(where, meta=new_meta, label=label)

    def _getitem_outer_int(self, where: int | tuple[Any, ...]) -> Any:
        if where == 0 or (isinstance(where, tuple) and where[0] == 0):
            pass
        elif not self.known_divisions:
            self.eager_compute_divisions()

        new_meta: Any | None = None
        # multiple objects passed to getitem. collections passed in
        # the tuple of objects have not been tested!
        if isinstance(where, tuple):
            if not isinstance(where[0], int):
                raise TypeError("Expected where[0] to be and integer.")
            if where[0] == 0:
                pidx, outer_where = 0, 0
            else:
                pidx, outer_where = normalize_single_outer_inner_index(
                    self.divisions, where[0]  # type: ignore
                )
            partition = self.partitions[pidx]
            rest = where[1:]
            where = (outer_where, *rest)
            if partition._meta is not None:
                metad = to_meta(where)
                new_meta = partition._meta[metad]
        # single object passed to getitem
        elif isinstance(where, int):
            if where == 0:
                pidx, where = 0, 0
            else:
                pidx, where = normalize_single_outer_inner_index(self.divisions, where)  # type: ignore
            partition = self.partitions[pidx]
            if partition._meta is not None:
                new_meta = partition._meta[where]
                # new_meta = make_unknown_length(partition._meta)[where]
                # new_meta = ak.Array(
                #     ak.to_backend(
                #         partition._meta,
                #         "typetracer",
                #         highlevel=False,
                #     ).to_typetracer(forget_length=True)
                # )[where]

        # if we know a new array is going to be made, just call the
        # trivial inner on the new partition.
        if isinstance(new_meta, ak.Array):
            result = partition._getitem_trivial_map_partitions(where, meta=new_meta)
            result._divisions = (0, None)
            return result

        # otherwise make sure we have one of the other potential results.
        if not isinstance(new_meta, (ak.Record, TypeTracerArray, OneOf, MaybeNone)):
            raise DaskAwkwardNotImplemented("Key type not supported for this array.")

        token = tokenize(partition, where)
        name = f"getitem-{token}"
        dsk = {
            (name, 0): (
                operator.getitem,
                partition.__dask_keys__()[0],
                where,
            )
        }
        hlg = HighLevelGraph.from_collections(
            name,
            AwkwardMaterializedLayer(dsk, previous_layer_names=[self.name]),
            dependencies=[partition],
        )
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, name, meta=new_meta)
        else:
            return new_scalar_object(hlg, name, meta=new_meta)

    def _getitem_slice_on_zero(self, where):
        # normalise
        sl = where[0]
        rest = tuple(where[1:])
        step = sl.step or 1
        start = sl.start or 0

        if not self.known_divisions:
            self.eager_compute_divisions()
        stop = sl.stop or self.defined_divisions[-1]
        start = start if start >= 0 else self.defined_divisions[-1] + start
        stop = stop if stop >= 0 else self.defined_divisions[-1] + stop
        if step < 0:
            raise DaskAwkwardNotImplemented("negative step slice on zeroth dimension")

        # setup
        token = tokenize(self, where)
        name = f"getitem-{token}"
        remainder = 0
        outpart = 0
        divisions = [0]
        dask = {}
        # make low-level graph
        for i in range(self.npartitions):
            if start > self.defined_divisions[i + 1]:
                # first partition not yet found
                continue
            if stop < self.defined_divisions[i] and dask:
                # no more partitions with valid rows
                # does **NOT** exit if there are no partitions yet, to make sure there is always
                # at least one, needed to get metadata of empty output right
                break
            slice_start = max(start - self.defined_divisions[i], 0 + remainder)
            slice_end = min(
                stop - self.defined_divisions[i],
                self.defined_divisions[i + 1] - self.defined_divisions[i],
            )
            if (
                slice_end == slice_start
                and (self.defined_divisions[i + 1] - self.defined_divisions[i])
                and dask
            ):
                # in case of zero-row last partition (if not only partition)
                break
            dask[(name, outpart)] = (
                _zero_getitem,
                (self.name, i),
                slice(slice_start, slice_end, step),
                rest,
            )
            outpart += 1
            remainder = (
                (self.defined_divisions[i] + slice_start)
                - self.defined_divisions[i + 1]
            ) % step
            remainder = step - remainder if remainder < 0 else remainder
            nextdiv = math.ceil((slice_end - slice_start) / step)
            divisions.append(divisions[-1] + nextdiv)

        hlg = HighLevelGraph.from_collections(
            name,
            AwkwardMaterializedLayer(dask, previous_layer_names=[self.name]),
            dependencies=[self],
        )
        return new_array_object(
            hlg,
            name,
            meta=self._meta,
            behavior=self.behavior,
            divisions=tuple(divisions),
        )

    def _getitem_tuple(self, where):
        where, n_field_accesses = field_access_to_front(where)

        if isinstance(where[0], int):
            return self._getitem_outer_int(where)

        elif isinstance(where[0], (str, list)):
            first, rest = where[:n_field_accesses], where[n_field_accesses:]
            if rest:
                return self._getitem_trivial_map_partitions(first)[rest]
            return self._getitem_trivial_map_partitions(first)

        elif isinstance(where[0], slice) and is_empty_slice(where[0]):
            return self._getitem_trivial_map_partitions(where)

        elif isinstance(where[0], slice):
            return self._getitem_slice_on_zero(where)
        # boolean array
        elif isinstance(where[0], Array):
            try:
                dtype = where[0].layout.dtype.type
            except AttributeError:
                dtype = where[0].layout.content.dtype.type
            if issubclass(dtype, (np.bool_, numbers.Integral)):
                return self._getitem_outer_bool_or_int_lazy_array(where)

        elif where[0] is Ellipsis:
            if len(where) <= self.ndim:
                return self._getitem_trivial_map_partitions(where)

            raise DaskAwkwardNotImplemented(
                "Array slicing doesn't currently support Ellipsis where "
                "the total number of sliced axes is greater than the "
                "dimensionality of the array."
            )

        raise DaskAwkwardNotImplemented(
            f"Array.__getitem__ doesn't support multi object: {where}"
        )

    def _getitem_single(self, where):
        # a single string
        if isinstance(where, str):
            return self._getitem_outer_str_or_list(where, label=where)

        # an empty slice

        elif is_empty_slice(where):
            return self

        elif isinstance(where, list):
            return self._getitem_outer_str_or_list(where)

        elif isinstance(where, slice):
            return self._getitem_slice_on_zero((where,))

        # a single integer
        elif isinstance(where, int):
            return self._getitem_outer_int(where)

        elif isinstance(where, Array):
            layout = where.layout
            while not hasattr(layout, "dtype"):
                layout = layout.content
            dtype = layout.dtype.type
            if issubclass(dtype, (np.bool_, numbers.Integral)):
                return self._getitem_outer_bool_or_int_lazy_array(where)

        # a single ellipsis
        elif where is Ellipsis:
            return self

        elif self.npartitions == 1:
            return self.map_partitions(operator.getitem, where)

        raise DaskAwkwardNotImplemented(f"__getitem__ doesn't support where={where}.")

    def __getitem__(self, where):
        """Select items from the collection.

        Heavily under construction.

        Arguments
        ---------
        where : many types supported
            Selection criteria.

        Returns
        -------
        Array | Record | Scalar
            Resulting collection.

        """

        # don't accept lists containing integers.
        if isinstance(where, list):
            if any(isinstance(k, int) for k in where):
                # this is something we'll likely never support so we
                # do not use the DaskAwkwardNotImplemented exception.
                raise RuntimeError("Lists containing integers are not supported.")

        if isinstance(where, tuple):
            return self._getitem_tuple(where)

        return self._getitem_single(where)

    def _is_method_heuristic(self, resolved: Any) -> bool:
        return callable(resolved)

    def __getattr__(self, attr: str) -> Any:
        if attr not in (self.fields or []):
            try:
                cls_method = getattr_static(self._meta, attr)
            except AttributeError:
                raise AttributeError(f"{attr} not in fields.")
            else:
                if hasattr(cls_method, "_dask_get"):
                    return cls_method._dask_get(self._meta, type(self._meta), self)
                elif self._is_method_heuristic(cls_method):

                    @wraps(cls_method)
                    def wrapper(*args, **kwargs):
                        return self.map_partitions(
                            _BehaviorMethodFn(attr, **kwargs),
                            *args,
                            label=hyphenize(attr),
                        )

                    return wrapper
                else:
                    return self.map_partitions(
                        _BehaviorPropertyFn(attr),
                        label=hyphenize(attr),
                    )
        try:
            # at this point attr is either a field or we'll have to
            # raise an exception.
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def _map_partitions(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Array:
        """Maps a function to partitions without flattening the function inputs."""
        return _map_partitions(func, self, *args, **kwargs)

    def map_partitions(
        self,
        func: Callable,
        *args: Any,
        traverse: bool = True,
        **kwargs: Any,
    ) -> Array:
        """Map a function across all partitions of the collection.

        Parameters
        ----------
        func : Callable
            Function to call on all partitions.
        *args : Collections and function arguments
            Additional arguments passed to `func` after the
            collection, if arguments are Array collections
            they must be compatibly partitioned with the object this
            method is being called from.
        traverse : bool
            Unpack basic python containers to find dask collections.
        **kwargs : Any
            Additional keyword arguments passed to the `func`.

        Returns
        -------
        dask_awkward.Array
            The new collection.

        See Also
        --------
        dask_awkward.map_partitions

        """
        return map_partitions(func, self, *args, traverse=traverse, **kwargs)

    def eager_compute_divisions(self) -> None:
        """Force a compute of the divisions."""
        self._divisions = calculate_known_divisions(self)

    def clear_divisions(self) -> None:
        """Clear the divisions of a Dask Awkward Collection."""
        self._divisions = (None,) * (self.npartitions + 1)

    def __awkward_function__(self, func, array_likes, args, kwargs):
        import dask_awkward

        if any(isinstance(arg, ak.Array) for arg in array_likes):
            raise TypeError("cannot mix awkward.Array and dask_awkward.Array")

        fn_name = func.__qualname__
        try:
            fn = getattr(dask_awkward, fn_name)
        except AttributeError:
            try:
                import dask_awkward.lib.str

                fn = getattr(dask_awkward.str, fn_name)
            except AttributeError:
                return NotImplemented
        return fn(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise RuntimeError("Array ufunc supports only method == '__call__'")

        dak_arrays = tuple(a for a in inputs if isinstance(a, Array))
        if partition_compatibility(*dak_arrays) == PartitionCompatibility.NO:
            raise IncompatiblePartitions(*dak_arrays)

        return _map_partitions(
            ufunc,
            *inputs,
            output_divisions=1,
            **kwargs,
        )

    def __array__(self, *_, **__):
        raise NotImplementedError

    def to_delayed(self, optimize_graph: bool = True) -> list[Delayed]:
        """Convert the collection to a list of delayed objects.

        One dask.delayed.Delayed object per partition.

        Parameters
        ----------
        optimize_graph : bool
            If True the task graph associated with the collection will
            be optimized before conversion to the list of Delayed
            objects.

        See Also
        --------
        dask_awkward.to_delayed

        Returns
        -------
        list[Delayed]
            List of delayed objects (one per partition).

        """
        from dask_awkward.lib.io.io import to_delayed

        return to_delayed(self, optimize_graph=optimize_graph)

    def to_dask_array(
        self,
        *,
        dtype: Any = None,
        optimize_graph: bool = True,
    ) -> DaskArray:
        from dask_awkward.lib.io.io import to_dask_array

        return to_dask_array(self, dtype=dtype, optimize_graph=optimize_graph)

    def to_parquet(
        self,
        path: str,
        storage_options: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        from dask_awkward.lib.io.parquet import to_parquet

        return to_parquet(self, path, storage_options=storage_options, **kwargs)

    def to_dask_bag(self) -> DaskBag:
        from dask_awkward.lib.io.io import to_dask_bag

        return to_dask_bag(self)

    def head(self, nrow=10, compute=True):
        """First few rows of the array

        These rows are taken only from the first partition for simplicity. If that partition
        has fewer rows than ``nrow``, no attempt is made to fetch more from subsequent
        partitions.

        By default this is then processed eagerly and returned.
        """
        out = self.partitions[0].map_partitions(lambda x: x[:nrow], meta=self._meta)
        if compute:
            return out.compute()
        if self.known_divisions:
            out._divisions = (0, min(nrow, self.defined_divisions[1]))
        return out


class AwkwardMask:
    def __init__(self, array: Array) -> None:
        self.array = array

    def __getitem__(self, where: Array) -> Array:
        from dask_awkward.lib.structure import mask

        return mask(self.array, where, valid_when=True)


def _zero_getitem(arr: ak.Array, zeroth: slice, rest: tuple[slice, ...]) -> ak.Array:
    return arr.__getitem__((zeroth,) + rest)


def compute_typetracer(dsk: HighLevelGraph, name: str) -> ak.Array:
    key = (name, 0)
    return typetracer_array(
        Delayed(
            key,
            dsk.cull({key}),
            layer=name,
        ).compute()
    )


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    *,
    meta: ak.Array | None = None,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
    npartitions: int | None = None,
    divisions: tuple[int, ...] | tuple[None, ...] | None = None,
) -> Array:
    """Instantiate a new Array collection object.

    Parameters
    ----------
    dsk : dask.highlevelgraph.HighLevelGraph
        Graph backing the collection.
    name : str
        Unique name for the collection.
    meta : Array, optional
        Collection metadata; this is an awkward-array type tracer. If
        `meta` is ``None``, the first partition of the task graph
        (`dsk`) will be computed by default to determine the
        typetracer for the new Array. If the configuration option
        ``awkward.compute-unknown-meta`` is set to ``False``,
        undefined `meta` will be assigned an empty typetracer.
    behavior : dict, optional
        Custom ak.behavior for the output array.
    attrs : dict, optional
        Custom attributes for the output array.
    npartitions : int, optional
        Total number of partitions; if used `divisions` will be a
        tuple of length `npartitions` + 1 with all elements``None``.
    divisions : tuple[int | None, ...], optional
        Tuple identifying the locations of the divisions between the
        partitions.

    Returns
    -------
    Array
        Resulting collection.

    """
    if divisions is None:
        if npartitions is not None:
            divs: tuple[int, ...] | tuple[None, ...] = (None,) * (npartitions + 1)
        else:
            raise ValueError("One of either divisions or npartitions must be defined.")
    else:
        if npartitions is not None:
            raise ValueError(
                "Only one of either divisions or npartitions can be defined."
            )
        divs = divisions

    if meta is None:
        if dask.config.get("awkward.compute-unknown-meta"):
            actual_meta = compute_typetracer(dsk, name)
        else:
            actual_meta = empty_typetracer()
    else:
        if not isinstance(meta, ak.Array):
            raise TypeError(
                f"meta must be an instance of an Awkward Array, not {type(meta)}."
            )
        if ak.backend(meta) != "typetracer":
            raise TypeError(
                f"meta Array must have a typetracer backend, not {ak.backend(meta)}"
            )
        actual_meta = meta

    if behavior is not None:
        actual_meta.behavior = behavior
    if attrs is not None:
        actual_meta.attrs = attrs

    out = Array(dsk, name, actual_meta, divs)
    if actual_meta.__doc__ != actual_meta.__class__.__doc__:
        out.__doc__ = actual_meta.__doc__

    return out


def partitionwise_layer(
    func: Callable,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> AwkwardBlockwiseLayer:
    """Create a partitionwise graph layer.

    Parameters
    ----------
    func : Callable
        Function to apply on all partitions.
    name : str
        Name for the layer.
    *args : Any
        Arguments that will be passed to `func`.
    **kwargs : Any
        Keyword arguments that will be passed to `func`.

    Returns
    -------
    dask.blockwise.Blockwise
        The Dask HighLevelGraph Blockwise layer.

    """
    pairs: list[Any] = []
    numblocks: dict[str, tuple[int, ...]] = {}
    for arg in args:
        if isinstance(arg, Array):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (arg.npartitions,)
        elif isinstance(arg, BlockwiseDep):
            if len(arg.numblocks) == 1:
                pairs.extend([arg, "i"])
            elif len(arg.numblocks) == 2:
                pairs.extend([arg, "ij"])
        elif is_arraylike(arg) and is_dask_collection(arg) and arg.ndim == 1:
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = arg.numblocks
        elif isinstance(arg, Scalar):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (1,)
        elif isinstance(arg, Delayed):
            if _dask_uses_tasks:
                pairs.extend([TaskRef(arg.key), None])
            else:
                pairs.extend([arg.key, None])
        elif is_dask_collection(arg):
            raise DaskAwkwardNotImplemented(
                "Use of Array with other Dask collections is currently unsupported."
            )
        else:
            pairs.extend([arg, None])
    layer = dask_blockwise(
        func,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )
    layer = AwkwardBlockwiseLayer.from_blockwise(layer)
    return layer


class ArgsKwargsPackedFunction:
    def __init__(self, the_fn, arg_repackers, kwarg_repacker, arg_lens_for_repackers):
        self.fn = the_fn
        self.arg_repackers = arg_repackers
        self.kwarg_repacker = kwarg_repacker
        self.arg_lens_for_repackers = arg_lens_for_repackers

    def __call__(self, *args_deps_expanded):
        """This packing function receives a list of strictly
        ordered arguments. The first range of arguments,
        [0:sum(self.arg_lens_for_repackers)], corresponding to
        the origin *args of self.fn but flattened to a list of
        dask collections and non-dask-collection-containing arguments.
        The remainder are the dask-collection-deps of self.fn's original
        kwargs. The lengths of expected flattened inputs for each arg are
        specified when this class is created, and we use that to process
        the input flattened list of arguments sequentially.

        The various repackers deal with restructuring the received flattened
        list into the shape that self.fn expects.
        """
        args = []
        len_args = 0
        for repacker, n_args in zip(self.arg_repackers, self.arg_lens_for_repackers):
            args.append(
                repacker(args_deps_expanded[len_args : len_args + n_args])[0]
                if repacker is not None
                else args_deps_expanded[len_args]
            )
            len_args += n_args
        kwargs = self.kwarg_repacker(args_deps_expanded[len_args:])[0]
        return self.fn(*args, **kwargs)


def _map_partitions(
    fn: Callable,
    *args: Any,
    label: str | None = None,
    token: str | None = None,
    meta: Any | None = None,
    output_divisions: int | None = None,
    **kwargs: Any,
) -> Array:
    """Map a callable across all partitions of any number of collections.
    No wrapper is used to flatten the function arguments. This is meant for
    dask-awkward internal use or in situations where input data are sanitized.

    The parameters of this function are otherwise the same as map_partitions,
    but the limitation that args, kwargs must be non-nested and flat. They
    will not be traversed to extract all dask collections, except those in
    the first dimension of args or kwargs.
    """
    token = token or tokenize(fn, *args, output_divisions, **kwargs)
    label = hyphenize(label or funcname(fn))
    name = f"{label}-{token}"
    deps = [a for a in args if is_dask_collection(a)] + [
        v for v in kwargs.values() if is_dask_collection(v)
    ]
    dak_arrays = tuple(filter(lambda x: isinstance(x, Array), deps))

    if name in dak_cache:
        hlg, meta = dak_cache[name]
    else:
        lay = partitionwise_layer(
            fn,
            name,
            *args,
            **kwargs,
        )

        if meta is None:
            meta = map_meta(fn, *args, **kwargs)

        hlg = HighLevelGraph.from_collections(
            name,
            lay,
            dependencies=deps,
        )

        if len(dak_arrays) == 0:
            raise TypeError(
                "at least one argument passed to map_partitions "
                "should be a dask_awkward.Array collection."
            )
        dak_cache[name] = hlg, meta
    in_npartitions = dak_arrays[0].npartitions
    in_divisions = dak_arrays[0].divisions
    if output_divisions is not None:
        if output_divisions == 1:
            new_divisions = dak_arrays[0].divisions
        else:
            new_divisions = tuple(map(lambda x: x * output_divisions, in_divisions))
    else:
        new_divisions = in_divisions

    if output_divisions is not None:
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            divisions=new_divisions,
        )
    else:
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            npartitions=in_npartitions,
        )


def map_partitions(
    base_fn: Callable,
    *args: Any,
    label: str | None = None,
    token: str | None = None,
    meta: Any | None = None,
    output_divisions: int | None = None,
    traverse: bool = True,
    **kwargs: Any,
) -> Array:
    """Map a callable across all partitions of any number of collections.

    Parameters
    ----------
    base_fn : Callable
        Function to apply on all partitions, this will get wraped to
        handle kwargs, including dask collections.
    *args : Collections and function arguments
        Arguments passed to the function. Partitioned arguments (i.e.
        Dask collections) will have `fn` applied to each partition.
        Array collection arguments they must be compatibly
        partitioned.
    label : str, optional
        Label for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    token : str, optional
        Provide an already defined token. If ``None`` a new token will
        be generated.
    meta : Any, optional
        Metadata (typetracer) array for the result (if known). If
        unknown, `fn` will be applied to the metadata of the `args`;
        if that call fails, the first partition of the new collection
        will be used to compute the new metadata **if** the
        ``awkward.compute-known-meta`` configuration setting is
        ``True``. If the configuration setting is ``False``, an empty
        typetracer will be assigned as the metadata.
    output_divisions : int, optional
        If ``None`` (the default), the divisions of the output will be
        assumed unknown. If defined, the output divisions will be
        multiplied by a factor of `output_divisions`. A value of 1
        means constant divisions (e.g. a string based slice). Any
        value greater than 1 means the divisions were expanded by some
        operation. This argument is mainly for internal library
        function implementations.
    traverse : bool
        Unpack basic python containers to find dask collections.
    **kwargs : Any
        Additional keyword arguments passed to the `fn`.

    Returns
    -------
    dask_awkward.Array
        The new collection.

    Examples
    --------
    >>> import dask_awkward as dak
    >>> a = [[1, 2, 3], [4]]
    >>> b = [[5, 6, 7], [8]]
    >>> c = dak.from_lists([a, b])
    >>> c
    dask.awkward<from-lists, npartitions=2>
    >>> c.compute()
    <Array [[1, 2, 3], [4], [5, 6, 7], [8]] type='4 * var * int64'>
    >>> c2 = dak.map_partitions(np.add, c, c)
    >>> c2
    dask.awkward<add, npartitions=2>
    >>> c2.compute()
    <Array [[2, 4, 6], [8], [10, 12, 14], [16]] type='4 * var * int64'>

    Multiplying `c` (a Dask collection) with `a` (a regular Python
    list object) will multiply each partition of `c` by `a`:

    >>> d = dak.map_partitions(np.multiply, c, a)
    dask.awkward<multiply, npartitions=2>
    >>> d.compute()
    <Array [[1, 4, 9], [16], [5, 12, 21], [32]] type='4 * var * int64'>

    This is effectively the same as `d = c * a`
    """

    opt_touch_all = kwargs.pop("opt_touch_all", None)
    if opt_touch_all is not None:
        warnings.warn(
            "The opt_touch_all argument does nothing.\n"
            "This warning will be removed in a future version of dask-awkward "
            "and the function call will likely fail."
        )

    kwarg_flat_deps, kwarg_repacker = unpack_collections(kwargs, traverse=traverse)
    flat_deps, _ = unpack_collections(*args, *kwargs.values(), traverse=traverse)

    if len(flat_deps) == 0:
        message = (
            "map_partitions expects at least one Dask collection instance, "
            "you are passing non-Dask collections to dask-awkward code.\n"
            "observed argument types:\n"
        )
        for arg in args:
            message += f"- {type(arg)}"
        raise TypeError(message)

    if len(kwargs) == 0:  # non-critical FIXME: use kwarg_flat_deps
        non_traversed_deps, _ = unpack_collections(*args, traverse=False)
        if len(flat_deps) == len(non_traversed_deps) and all(
            id(traversed_dep) == id(non_traversed_dep)
            for traversed_dep, non_traversed_dep in zip(flat_deps, non_traversed_deps)
        ):
            return _map_partitions(
                base_fn,
                *args,
                label=label,
                token=token,
                meta=meta,
                output_divisions=output_divisions,
                **kwargs,
            )

    arg_flat_deps_expanded = []
    arg_repackers = []
    arg_lens_for_repackers = []
    for arg in args:
        this_arg_flat_deps, repacker = unpack_collections(arg, traverse=traverse)
        if (
            len(this_arg_flat_deps) > 0
        ):  # if the deps list is empty this arg does not contain any dask collection, no need to repack!
            arg_flat_deps_expanded.extend(this_arg_flat_deps)
            arg_repackers.append(repacker)
            arg_lens_for_repackers.append(len(this_arg_flat_deps))
        else:
            arg_flat_deps_expanded.append(arg)
            arg_repackers.append(None)
            arg_lens_for_repackers.append(1)

    fn = ArgsKwargsPackedFunction(
        base_fn,
        arg_repackers,
        kwarg_repacker,
        arg_lens_for_repackers,
    )
    return _map_partitions(
        fn,
        *arg_flat_deps_expanded,
        *kwarg_flat_deps,
        label=label,
        token=token,
        meta=meta,
        output_divisions=output_divisions,
    )


def _chunk_reducer_non_positional(
    chunk: ak.Array,
    is_axis_none: bool,
    *,
    reducer: Callable,
    mask_identity: bool,
) -> ak.Array:
    return reducer(
        chunk,
        keepdims=True,
        axis=-1 if is_axis_none else 0,
        mask_identity=mask_identity,
    )


def _concat_reducer_non_positional(
    partials: list[ak.Array], is_axis_none: bool
) -> ak.Array:
    concat_axis = -1 if is_axis_none else 0
    return ak.concatenate(partials, axis=concat_axis)


def _finalise_reducer_non_positional(
    partial: ak.Array,
    is_axis_none: bool,
    *,
    reducer: Callable,
    mask_identity: bool,
    keepdims: bool,
) -> ak.Array:
    return reducer(
        partial,
        axis=None if is_axis_none else 0,
        keepdims=keepdims,
        mask_identity=mask_identity,
    )


def _prepare_axis_none_chunk(chunk: ak.Array) -> ak.Array:
    # TODO: this is private Awkward code. We should figure out how to export it
    # if needed
    (layout,) = ak_do_remove_structure(
        ak.to_layout(chunk),
        flatten_records=False,
        drop_nones=False,
        keepdims=True,
        allow_records=False,
    )
    return ak.Array(layout, behavior=chunk.behavior)


def non_trivial_reduction(
    *,
    label: str,
    array: Array,
    axis: int | None,
    is_positional: bool,
    keepdims: bool,
    mask_identity: bool,
    reducer: Callable,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
    combiner: Callable | None = None,
    token: str | None = None,
    dtype: Any | None = None,
    split_every: int | bool | None = None,
) -> Array | Scalar:
    if is_positional:
        raise NotImplementedError("positional reducers at axis=0 or axis=None")

    # Regularise the axis to (0, None)
    if axis == 0 or axis == -1 * array.ndim:
        axis = 0
    elif axis is not None:
        raise ValueError(axis)

    if combiner is None:
        combiner = reducer

    # is_positional == True is not implemented
    # if is_positional:
    #     assert combiner is reducer

    # For `axis=None`, we prepare each array to have the following structure:
    #   [[[ ... [x1 x2 x3 ... xN] ... ]]] (length-1 outer lists)
    # This makes the subsequent reductions an `axis=-1` reduction
    if axis is None:
        prepared_array = map_partitions(_prepare_axis_none_chunk, array)
    else:
        prepared_array = array

    from dask_awkward.layers import AwkwardTreeReductionLayer

    token = token or tokenize(
        array,
        reducer,
        combiner,
        label,
        dtype,
        split_every,
        axis,
        mask_identity,
        keepdims,
    )
    name_tree_node = f"{label}-tree-node-{token}"
    name_finalize = f"{label}-finalize-{token}"

    chunked_fn = partial(
        _chunk_reducer_non_positional,
        reducer=reducer,
        is_axis_none=axis is None,
        mask_identity=mask_identity,
    )
    tree_node_fn = partial(
        _chunk_reducer_non_positional,
        reducer=combiner,
        is_axis_none=axis is None,
        mask_identity=mask_identity,
    )
    concat_fn = partial(_concat_reducer_non_positional, is_axis_none=axis is None)
    finalize_fn = partial(
        _finalise_reducer_non_positional,
        reducer=combiner,
        is_axis_none=axis is None,
        keepdims=keepdims,
        mask_identity=mask_identity,
    )

    if split_every is None:
        split_every = dask.config.get("awkward.aggregation.split-every", 8)
    elif split_every is False:
        split_every = sys.maxsize
    else:
        pass

    chunked = map_partitions(chunked_fn, prepared_array, meta=empty_typetracer())

    trl = AwkwardTreeReductionLayer(
        name=name_finalize,
        name_input=chunked.name,
        npartitions_input=prepared_array.npartitions,
        concat_func=concat_fn,
        tree_node_func=tree_node_fn,
        finalize_func=finalize_fn,
        split_every=split_every,
        tree_node_name=name_tree_node,
    )

    graph = HighLevelGraph.from_collections(name_finalize, trl, dependencies=(chunked,))

    meta = reducer(
        array._meta,
        axis=axis,
        keepdims=keepdims,
        mask_identity=mask_identity,
    )
    if isinstance(meta, ak.highlevel.Array):
        return new_array_object(graph, name_finalize, meta=meta, npartitions=1)
    else:
        return new_scalar_object(graph, name_finalize, meta=meta)


def calculate_known_divisions(array: Array) -> tuple[int, ...]:
    """Determine the divisions of a collection.

    This function triggers an immediate computation.

    Parameters
    ----------
    array : dask_awkward.Array
        Awkard array collection.

    Returns
    -------
    tuple[int, ...]
        Locations (indices) of division boundaries.

    """
    num = map_partitions(ak.num, array, axis=0, meta=empty_typetracer())

    # if only 1 partition things are simple
    if array.npartitions == 1:
        return (0, num.compute())

    # if more than 1 partition cumulative sum required
    cs = list(np.cumsum(num.compute()))
    return tuple([0, *cs])


def _type(array: Array) -> Type | None:
    """Get the type object associated with an array.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection.

    Returns
    -------
    Type
        The awkward type object of the array; if the array does not
        contain metadata ``None`` is returned.

    """
    if array._meta is not None:
        return array._meta.layout.form.type
    return None


def is_awkward_collection(obj: Any) -> bool:
    """Check if an object is a Dask Awkward collection.

    Parameters
    ----------
    obj : Any
        The object of interest.

    Returns
    -------
    bool
        True if `obj` is an Awkward Dask collection.

    """
    return isinstance(obj, (Array, Record, Scalar))


def is_typetracer(obj: Any) -> bool:
    """Check if an object is an Awkward typetracer.

    Typetracers can be one of these categories:
    - Array
    - Record
    - TypeTracerArray
    - MaybeNone
    - OneOf

    Parameters
    ----------
    obj : Any
        The object to test.

    Returns
    -------
    bool
        True if the `obj` is a typetracer like object.

    """
    # array/record typetracer
    if isinstance(obj, (ak.Array, ak.Record)):
        backend = obj.layout.backend

        if not backend.nplike.known_data:
            return True
    # scalar-like typetracer
    elif is_unknown_scalar(obj) or isinstance(obj, (MaybeNone, OneOf)):
        return True
    return False


def meta_or_identity(obj: Any) -> Any:
    """Retrieve the meta of an object or simply pass through.

    Parameters
    ----------
    obj : Any
        The object of interest.

    Returns
    -------
    Any
        If `obj` is an Awkward Dask collection it is `obj._meta`; if
        not we simply return `obj`.

    Examples
    --------
    >>> import awkward as ak
    >>> import dask_awkward as dak
    >>> from dask_awkward.core import meta_or_identity
    >>> x = ak.from_iter([[1, 2, 3], [4]])
    >>> x = dak.from_awkward(x, npartitions=2)
    >>> x
    dask.awkward<from-awkward, npartitions=2>
    >>> meta_or_identity(x)
    <Array-typetracer type='?? * var * int64'>
    >>> meta_or_identity(5)
    5
    >>> meta_or_identity("foo")
    'foo'

    """
    if is_awkward_collection(obj):
        return obj._meta
    elif is_dask_collection(obj) and is_arraylike(obj):
        return ak.Array(
            ak.from_numpy(obj._meta).layout.to_typetracer(forget_length=True)
        )
    return obj


@overload
def to_meta(objects: Sequence[Any]) -> tuple[Any, ...]: ...


@overload
def to_meta(objects: dict[str, Any]) -> dict[str, Any]: ...


def to_meta(objects):
    """Convert sequence or dict of Dask Awkward collections to their metas.

    Parameters
    ----------
    objects : Sequence[Any] or dict[str, Any]
        Sequence or dictionary of objects to retrieve metas from.

    Returns
    -------
    tuple[Any, ...] or dict[str, Any]
        The sequence of objects (or dictionary) where collections have
        been replaced with their metadata.

    """
    if isinstance(objects, dict):
        return {k: meta_or_identity(v) for k, v in objects.items()}
    return tuple(map(meta_or_identity, objects))


def length_zero_array_or_identity(obj: Any) -> Any:
    if is_awkward_collection(obj):
        return ak.typetracer.length_zero_if_typetracer(obj._meta, behavior=obj.behavior)
    return obj


def to_length_zero_arrays(objects: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(map(length_zero_array_or_identity, objects))


def map_meta(
    fn: Callable | ArgsKwargsPackedFunction, *deps: Any, **kwargs: Any
) -> ak.Array | None:
    # NOTE: fn to be a *packed* function (so flat deps or ArgsKwargsPackedFunction)
    #       if ArgsKwargsPackedFunction we do not allow kwargs
    #       as defined up in map_partitions. be careful!
    if isinstance(fn, ArgsKwargsPackedFunction) and len(kwargs) > 0:
        raise ValueError("ArgsKwargsPackedFunctions may not have additional kwargs!")
    try:
        if isinstance(fn, ArgsKwargsPackedFunction):
            meta = fn(*to_meta(deps))
        else:
            meta = fn(*to_meta(deps), **kwargs)
        return meta
    except Exception as err:
        # if compute-unknown-meta is False then we don't care about
        # this failure and we return None.
        if not dask.config.get("awkward.compute-unknown-meta"):
            return None

        # if the metadata function call failed and raise-failed-meta
        # is True, then we want to raise the exception here.
        if dask.config.get("awkward.raise-failed-meta"):
            log.debug(
                f"metadata determination failed: {err}\n"
                f"The config option `awkward.raise-failed-meta` to "
                f"allow this failure was recently deprecated, and can be "
                f"set to False to preserve this behavior before it is removed."
            )
            raise

        # if the metadata function failed and we want to move on to
        # trying the length zero array calculation then we log a
        # warning and pass to the next try-except block.
        else:
            extras = f"function call: {fn}\n" f"metadata: {deps}\n"
            log.warning(
                f"metadata could not be determined from operating upon the "
                f"input array metadata. Falling back to a legacy workaround — "
                f"please report this at https://github.com/dask-contrib/dask-awkward/issues. \n"
                f"{extras}"
            )
        pass
    try:
        arg_lzas = to_length_zero_arrays(deps)
        meta = ak.typetracer.typetracer_from_form(fn(*arg_lzas).layout.form)
        return meta
    except Exception:
        # if compute-unknown-meta is True and we've gotten to this
        # point, we want to throw a warning because a compute is going
        # to happen as a consequence of us not being able to determine
        # metadata.
        if dask.config.get("awkward.compute-unknown-meta"):
            extras = f"function call: {fn}\n" f"metadata: {deps}\n"
            warnings.warn(
                "metadata could not be determined; "
                "a compute on the first partition will occur.\n"
                f"{extras}",
                UserWarning,
            )
    return None


def typetracer_array(a: ak.Array | Array) -> ak.Array:
    """Retrieve the typetracer Array from a concrete or lazy instance.

    Parameters
    ----------
    a : ak.Array | Array
        Array of interest.

    Returns
    -------
    ak.Array
        Typetracer array associated with `a`.

    """
    if isinstance(a, Array):
        return a._meta
    elif isinstance(a, ak.Array):
        return ak.Array(
            a.layout.to_typetracer(forget_length=True),
            behavior=a._behavior,
            attrs=a._attrs,
        )
    else:
        msg = (
            "`a` should be an awkward array or a Dask awkward collection.\n"
            f"Got type {type(a)}"
        )
        raise TypeError(msg)


def empty_typetracer() -> ak.Array:
    """Instantiate a typetracer array with unknown length.

    Returns
    -------
    ak.Array
        Length-less typetracer array (content-less array).

    """
    a = ak.Array([])
    return ak.Array(a.layout.to_typetracer(forget_length=True))


class _BehaviorMethodFn:
    def __init__(self, attr: str, **kwargs: Any) -> None:
        self.attr = attr
        self.kwargs = kwargs

    def __call__(self, coll: ak.Array, *args: Any) -> ak.Array:
        return getattr(coll, self.attr)(*args, **self.kwargs)


class _BehaviorPropertyFn:
    def __init__(self, attr: str) -> None:
        self.attr = attr

    def __call__(self, coll: ak.Array) -> ak.Array:
        return getattr(coll, self.attr)


def normalize_single_outer_inner_index(
    divisions: tuple[int, ...], index: int
) -> tuple[int, int]:
    """Determine partition index and inner index for some divisions.

    Parameters
    ----------
    divisions : tuple[int, ...]
        The divisions of a Dask awkward collection.
    index : int
        The overall index (for the complete collection).

    Returns
    -------
    partition_index : int
        Which partition in the collection.
    new_index : int
        Which inner index in the determined partition.

    Examples
    --------
    >>> from dask_awkward.utils import normalize_single_outer_inner_index
    >>> divisions = (0, 3, 6, 9)
    >>> normalize_single_outer_inner_index(divisions, 0)
    (0, 0)
    >>> normalize_single_outer_inner_index(divisions, 5)
    (1, 2)
    >>> normalize_single_outer_inner_index(divisions, 8)
    (2, 2)

    """
    if index < 0:
        index = divisions[-1] + index
    if len(divisions) == 2:
        return (0, int(index))
    partition_index = int(np.digitize(index, divisions)) - 1
    new_index = index - divisions[partition_index]
    return (int(partition_index), int(new_index))


def make_unknown_length(array: ak.Array) -> ak.Array:
    """Make any highlevel Array a highlevel typetracer Array with unknown length.

    Parameters
    ----------
    array : ak.Array
        Array of interest

    Returns
    -------
    ak.Array
        Highlevel typetracer Array with unknown length.

    """
    return ak.Array(ak.to_layout(array).to_typetracer(forget_length=True))


class PartitionCompatibility(IntEnum):
    """Sum type for describing partition compatibility.

    Use the :func:`partition_compatibility` function as an entry point
    to instances of this class.

    Attributes
    ----------
    NO
        The compatibility is absolutely false; either an unequal
        number of partitions or known divisions do not match
    MAYBE
        The compatibility is possible; the total number of partitions
        are equal but some divisions are unknown so therefore it's
        possible that partitions are not compatible, but this cannot
        be determined without some compute.
    YES
        The compatibility is absolutely true; equal number of
        partitions and known divisions match.

    See Also
    --------
    dask_awkward.partition_compatibility

    """

    NO = 0
    MAYBE = 1
    YES = 2

    @staticmethod
    def _check(*args: Array) -> PartitionCompatibility:
        # first check to see if all arguments have the same number of
        # partitions; this is _always_ defined.
        for arg in args[1:]:
            if args[0].npartitions != arg.npartitions:
                return PartitionCompatibility.NO

        # now we check if divisions are compatible. Sometimes divisions
        # are unknown and we just have a tuple of Nones; but if divisions
        # are known we want to check if they are compatible.
        refarr: Array | None = None
        for arg in args:
            if arg.known_divisions:
                refarr = arg
                break
        # if we never hit the break just return True because we have no
        # known division Arrays.
        else:
            return PartitionCompatibility.MAYBE

        # at this point we have a reference array to compare divisions
        ngood = 0
        for arg in args:
            if arg.known_divisions:
                if arg.divisions != refarr.divisions:
                    return PartitionCompatibility.NO
                else:
                    ngood += 1

        # the ngood counter tells us if all divisions were present and are equal
        if ngood == len(args):
            return PartitionCompatibility.YES

        # if ngood is less than len(args) then we fall back on maybe compatible
        return PartitionCompatibility.MAYBE


def partition_compatibility(*args: Array) -> PartitionCompatibility:
    """Check if multiple collections have compatible partitions.

    Parameters
    ----------
    *args : Array
        Any number of array collections to check.

    Returns
    -------
    PartitionCompatibility
        Result of the check.

    Examples
    --------

    Starting with an absolutely compatible comparison:

    >>> import dask_awkward as dak
    >>> import awkward as ak
    >>> concrete = ak.Array([[1, 2, 3], [4], [5, 6], [0, 0, 0, 0]])
    >>> lazy = dak.from_awkward(concrete, npartitions=2)
    >>> selection = dak.sum(lazy, axis=1) == 0
    >>> dak.partition_compatibility(lazy, selection)
    <PartitionCompatibility.YES: 0>

    The selection doesn't change the length of the arrays at each
    partition, so the divisions are known to be conserved for those
    operations (the sum on ``axis=1`` along with the equality
    comparison).

    In general we have no way of knowing what the resulting divisions
    will be after a boolean selection, but the total number of
    partitions will be conserved, so we have to report ``MAYBE``:

    >>> selected_lazy = lazy[selection]
    >>> dak.partition_compatibility(lazy, lazy_selection)
    <PartitionCompatibility.MAYBE: 2>

    Due the simple nature of this example we know that after the
    selection the partitions will not be compatible (because it's
    clear only 1 element of the original array will survive the
    selection, so the divisions will change after that compute). Now
    we can eagerly compute what the divisions will be on the
    ``lazy_selection`` collection and get a ``NO`` result:

    >>> lazy_selection.eager_compute_divisions()
    >>> dak.partition_compatibility(lazy, lazy_selection)
    <PartitionCompatibility.NO: 1>

    Remember that :func:`Array.eager_compute_divisions` is going to
    trigger a compute to determine the divisions (to know divisions we
    need to know the length of each partition)

    """
    return PartitionCompatibility._check(*args)


HowStrictT = Union[Literal[1], Literal[2], PartitionCompatibility]


def compatible_partitions(
    *args: Array,
    how_strict: HowStrictT = PartitionCompatibility.MAYBE,
) -> bool:
    """Check if all arguments are compatibly partitioned.

    In operations where the blocks of multiple collections are used
    simultaneously, we need the collections to be equally partitioned.
    If the first argument has known divisions, other collections with
    known divisions will be tested against the first arguments
    divisions.

    Parameters
    ----------
    *args : Array
        Array collections of interest.
    how_strict : PartitionCompatibility or Literal[1] or Literal[2]
        Strictness level for the compatibility. If
        ``PartitionCompatbility.MAYBE`` or the integer 1, the check
        will return ``True`` if the arrays are maybe compatible (that
        is, some unknown divisions exist but the total number of
        partitions are compatible). If ``PartitionCompatibility.YES``
        or the integer 2, the check will return ``True`` if and only
        if the arrays are absolutely compatible (that is, all
        divisions are known and they are equal).

    Returns
    -------
    bool
        ``True`` if the collections have compatible partitions at the
        level of requested strictness.

    See Also
    --------
    dask_awkward.PartitionCompatibility
    dask_awkward.partition_compatibility

    """
    partcomp = partition_compatibility(*args)
    if partcomp == PartitionCompatibility.NO:
        return False
    elif partcomp == PartitionCompatibility.MAYBE:
        return how_strict == 1
    return True
