from __future__ import annotations

import keyword
import operator
from functools import partial
from math import ceil
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Mapping

import awkward._v2 as ak
import numpy as np
from awkward._v2._connect.numpy import NDArrayOperatorsMixin
from dask.base import DaskMethodsMixin, dont_optimize, is_dask_collection, tokenize
from dask.blockwise import blockwise as upstream_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import IndexCallable, cached_property, funcname, key_split

from .utils import is_empty_slice, normalize_single_outer_inner_index

if TYPE_CHECKING:
    from awkward._v2.contents.content import Content
    from awkward._v2.forms.form import Form
    from awkward._v2.types.type import Type
    from dask.blockwise import Blockwise


def _finalize_array(results: Any) -> Any:
    if any(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)
    elif len(results) == 1 and isinstance(results[0], int):
        return results[0]
    elif all(isinstance(r, int) for r in results):
        return ak.Array(results)
    else:
        msg = (
            "Unexpected results of a computation.\n "
            f"results: {results}"
            f"type of first result: {type(results[0])}"
        )
        raise RuntimeError(msg)


def _finalize_scalar(results: Any) -> Any:
    return results[0]


class Scalar(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, name: str, meta: Any | None = None) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._meta: Any | None = meta

    def __dask_graph__(self) -> HighLevelGraph:
        return self._dask

    def __dask_keys__(self) -> list[str]:
        return [self._name]

    def __dask_layers__(self) -> tuple[str, ...]:
        if isinstance(self._dask, HighLevelGraph) and len(self._dask.layers) == 1:
            return tuple(self._dask.layers)
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    @staticmethod
    def __dask_optimize__(dsk: Any, keys: Any, **kwargs: Any) -> HighLevelGraph:
        return dont_optimize(dsk, keys, **kwargs)

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self) -> Any:
        return _finalize_scalar, ()

    def __dask_postpersist__(self) -> Any:
        return self._rebuild, ()

    def _rebuild(
        self,
        dsk: HighLevelGraph,
        *,
        rename: Mapping[str, str] | None = None,
    ) -> Any:
        name = self._name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.meta)

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> Any | None:
        return self._meta

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"dask.awkward<{key_split(self.name)}, type=Scalar>"


def new_scalar_object(dsk: HighLevelGraph, name: str, meta: Any) -> Scalar:
    return Scalar(dsk, name, meta)


class Record(Scalar):
    def __init__(self, dsk: HighLevelGraph, name: str, meta: Any | None = None) -> None:
        super().__init__(dsk, name, meta)

    def __getitem__(self, key: str) -> Any:
        token = tokenize(self, key)
        name = f"getitem-{token}"
        new_meta = self._meta[key]  # type: ignore

        # first check for array type return
        if isinstance(new_meta, ak.Array):
            graphlayer = {(name, 0): (operator.getitem, self.name, key)}
            hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
            return new_array_object(hlg, name, new_meta, npartitions=1)

        # then check for scalar (or record) type
        graphlayer = {name: (operator.getitem, self.name, key)}  # type: ignore
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, name, new_meta)
        else:
            return new_scalar_object(hlg, name, new_meta)

    def __getattr__(self, attr: str) -> Any:
        try:
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def __str__(self) -> str:
        return f"dask.awkward<{key_split(self.name)}, type=Record>"

    @property
    def fields(self) -> list[str] | None:
        if self.meta is not None:
            return self.meta.fields
        return None

    def _ipython_key_completions_(self) -> list[str]:
        if self.meta is not None:
            return self.meta._ipython_key_completions_()
        return []

    def __dir__(self) -> list[str]:
        from awkward._v2.highlevel import _dir_pattern

        fields = [] if self.meta is None else self.meta._layout.fields
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


def new_record_object(dsk: HighLevelGraph, name: str, meta: Any) -> Record:
    return Record(dsk, name, meta)


class Array(DaskMethodsMixin, NDArrayOperatorsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`dask_awkward.from_parquet`,
    :py:func:`dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        meta: ak.Array | None,
        divisions: tuple[int | None, ...],
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._divisions = divisions
        self.meta = meta

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> list[tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return _finalize_array, ()

    def __dask_postpersist__(self) -> Any:
        return self._rebuild, ()

    @staticmethod
    def __dask_optimize__(dsk: Any, keys: Any, **kwargs: Any) -> HighLevelGraph:
        return dont_optimize(dsk, keys, **kwargs)

    __dask_scheduler__ = staticmethod(threaded_get)

    def _rebuild(
        self,
        dsk: HighLevelGraph,
        *,
        rename: Mapping[str, str] | None = None,
    ) -> Array:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.meta, divisions=self.divisions)

    def __len__(self) -> int:
        self._compute_divisions()
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

    def __repr__(self) -> str:
        return self.__str__()

    # def _ipython_display_(self) -> None:
    #     if self.meta is None:
    #         return None
    #
    #     import json
    #
    #     from IPython.display import display_json
    #
    #     display_json(json.loads(self.meta.form.to_json()), raw=True)

    def _ipython_key_completions_(self) -> list[str]:
        if self.meta is not None:
            return self.meta._ipython_key_completions_()
        return []

    def __dir__(self) -> list[str]:
        from awkward._v2.highlevel import _dir_pattern

        fields = [] if self.meta is None else self.meta._layout.fields
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

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def name(self) -> str:
        return self._name

    @property
    def ndim(self) -> int | None:
        return ndim(self)

    @property
    def divisions(self) -> tuple[int | None, ...]:
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        return len(self.divisions) > 0 and None not in self.divisions

    @property
    def npartitions(self) -> int:
        return len(self.divisions) - 1

    @property
    def meta(self) -> ak.Array | None:
        return self._meta

    @meta.setter
    def meta(self, m: ak.Array | None) -> None:
        if m is not None and not isinstance(m, ak.Array):
            raise TypeError("meta must be an instance of an Awkward Array.")
        self._meta = m

    @property
    def layout(self) -> Content:
        if self.meta is not None:
            return self.meta.layout
        raise ValueError("This collections meta is None; unknown layout.")

    @property
    def typetracer(self) -> ak.Array | None:
        return self.meta

    @property
    def fields(self) -> list[str] | None:
        if self.meta is not None:
            return self.meta.fields
        return None

    @property
    def form(self) -> Form | None:
        if self.meta is not None:
            return self.meta.layout.form
        return None

    @cached_property
    def keys_array(self) -> np.ndarray:
        return np.array(self.__dask_keys__(), dtype=object)

    def _partitions(self, index: Any) -> Array:
        if not isinstance(index, tuple):
            index = (index,)
        token = tokenize(self, index)
        from dask.array.slicing import normalize_index

        raw = normalize_index(index, (self.npartitions,))
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in raw)  # type: ignore
        name = f"partitions-{token}"
        new_keys = self.keys_array[index].tolist()
        dsk = {(name, i): tuple(key) for i, key in enumerate(new_keys)}
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])

        # if a single partition was requested we trivially know the new divisions.
        if len(raw) == 1 and isinstance(raw[0], int) and self.known_divisions:
            new_divisions = (
                0,
                self.divisions[raw[0] + 1] - self.divisions[raw[0]],  # type: ignore
            )
        # otherwise nullify the known divisions
        else:
            new_divisions = (None,) * (len(new_keys) + 1)  # type: ignore

        return new_array_object(graph, name, None, divisions=tuple(new_divisions))

    @property
    def partitions(self) -> IndexCallable:
        """Get a specific partition or slice of partitions.

        Returns
        -------
        dask.utils.IndexCallable

        Examples
        --------
        >>> import dask_awkward as dak
        >>> import awkward._v2 as ak
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

    def _getitem_trivial_inner(self, key: Any) -> Array:
        token = tokenize(self, key)
        name = f"getitem-{token}"
        graphlayer = partitionwise_layer(
            lambda x, gikey: operator.getitem(x, gikey), name, self, gikey=key
        )
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        meta = self.meta[key] if self.meta is not None else None
        return new_array_object(hlg, name, meta=meta, divisions=self.divisions)

    def _getitem_single_int(self, key: int) -> Any:
        # determine which partition to grab from (pidx) and which
        # index _inside_ of (relative to) that partition to then call
        # getitem with (rewriting key).
        self._divisions = calculate_known_divisions(self)
        pidx, key = normalize_single_outer_inner_index(self.divisions, key)  # type: ignore
        partition = self.partitions[pidx]
        new_meta = partition.meta[key]

        # if we know a new array is going to be made, just call the
        # trivial inner on the new partition.
        if isinstance(new_meta, ak.Array):
            result = partition._getitem_trivial_inner(key)
            result._divisions = (0, None)
            return result

        # otherwise make sure we have one of the other potential results.
        from awkward._v2._typetracer import MaybeNone, OneOf, UnknownScalar

        if not (
            isinstance(new_meta, ak.Record)
            or isinstance(new_meta, UnknownScalar)
            or isinstance(new_meta, OneOf)
            or isinstance(new_meta, MaybeNone)
        ):
            raise NotImplementedError("Key not supported for this array.")

        token = tokenize(partition, key)
        name = f"getitem-{token}"
        dsk = {
            name: (
                lambda x, gikey: operator.getitem(x, gikey),
                partition.__dask_keys__()[0],
                key,
            )
        }
        hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[partition])
        if isinstance(new_meta, ak.Record):
            return new_record_object(hlg, name, new_meta)
        else:
            return new_scalar_object(hlg, name, new_meta)

    def _getitem_boolean_lazy_array(self, key: Any) -> Any:
        if key.known_divisions and self.known_divisions:
            if key.divisions != self.divisions:
                raise ValueError(
                    "The boolean array (they key in this getitem call) "
                    "must be partitioned in the same way as this array."
                )
        else:
            if key.npartitions != self.npartitions:
                raise ValueError(
                    "The boolean array (they key in this getitem call) "
                    "must be partitioned in the same way as this array."
                )

        new_meta = None
        if key.meta is not None:
            new_meta = operator.getitem(self.meta, key.meta)

        s_keys = self.__dask_keys__()
        k_keys = key.__dask_keys__()
        name = f"getitem-{tokenize(self, key)}"
        dsk = {
            (name, i): (operator.getitem, sk, kk)
            for i, (sk, kk) in enumerate(zip(s_keys, k_keys))
        }
        hlg = HighLevelGraph.from_collections(name, dsk, dependencies=(self, key))
        return new_array_object(hlg, name, divisions=self.divisions, meta=new_meta)

    def __getitem__(self, key: Any) -> Any:
        """Select items from the collection.

        Heavily under construction.

        Arguments
        ---------
        key : many types supported
            Selection criteria.

        Returns
        -------
        Array | Record | Scalar
            Resulting collection.

        """

        # don't accept lists containing integers.
        if isinstance(key, list):
            if any(isinstance(k, int) for k in key):
                raise NotImplementedError("Lists containing integers not supported.")

        # a single string or a list.
        if isinstance(key, (str, list)):
            return self._getitem_trivial_inner(key=key)

        # an empty slice
        elif is_empty_slice(key):
            return self._getitem_trivial_inner(key=key)

        # a single ellipsis
        elif key is Ellipsis:
            return self._getitem_trivial_inner(key=key)

        # a single integer
        elif isinstance(key, int):
            return self._getitem_single_int(key=key)

        elif isinstance(key, Array) and issubclass(
            key.layout.content.dtype.type, (np.bool_, bool)
        ):
            return self._getitem_boolean_lazy_array(key=key)

        # unimplemented
        elif isinstance(key, slice):
            pass
        elif isinstance(key, tuple):
            pass

        raise NotImplementedError(f"__getitem__ doesn't support key {key}")

    def __getattr__(self, attr: str) -> Any:
        try:
            return self.__getitem__(attr)
        except (IndexError, KeyError):
            raise AttributeError(f"{attr} not in fields.")

    def map_partitions(
        self,
        func: Callable,
        *args: Any,
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
        return map_partitions(func, self, *args, **kwargs)

    def _compute_divisions(self) -> None:
        self._divisions = calculate_known_divisions(self)

    def clear_divisions(self) -> None:
        """Clear the divisions of a Dask Awkward Collection."""
        self._divisions = (None,) * (self.npartitions + 1)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplementedError("Array ufunc supports only method == '__call__'")

        new_meta = None

        # divisions need to be compat. (identical for now?)

        inputs_meta = []
        for inp in inputs:
            # if input is a Dask Awkward Array collection, grab it's meta
            if isinstance(inp, Array):
                inputs_meta.append(inp.meta)
            # if input is a concrete Awkward Array, grab it's typetracer
            elif isinstance(inp, ak.Array):
                inputs_meta.append(ak.Array(inp.layout.typetracer))
            # otherwise pass along
            else:
                inputs_meta.append(inp)

        # compute new meta from inputs
        new_meta = ufunc(*inputs_meta)

        return map_partitions(ufunc, *inputs, meta=new_meta, **kwargs)


def _first_partition(array: Array) -> ak.Array:
    """Compute the first partition of a Array collection.

    Parameters
    ----------
    array : dask_awkward.Array
        Awkward collection.

    Returns
    -------
    ak.Array
        Concrete awkward array for the first partition of the Dask
        awkward array.

    """
    computed = array.__dask_scheduler__(
        array.__dask_graph__(), array.__dask_keys__()[0]
    )
    return computed


def _get_typetracer(array: Array) -> ak.Array:
    """Obtain the awkward type tracer associated with an array.

    This will compute the first partition of the array collection if
    necessary.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection.

    Returns
    -------
    Content
        Awkward Content object representing the typetracer (metadata).

    """
    if array.meta is not None:
        return array.meta
    first_part = _first_partition(array)
    if not isinstance(first_part, ak.Array):
        raise TypeError(f"Should have an ak.Array type, got {type(first_part)}")
    return ak.Array(first_part.layout.typetracer)


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    meta: ak.Array | None = None,
    npartitions: int | None = None,
    divisions: tuple[Any, ...] | None = None,
) -> Array:
    """Instantiate a new Array collection object.

    Parameters
    ----------
    dsk : dask.highlevelgraph.HighLevelGraph
        Graph backing the collection.
    name : str
        Unique name for the collection.
    meta : Content, the awkward-array type tracing information, optional
        Object metadata; this is an awkward-array type tracer.
    npartitions : int, optional
        Total number of partitions; if used `divisions` will be a
        tuple of length `npartitions` + 1 with all elements``None``.
    divisions : tuple[int or None, ...], optional
        Tuple identifying the locations of the divisions between the
        partitions.

    Returns
    -------
    Array
        Resulting collection.

    """
    if divisions is None and npartitions is not None:
        divisions = (None,) * (npartitions + 1)
    elif divisions is not None and npartitions is not None:
        raise ValueError("Only one of either divisions or npartitions must be defined.")
    elif divisions is None and npartitions is None:
        raise ValueError("One of either divisions or npartitions must be defined.")

    array = Array(dsk, name, meta, divisions)  # type: ignore

    if meta is None:
        try:
            array.meta = _get_typetracer(array)
        except (AttributeError, AssertionError, TypeError):
            array.meta = None

    return array


def partitionwise_layer(
    func: Callable,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> Blockwise:
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
    numblocks: dict[Any, int | tuple[int, ...]] = {}
    for arg in args:
        if isinstance(arg, Array):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (arg.npartitions,)
        elif is_dask_collection(arg):
            raise NotImplementedError(
                "Use of Array with other Dask collections is currently unsupported."
            )
        else:
            pairs.extend([arg, None])
    return upstream_blockwise(
        func,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )


def map_partitions(
    func: Callable,
    *args: Any,
    name: str | None = None,
    meta: ak.Array | None = None,
    **kwargs: Any,
) -> Array:
    """Map a callable across all partitions of a collection.

    Parameters
    ----------
    func : Callable
        Function to apply on all partitions.
    *args : Collections and function arguments
        Arguments passed to the function, if arguments are
        Array collections they must be compatibly
        partitioned.
    name : str, optional
        Name for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    **kwargs : Any
        Additional keyword arguments passed to the `func`.

    Returns
    -------
    dask_awkward.Array
        The new collection.

    """
    token = tokenize(func, *args, **kwargs)
    name = name or funcname(func)
    name = f"{name}-{token}"
    lay = partitionwise_layer(func, name, *args, **kwargs)
    deps = [a for a in args if is_dask_collection(a)] + [
        v for _, v in kwargs.items() if is_dask_collection(v)
    ]
    hlg = HighLevelGraph.from_collections(name, lay, dependencies=deps)
    return new_array_object(hlg, name, meta, divisions=args[0].divisions)


def pw_reduction_with_agg_to_scalar(
    array: Array,
    func: Callable,
    agg: Callable,
    *,
    name: str | None = None,
    **kwargs: Any,
) -> Scalar:
    """Partitionwise operation with aggregation to scalar.

    Parameters
    ----------
    array : dask_awkward.Array
        Awkward array collection.
    func : Callable
        Function to apply on all partitions.
    agg : Callable
        Function to aggregate the result on each partition.
    name : str | None
        Name for the computation, if ``None`` we use the name of
        `func`.
    **kwargs : Any
        Keyword arguments passed to `func`.

    Returns
    -------
    Scalar
        Resulting scalar Dask collection.

    """
    token = tokenize(array)
    name = func.__name__ if name is None else name
    name = f"{name}-{token}"
    func = partial(func, **kwargs)
    dsk = {(name, i): (func, k) for i, k in enumerate(array.__dask_keys__())}
    dsk[name] = (agg, list(dsk.keys()))  # type: ignore
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[array])
    return new_scalar_object(hlg, name, None)


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
    # if divisions are known, quick return
    if array.known_divisions:
        return array.divisions  # type: ignore

    # if more than 1 partition use cumulative sum
    if array.npartitions > 1:
        nums = array.map_partitions(len).compute()
        cs = list(np.cumsum(nums))
        return tuple([0, *cs])

    # if only 1 partition just get it's length
    return (0, array.map_partitions(len).compute())


class TrivialPartitionwiseOp:
    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._func = func
        self.name = func.__name__ if name is None else name
        self._kwargs = kwargs

    def __call__(self, collection: Array, **kwargs: Any) -> Array:
        # overwrite any saved kwargs in self._kwargs
        for k, v in kwargs.items():
            self._kwargs[k] = v
        return map_partitions(self._func, collection, name=self.name, **self._kwargs)


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
    if array.meta is not None:
        return array.meta.layout.form.type
    return None


def ndim(array: Array) -> int | None:
    """Number of dimensions before reaching a numeric type or a record.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection

    Returns
    -------
    int or None
        Number of dimensions as an integer, or ``None`` if the
        collection does not contain metadata.

    """
    if array.meta is not None:
        return array.meta.ndim
    return None


def fields(array: Array) -> list[str] | None:
    """Get the fields of a Array collection.

    Parameters
    ----------
    array : dask_awkward.Array
        The collection.

    Returns
    -------
    list[str] or None
        The fields of the array; if the array does not contain
        metadata ``None`` is returned.

    """
    if array.meta is not None:
        return array.meta.fields
    return None


def from_awkward(source: ak.Array, npartitions: int, name: str | None = None) -> Array:
    if name is None:
        name = f"from-awkward-{tokenize(source, npartitions)}"
    nrows = len(source)
    chunksize = int(ceil(nrows / npartitions))
    locs = list(range(0, nrows, chunksize)) + [nrows]
    llg = {
        (name, i): source[start:stop]
        for i, (start, stop) in enumerate(zip(locs[:-1], locs[1:]))
    }
    hlg = HighLevelGraph.from_collections(name, llg, dependencies=set())
    return new_array_object(
        hlg,
        name,
        divisions=tuple(locs),
        meta=ak.Array(source.layout.typetracer),
    )
