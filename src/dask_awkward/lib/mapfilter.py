from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import awkward as ak
from dask.highlevelgraph import HighLevelGraph
from dask.typing import DaskCollection

from dask_awkward.lib.core import Array as DakArray
from dask_awkward.lib.core import (
    _map_partitions_prepare,
    _to_packed_fn_args,
    dak_cache,
    empty_typetracer,
    new_array_object,
    partitionwise_layer,
    to_meta,
    typetracer_array,
)
from dask_awkward.utils import DaskAwkwardNotImplemented


def _single_return_map_partitions(
    hlg: HighLevelGraph,
    name: str,
    meta: tp.Any,
    npartitions: int,
) -> tp.Any:
    from dask.utils import (
        is_arraylike,
        is_dataframe_like,
        is_index_like,
        is_series_like,
    )

    # ak.Array (this is dak.map_partitions case)
    if isinstance(meta, ak.Array):
        # convert to typetracer if not already
        # this happens when the user provides a concrete array (e.g. np.array)
        # and then wraps it with ak.Array as a return type
        if not ak.backend(meta) == "typetracer":
            meta = ak.to_backend(meta, "typetracer")
        return new_array_object(
            hlg,
            name=name,
            meta=meta,
            npartitions=npartitions,
        )
    # TODO: array, dataframe, series, index
    elif (
        is_arraylike(meta)
        or is_dataframe_like(meta)
        or is_series_like(meta)
        or is_index_like(meta)
    ):
        msg = (
            f"{meta=} is not (yet) supported as return type. If possible, "
            "you can convert it to ak.Array, or wrap it with a python container."
        )
        raise NotImplementedError(msg)
    # don't know? -> put it in a bag
    else:
        from dask.bag.core import Bag

        return Bag(dsk=hlg, name=name, npartitions=npartitions)


def _multi_return_map_partitions(
    hlg: HighLevelGraph,
    name: str,
    meta: tp.Any,
    npartitions: int,
) -> tp.Any:
    # single-return case, this is equal to `dak.map_partitions`
    # but supports other DaskCollections in addition
    if not isinstance(meta, tuple):
        return _single_return_map_partitions(
            hlg=hlg,
            name=name,
            meta=meta,
            npartitions=npartitions,
        )
    # multi-return case
    else:
        from operator import itemgetter
        from typing import cast

        # create tmp dask collection for HLG creation
        tmp = new_array_object(
            hlg, name=name, meta=empty_typetracer(), npartitions=npartitions
        )

        ret = []
        for i, m_pick in enumerate(meta):
            # add a "select/pick" layer
            # to get the ith element of the output
            ith_name = f"{name}-pick-{i}th"

            if ith_name in dak_cache:
                hlg_pick, m_pick = dak_cache[ith_name]
            else:
                lay_pick = partitionwise_layer(itemgetter(i), ith_name, tmp)
                hlg_pick = HighLevelGraph.from_collections(
                    name=ith_name,
                    layer=lay_pick,
                    dependencies=[cast(DaskCollection, tmp)],
                )
                dak_cache[ith_name] = hlg_pick, m_pick
            ret.append(
                _single_return_map_partitions(
                    hlg=hlg_pick,
                    name=ith_name,
                    meta=m_pick,
                    npartitions=npartitions,
                )
            )
        return tuple(ret)


class UntraceableFunctionError(Exception): ...


def _func_args(fn: tp.Callable, *args: tp.Any, **kwargs: tp.Any) -> tp.Mapping:
    import inspect

    ba = inspect.signature(fn).bind(*args, **kwargs)
    return ba.arguments


def _reports2needs(reports: tp.Mapping) -> dict:
    import ast
    from collections import defaultdict

    needs = defaultdict(list)
    for arg, report in reports.items():
        # this should maybe be differently treated?
        keys = set(report.shape_touched) | set(report.data_touched)
        for key in keys:
            slce = ast.literal_eval(key)
            # only strings are actual slice paths to columns,
            # `None` or `ints` are path values to non-record array types,
            # see: https://github.com/scikit-hep/awkward/pull/3311
            slce = tuple(it for it in slce if isinstance(it, str))
            needs[arg].append(slce)
    return needs


def _replace_arrays_with_typetracers(meta: tp.Any) -> tp.Any:
    def _to_tracer(meta: tp.Any) -> tp.Any:
        if isinstance(meta, ak.Array):
            if not ak.backend(meta) == "typetracer":
                meta = typetracer_array(meta)
        elif isinstance(meta, DakArray):
            meta = to_meta([meta])[0]
        return meta

    if isinstance(meta, tuple):
        meta = tuple(map(_to_tracer, meta))
    else:
        meta = _to_tracer(meta)
    return meta


def prerun(
    fn: tp.Callable, *args: tp.Any, **kwargs: tp.Any
) -> tuple[tp.Any, tp.Mapping]:
    """
    Pre-runs the provided function with typetracer arrays to determine the necessary columns
    that should be touched explicitly and to infer the metadata of the function's output.

    Parameters
    ----------
    fn : Callable
        The function to be pre-run.
    *args : Any
        Positional arguments to be passed to the function.
    **kwargs : Any
        Keyword arguments to be passed to the function.

    Returns
    -------
    tuple[Any, Mapping]
        A tuple containing the output of the function when run with typetracer arrays and
        a mapping of the touched columns (prepared to use with ``mapfilter(needs=...)``) generated during the typetracing step.

    Examples
    --------
    >>> import awkward as ak
    >>> import dask_awkward as dak
    >>>
    >>> ak_array = ak.zip({"foo": [1, 2, 3, 4], "bar": [1, 1, 1, 1]})
    >>> dak_array = dak.from_awkward(ak_array, 2)
    >>>
    >>> def process(array: ak.Array) -> ak.Array:
    >>>   return array.foo + array.bar
    >>>
    >>> meta, needs = dak.prerun(process, array)
    >>> print(meta)
    <Array-typetracer [...] type='## * int64'>
    >>> print(needs)
    {'array': [('bar',), ('foo',)]}
    """
    # unwrap `mapfilter`
    if isinstance(fn, mapfilter):
        fn = fn.fn

    in_arguments = _func_args(fn, *args, **kwargs)

    # replace ak.Arrays with typetracers and store the reports
    reports = {}
    fun_kwargs = {}
    args_metas = {
        arg: _replace_arrays_with_typetracers(val) for arg, val in in_arguments.items()
    }

    # can't typetrace if no ak.Arrays are present
    ak_arrays = tuple(filter(lambda x: isinstance(x, ak.Array), args_metas.values()))
    if not ak_arrays:
        return None, {}

    def _render_buffer_key(
        form: ak.forms.Form,
        form_key: str,
        attribute: str,
    ) -> str:
        return form_key

    # prepare function arguments
    for arg, val in args_metas.items():
        if isinstance(val, ak.Array):
            if not ak.backend(val) == "typetracer":
                val = typetracer_array(val)
            tracer, report = ak.typetracer.typetracer_with_report(
                val.layout.form_with_key_path(root=()),
                highlevel=True,
                behavior=val.behavior,
                attrs=val.attrs,
                buffer_key=_render_buffer_key,
            )
            reports[arg] = report
            fun_kwargs[arg] = tracer
        else:
            fun_kwargs[arg] = val

    # try to run the function once with typetracers
    try:
        out = fn(**fun_kwargs)
    except Exception as err:
        import traceback

        # get line number of where the error occurred in the provided function
        # traceback 0: this function, 1: the provided function, >1: the rest of the stack
        tb = traceback.extract_tb(err.__traceback__)
        line_number = tb[1].lineno

        # add also the reports of the typetracer to the error message,
        # and format them as 'needs' wants it to be
        needs = dict(_reports2needs(reports=reports))

        msg = (
            f"'{fn}' is not traceable, an error occurred at line {line_number}. "
            "'dak.mapfilter' can circumvent this by providing 'needs' and "
            "'meta' arguments to it.\n\n"
            "- 'needs': mapping where the keys point to input argument "
            "dask_awkward arrays and the values to columns that should "
            "be touched explicitly. The typetracing step could determine "
            "the following necessary columns until the exception occurred:\n\n"
            f"{needs=}\n\n"
            "- 'meta': value(s) of what the wrapped function would "
            "return. For arrays, only the shape and type matter."
        )
        raise UntraceableFunctionError(msg) from err
    return out, dict(_reports2needs(reports))


@dataclass
class mapfilter:
    """
    A decorator to map a callable across all partitions of any number of collections.
    The function will be treated as a single node in the Dask graph.

    Parameters
    ----------
    fn : Callable
        The function to apply on all partitions. This will get wrapped to handle kwargs, including Dask collections.
    label : str, optional
        Label for the Dask graph layer; if left as ``None`` (default), the name of the function will be used.
    token : str, optional
        Provide an already defined token. If ``None``, a new token will be generated.
    meta : Any, optional
        Metadata for the result (if known). If unknown, `fn` will be applied to the metadata of the `args`.
        If provided, the tracing step will be skipped and the provided metadata is used as return value(s) of `fn`.
    traverse : bool
        Unpack basic Python containers to find Dask collections.
    needs : dict, optional
        If ``None`` (the default), nothing is touched in addition to the standard typetracer report.
        In certain cases, it is necessary to touch additional objects **explicitly** to get the correct typetracer report.
        For this, provide a dictionary that maps input arguments that are arrays to the columns of that array that should be touched.
        If ``needs`` is used together with ``meta``, **only** the columns provided by the ``needs`` argument will be touched explicitly.

    Examples
    --------

    .. code-block:: ipython

        import awkward as ak
        import dask_awkward as dak

        ak_array = ak.zip({"foo": [1, 2, 3, 4], "bar": [1, 1, 1, 1]})
        dak_array = dak.from_awkward(ak_array, 2)

        @dak.mapfilter
        def process(array: ak.Array) -> ak.Array:
          return array.foo * 2

        out = process(dak_array)
        print(out.compute())
        # <Array [2, 4, 6, 8] type='4 * int64'>
    """

    fn: tp.Callable
    label: str | None = None
    token: str | None = None
    meta: tp.Any | None = None
    traverse: bool = True
    needs: tp.Mapping | None = None

    def __post_init__(self) -> None:
        if self.needs is not None and not isinstance(self.needs, tp.Mapping):
            msg = (  # type: ignore[unreachable]
                "'needs' argument must be a mapping where the keys "
                "point to input argument dask_awkward arrays and the values "
                "to columns that should be touched explicitly, "
                f"got {self.needs!r} instead.\n\n"
                "Exemplary usage:\n"
                "\n@partial(mapfilter, needs={'array': ['col1', 'col2']})"
                "\ndef process(array: ak.Array) -> ak.Array:"
                "\n  return array.col1 + array.col2"
            )
            raise ValueError(msg)

    def wrapped_fn(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        in_arguments = _func_args(self.fn, *args, **kwargs)
        if self.needs is not None:
            tobe_touched = set()
            for arg in self.needs.keys():
                if arg in in_arguments:
                    tobe_touched.add(arg)
                else:
                    msg = f"Argument '{arg}' is not present in the function signature."
                    raise ValueError(msg)
            for arg in tobe_touched:
                array = in_arguments[arg]
                if not isinstance(array, ak.Array):
                    raise ValueError(
                        f"Can only touch columns of an awkward array, got {array}."
                    )
                if ak.backend(array) == "typetracer":
                    for slce in self.needs[arg]:
                        ak.typetracer.touch_data(array[slce])

        if self.meta is not None:
            ak_arrays = [
                arg for arg in in_arguments.values() if isinstance(arg, ak.Array)
            ]
            if all(ak.backend(arr) == "typetracer" for arr in ak_arrays):
                return self.meta
        return self.fn(*args, **kwargs)

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        fn, arg_flat_deps_expanded, kwarg_flat_deps = _to_packed_fn_args(
            self.wrapped_fn, *args, traverse=self.traverse, **kwargs
        )

        arg_flat_deps_expanded = _replace_arrays_with_typetracers(
            arg_flat_deps_expanded
        )
        kwarg_flat_deps = _replace_arrays_with_typetracers(kwarg_flat_deps)
        meta = _replace_arrays_with_typetracers(self.meta)
        in_typetracing_mode = arg_flat_deps_expanded or kwarg_flat_deps or meta

        try:
            hlg, meta, deps, name = _map_partitions_prepare(
                fn,
                *arg_flat_deps_expanded,
                *kwarg_flat_deps,
                label=self.label,
                token=self.token,
                meta=meta,
                output_divisions=None,
            )
        # handle the case where the function is not implemented for Dask arrays in dask-awkward
        except DaskAwkwardNotImplemented as err:
            raise err from None
        # handle the case where the function is not traceable - for whatever reason
        except Exception as err:
            if in_typetracing_mode:
                fn_args = _func_args(self.fn, *args, **kwargs)
                sig_str = ", ".join(f"{k}={v}" for k, v in fn_args.items())
                msg = (
                    f"Failed to trace the function '{self.fn}'. "
                    "You can use 'needs' and 'meta' to circumvent this step. "
                    "For this, it might be helpful to do a pre-run of the function:"
                    f"\n\n\tmeta, needs = dak.prerun({self.fn.__name__}, {sig_str})"
                    f"\n\nThis may help to infer the correct `needs` for `mapfilter`."
                )
                raise UntraceableFunctionError(msg) from err
            else:
                raise err from None

        if len(deps) == 0:
            raise ValueError("Need at least one input that is a dask collection.")
        elif len(deps) == 1:
            npart = deps[0].npartitions
        else:
            npart = deps[0].npartitions
            if not all(dep.npartitions == npart for dep in deps):
                msg = "All inputs must have the same number of partitions, got:"
                for dep in deps:
                    npartitions = dep.npartitions
                    msg += f"\n{dep}: {npartitions=}"
                raise ValueError(msg)

        return _multi_return_map_partitions(
            hlg=hlg,
            name=name,
            meta=meta,
            npartitions=npart,
        )
