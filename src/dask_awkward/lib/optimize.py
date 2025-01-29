from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast, no_type_check

import awkward as ak
import dask.config
from awkward.typetracer import touch_data
from dask.blockwise import Blockwise, fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import (
    AwkwardBlockwiseLayer,
    AwkwardInputLayer,
    _dask_uses_tasks,
)
from dask_awkward.lib.utils import typetracer_nochecks
from dask_awkward.utils import first

if _dask_uses_tasks:
    from dask.blockwise import GraphNode, Task, TaskRef

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport
    from dask.typing import Key

log = logging.getLogger(__name__)

COLUMN_OPT_FAILED_WARNING_MSG = """The necessary columns optimization failed; exception raised:

{exception} with message {message}.

Please see the FAQ section of the docs for more information:
https://dask-awkward.readthedocs.io/en/stable/more/faq.html

"""


def all_optimizations(dsk: Mapping, keys: Sequence[Key], **_: Any) -> Mapping:
    """Run all optimizations that benefit dask-awkward computations.

    This function will run both dask-awkward specific and upstream
    general optimizations from core dask.

    """
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(str(id(dsk)), dsk, dependencies=())

    # Perform dask-awkward specific optimizations.
    with typetracer_nochecks():
        dsk = optimize(dsk, keys=keys)
    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore

    return dsk


def optimize(dsk: HighLevelGraph, keys: Sequence[Key], **_: Any) -> Mapping:
    """Run optimizations specific to dask-awkward.

    This is currently limited to determining the necessary columns for
    input layers.

    """
    if dask.config.get("awkward.optimization.enabled"):
        which = dask.config.get("awkward.optimization.which")
        if "columns" in which:
            dsk = optimize_columns(dsk, keys)
        if "layer-chains" in which:
            dsk = rewrite_layer_chains(dsk, keys)

    return dsk


def _prepare_buffer_projection(
    dsk: HighLevelGraph, keys: Sequence[Key]
) -> tuple[dict[str, TypeTracerReport], dict[str, Any]] | None:
    """Prepare for buffer projection by building and evaluating a version of the graph that
    has been annotated for typetracer reporting.

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph to optimize.
    keys : list[str]
        Sequence of keys to optimize with respect to.
    """
    # Skip early if we can't meaningfully optimise any layers
    if not _has_projectable_awkward_io_layer(dsk):
        return None

    layer_to_projection_state: dict[str, Any] = {}
    layer_to_reports: dict[str, TypeTracerReport] = {}
    projection_layers = dict(dsk.layers)

    for name, lay in dsk.layers.items():
        if isinstance(lay, AwkwardInputLayer):
            # The layer supports buffer projection
            if lay.is_projectable:
                # Replace input layer with one that is ready for input projection using a report
                # Store the report for subsequent retrieval, and cache the transient state
                # that column projection later needs to finalise the optimisation
                (
                    projection_layers[name],
                    layer_to_reports[name],
                    layer_to_projection_state[name],
                ) = lay.prepare_for_projection()
            # Layers that don't support buffer projection might support mocking
            # This means that we at least do not have to compute them during evaluation of the optimisation graph
            elif lay.is_mockable:
                projection_layers[name] = lay.mock()
        # Layers that don't support buffer projection might support mocking
        # This means that we at least do not have to compute them during evaluation of the optimisation graph
        elif hasattr(lay, "mock"):
            projection_layers[name] = lay.mock()

    # Ensure that the buffers of each output are entirely touched
    for name in _ak_output_layer_names(dsk):
        projection_layers[name] = _mock_output(projection_layers[name])

    hlg = HighLevelGraph(projection_layers, dsk.dependencies)

    # The caller should apply this optimisation with respect to a number of output keys
    minimal_keys: set[Key] = set()
    for k in keys:
        if isinstance(k, tuple) and len(k) == 2:
            minimal_keys.add((k[0], 0))
        else:
            minimal_keys.add(k)

    # Now we try to compute for each possible output layer key (leaf
    # node on partition 0); this will cause the typetacer reports to
    # get correct fields/columns touched. If the result is a record or
    # an array we of course want to touch all of the data/fields.
    try:
        for layer in hlg.layers.values():
            layer.__dict__.pop("_cached_dict", None)

        results = get_sync(hlg, list(minimal_keys))

        # Touch all the buffers associated with the given output keys
        for out in results:
            if isinstance(out, (ak.Array, ak.Record)):
                touch_data(out)
    except Exception as err:
        on_fail = dask.config.get("awkward.optimization.on-fail")
        # this is the default, throw a warning but skip the optimization.
        if on_fail == "warn":
            warnings.warn(
                COLUMN_OPT_FAILED_WARNING_MSG.format(exception=type(err), message=err)
            )
        # option "pass" means do not throw warning but skip the optimization.
        elif on_fail == "pass":
            log.debug("Column projection optimization failed; optimization skipped.")
        # option "raise" to raise the exception here
        elif on_fail == "raise":
            raise
        else:
            raise ValueError(
                f"Invalid awkward.optimization.on-fail option: {on_fail}.\n"
                "Valid options are 'warn', 'pass', or 'raise'."
            )
        return None
    else:
        return layer_to_reports, layer_to_projection_state


def optimize_columns(dsk: HighLevelGraph, keys: Sequence[Key]) -> HighLevelGraph:
    """Run column projection optimization.

    This optimization determines which columns from an
    ``AwkwardInputLayer`` are necessary for a complete computation.

    For example, if a parquet dataset is loaded with fields:
    ``["foo", "bar", "baz.x", "baz.y"]``

    And the following task graph is made:

    >>> ds = dak.from_parquet("/path/to/dataset")
    >>> z = ds["foo"] - ds["baz"]["y"]

    Upon calling z.compute() the AwkwardInputLayer created in the
    from_parquet call will only read the parquet columns ``foo`` and
    ``baz.y``.

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph to optimize.

    Returns
    -------
    HighLevelGraph
        New, optimized task graph with column-projected ``AwkwardInputLayer``.

    """
    # 1. Build-and-evaluate typetracer-annotated graph
    projection_data = _prepare_buffer_projection(dsk, keys)
    if projection_data is None:
        return dsk

    # 2. Unpack result
    layer_to_reports, layer_to_projection_state = projection_data

    # 3. Project layers using projection state from (1)
    layers = dict(dsk.layers)
    for name, state in layer_to_projection_state.items():
        layers[name] = cast(AwkwardInputLayer, layers[name]).project(
            report=layer_to_reports[name], state=state
        )

    return HighLevelGraph(layers, dsk.dependencies)


def _layers_with_annotation(dsk: HighLevelGraph, key: str) -> list[str]:
    return [n for n, v in dsk.layers.items() if (v.annotations or {}).get(key)]


def _ak_output_layer_names(dsk: HighLevelGraph) -> list[str]:
    """Get a list output layer names.

    Output layer names are annotated with 'ak_output'.

    Parameters
    ----------
    dsk : HighLevelGraph
        Graph of interest.

    Returns
    -------
    list[str]
        Names of the output layers.

    """
    return _layers_with_annotation(dsk, "ak_output")


def _has_projectable_awkward_io_layer(dsk: HighLevelGraph) -> bool:
    """Check if a graph at least one AwkwardInputLayer that is project-able."""
    return any(
        isinstance(v, AwkwardInputLayer) and v.is_projectable
        for v in dsk.layers.values()
    )


def _touch_all_data(*args, **kwargs):
    """Mock writing an ak.Array to disk by touching data buffers."""
    for arg in args + tuple(kwargs.values()):
        touch_data(arg)


def _mock_output(layer):
    """Update a layer to run the _touch_all_data."""
    if _dask_uses_tasks:
        new_layer = copy.deepcopy(layer)
        task = new_layer.task.copy()
        # replace the original function with _touch_all_data
        # and keep the rest of the task the same
        task.func = _touch_all_data
        new_layer.task = task
        return new_layer
    else:
        assert len(layer.dsk) == 1

        new_layer = copy.deepcopy(layer)
        mp = new_layer.dsk.copy()
        for k in iter(mp.keys()):
            mp[k] = (_touch_all_data,) + mp[k][1:]
        new_layer.dsk = mp
        return new_layer


@no_type_check
def rewrite_layer_chains(dsk: HighLevelGraph, keys: Sequence[Key]) -> HighLevelGraph:
    """Smush chains of blockwise layers into a single layer.

    The logic here identifies chains by popping layers (in arbitrary
    order) from a set of all layers in the task graph and walking
    through the dependencies (parent layers) and dependents (child
    layers). If a multi layer chain is discovered we compress it into
    a single layer with the second loop below (for chain in chains;
    that step rewrites the graph). In the chain building logic, if a
    layer exists in the `keys` argument (the keys necessary for the
    compute that we are optimizing for), we shortcircuit the logic to
    ensure we do not chain layers that contain a necessary key inside
    (these layers are called `required_layers` below).

    Parameters
    ----------
    dsk : HighLevelGraph
        Task graph to optimize.
    keys : Any
        Keys that are requested by the compute that is being
        optimized.

    Returns
    -------
    HighLevelGraph
        New, optimized task graph.

    """
    # dask.optimization.fuse_liner for blockwise layers
    import copy

    chains = []
    deps = copy.copy(dsk.dependencies)

    required_layers = {k[0] for k in keys if isinstance(k, tuple)}
    layers = {}
    # find chains; each chain list is at least two keys long
    dependents = dsk.dependents
    all_layers = set(dsk.layers)
    while all_layers:
        layer_key = all_layers.pop()
        layer = dsk.layers[layer_key]
        if not isinstance(layer, AwkwardBlockwiseLayer):
            # shortcut to avoid making comparisons
            layers[layer_key] = layer  # passthrough unchanged
            continue
        children = dependents[layer_key]
        chain = [layer_key]
        current_layer_key = layer_key
        while (
            len(children) == 1
            and dsk.dependencies[first(children)] == {current_layer_key}
            and isinstance(dsk.layers[first(children)], AwkwardBlockwiseLayer)
            and len(dsk.layers[current_layer_key]) == len(dsk.layers[first(children)])
            and current_layer_key not in required_layers
        ):
            # walk forwards
            current_layer_key = first(children)
            chain.append(current_layer_key)
            all_layers.remove(current_layer_key)
            children = dependents[current_layer_key]

        parents = dsk.dependencies[layer_key]
        while (
            len(parents) == 1
            and dependents[first(parents)] == {layer_key}
            and isinstance(dsk.layers[first(parents)], AwkwardBlockwiseLayer)
            and len(dsk.layers[layer_key]) == len(dsk.layers[first(parents)])
            and next(iter(parents)) not in required_layers
        ):
            # walk backwards
            layer_key = first(parents)
            chain.insert(0, layer_key)
            all_layers.remove(layer_key)
            parents = dsk.dependencies[layer_key]
        if len(chain) > 1:
            chains.append(chain)
            layers[chain[-1]] = copy.copy(
                dsk.layers[chain[-1]]
            )  # shallow copy to be mutated
        else:
            layers[layer_key] = layer  # passthrough unchanged

    # do rewrite
    for chain in chains:
        # inputs are the inputs of chain[0]
        # outputs are the outputs of chain[-1]
        # .dsk is composed from the .dsk of each layer
        outkey = chain[-1]
        layer0 = cast(Blockwise, dsk.layers[chain[0]])
        outlayer = layers[outkey]
        numblocks = [nb[0] for nb in layer0.numblocks.values() if nb[0] is not None][0]
        deps[outkey] = deps[chain[0]]
        [deps.pop(ch) for ch in chain[:-1]]

        if _dask_uses_tasks:
            all_tasks = [layer0.task]
        else:
            subgraph = layer0.dsk.copy()
        indices = list(layer0.indices)
        parent = chain[0]

        outlayer.io_deps = layer0.io_deps  # mypy: ignore
        for chain_member in chain[1:]:
            layer = dsk.layers[chain_member]
            for k in layer.io_deps:  # mypy: ignore
                outlayer.io_deps[k] = layer.io_deps[k]

            if _dask_uses_tasks:
                func = layer.task.func
                args = [
                    arg.key if isinstance(arg, GraphNode) else arg
                    for arg in layer.task.args
                ]
                kwargs = {
                    k: v.key if isinstance(v, GraphNode) else v
                    for k, v in layer.task.kwargs.items()
                }
                # how to do this with `.substitute(...)`?
                args2 = _recursive_replace(args, layer, parent, indices)
                kwargs2 = {
                    k: v
                    for k, v in zip(
                        kwargs.keys(),
                        _recursive_replace(kwargs.values(), layer, parent, indices),
                    )
                }
                all_tasks.append(Task(chain_member, func, *args2, **kwargs2))
            else:
                func, *args = layer.dsk[chain_member]  # mypy: ignore
                args2 = _recursive_replace(args, layer, parent, indices)
                subgraph[chain_member] = (func,) + tuple(args2)
            parent = chain_member
        outlayer.numblocks = {
            i[0]: (numblocks,) for i in indices if i[1] is not None
        }  # mypy: ignore
        if _dask_uses_tasks:
            outlayer.task = Task.fuse(*all_tasks)
        else:
            outlayer.dsk = subgraph  # mypy: ignore
        if hasattr(outlayer, "_dims"):
            del outlayer._dims
        outlayer.indices = tuple(  # mypy: ignore
            (i[0], (".0",) if i[1] is not None else None) for i in indices
        )
        outlayer.output_indices = (".0",)  # mypy: ignore
        outlayer.inputs = getattr(layer0, "inputs", set())  # mypy: ignore
        if hasattr(outlayer, "_cached_dict"):
            del outlayer._cached_dict  # reset, since original can be mutated
    return HighLevelGraph(layers, deps)


def _recursive_replace(args, layer, parent, indices):
    args2 = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("__dask_blockwise__"):
            ind = int(arg[18:])
            if layer.indices[ind][1] is None:
                # this is a simple arg
                args2.append(layer.indices[ind][0])
            elif layer.indices[ind][0] == parent:
                # arg refers to output of previous layer
                if _dask_uses_tasks:
                    args2.append(TaskRef(parent))
                else:
                    args2.append(parent)
            else:
                # arg refers to things defined in io_deps
                indices.append(layer.indices[ind])
                arg2 = f"__dask_blockwise__{len(indices) - 1}"
                if _dask_uses_tasks:
                    args2.append(TaskRef(arg2))
                else:
                    args2.append(arg2)
        elif isinstance(arg, list):
            args2.append(_recursive_replace(arg, layer, parent, indices))
        elif isinstance(arg, tuple):
            args2.append(tuple(_recursive_replace(arg, layer, parent, indices)))
        # elif isinstance(arg, dict):
        else:
            args2.append(arg)
    return args2
