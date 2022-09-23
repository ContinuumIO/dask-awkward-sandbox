from __future__ import annotations

import operator
from collections.abc import Hashable, Mapping
from typing import Any

import dask.config
from dask.blockwise import Blockwise, fuse_roots, optimize_blockwise
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
from dask.local import get_sync

from dask_awkward.layers import AwkwardIOLayer

NEW_METHOD = 1


def basic_optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **kwargs: Any,
) -> Mapping:
    if not isinstance(keys, (list, set)):
        keys = (keys,)  # pragma: no cover
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())

    if dask.config.get("awkward.column-projection-optimization"):
        # our column optimization
        dsk = optimize_iolayer_columns(dsk)

    # Perform Blockwise optimizations for HLG input
    dsk = optimize_blockwise(dsk, keys=keys)
    # cull unncessary tasks
    dsk = dsk.cull(set(keys))  # type: ignore
    # fuse nearby layers
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore

    return dsk


def _attempt_compute_with_columns(dsk: HighLevelGraph, columns: list[str]) -> None:
    layers = dsk.layers.copy()
    deps = dsk.dependencies
    pio_layer_names = [
        k for k, v in dsk.layers.items() if isinstance(v, AwkwardIOLayer)
    ]
    top_io_layer_name = pio_layer_names[0]
    layers[top_io_layer_name] = layers[top_io_layer_name].project_and_mock(columns)
    # final necessary key is the 0th partition of the last layer in
    # the graph (hence the toposort to find last layer).
    final_key = (dsk._toposort_layers()[-1], 0)
    new_hlg = HighLevelGraph(layers, deps).cull([final_key])
    get_sync(new_hlg, list(new_hlg.keys()))
    return None


def _necessary_columns(dsk: HighLevelGraph) -> list[str]:
    # staring fields should be those belonging to the AwkwardIOLayer's
    # metadata (typetracer) array.
    out_meta = list(dsk.layers.values())[-1]._meta
    columns: list[str] = []
    for k, v in dsk.layers.items():
        if isinstance(v, AwkwardIOLayer):
            columns = v._meta.layout.form.columns()
            break

    # can only select output columns that exist in the input
    # (other names may have come from aliases)
    keep = [c for c in out_meta.layout.form.columns() if c in columns]

    for c in columns:
        if c in keep:
            continue
        holdout = c
        allcolumns = set(columns)
        remaining = list(allcolumns - {holdout})
        try:
            _attempt_compute_with_columns(dsk, columns=remaining)
        except IndexError:
            keep.append(holdout)
    if keep == columns:
        keep = []
    return keep


def _is_getitem(layer: Blockwise) -> bool:
    """Determine if a layer is an ``operator.getitem`` call."""
    if not isinstance(layer, Blockwise):
        return False
    return layer.dsk[layer.output][0] == operator.getitem


def _requested_columns(layer: Blockwise) -> set[str]:
    """Determine the columns requested in an ``operator.getitem`` call."""
    fn_arg = layer.indices[1][0]
    if isinstance(fn_arg, list):
        return set(fn_arg)
    return {fn_arg}


def optimize_iolayer_columns(dsk: HighLevelGraph) -> HighLevelGraph:
    # find layers that are AwkwardIOLayer with a project_columns io_func method.
    # projectable-I/O --> "pio"
    pio_layer_names = [
        n
        for n, v in dsk.layers.items()
        if isinstance(v, AwkwardIOLayer) and hasattr(v.io_func, "project_columns")
    ]

    # if the task graph doesn't contain a column-projectable
    # AwkwardIOLayer then bail on this optimization (just return the
    # existing task graph).
    if not pio_layer_names:
        return dsk

    if NEW_METHOD:
        layers = dsk.layers.copy()
        deps = dsk.dependencies.copy()

        for pio_layer_name in pio_layer_names:
            seed_columns = set()
            these_deps = dsk.dependents[pio_layer_name]
            getitem_dependents = [k for k in these_deps if _is_getitem(dsk.layers[k])]
            for gid in getitem_dependents:
                seed_columns |= _requested_columns(dsk.layers[gid])

            if seed_columns:
                new_layer = layers[pio_layer_name].project_columns(list(seed_columns))
                layers[pio_layer_name] = new_layer

        return HighLevelGraph(layers, deps)

    else:
        # determine the necessary columns to complete the executation of
        # the metadata (typetracer) based task graph.
        necessary_cols = _necessary_columns(dsk)

        # if necessary cols is empty just return the input graph
        if not necessary_cols:
            return dsk

        layers = dsk.layers.copy()
        deps = dsk.dependencies.copy()

        for pio_layer_name in pio_layer_names:
            new_layer = layers[pio_layer_name].project_columns(necessary_cols)
            layers[pio_layer_name] = new_layer
            break

        return HighLevelGraph(layers, deps)
