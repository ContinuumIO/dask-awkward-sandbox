from __future__ import annotations

from typing import cast

from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers.layers import AwkwardInputLayer
from dask_awkward.lib.core import Array


def optimize_columns(array: Array, columns: dict[str, frozenset[str]]) -> Array:
    """
    Manually updates the AwkwardInputLayer(s) with the specified columns. This is useful
    for tracing the necessary buffers for a given computation once, and then reusing the
    typetracer reports to touch only the necessary columns for other datasets.

    Calling this function will update the `AwkwardInputLayer`'s `necessary_columns` attribute,
    i.e. pruning the columns that are not wanted. This replaces the automatic column optimization,
    which is why one should be careful when using this function combined with `.compute(optimize_graph=True)`.


    Parameters
    ----------
    array : Array
        The dask-awkward array to be optimized.
    columns : dict[str, frozenset[str]]
        The columns to be touched.

    Returns
    -------
    Array
        A new Dask-Awkward array with only the specified columns.
    """
    if not isinstance(array, Array):
        raise TypeError(
            f"Expected `dak_array` to be of type `dask_awkward.Array`, got {type(array)}"
        )

    dsk = array.dask
    layers = dict(dsk.layers)
    deps = dict(dsk.dependencies)

    for name, cols in columns.items():
        io_layer = cast(AwkwardInputLayer, layers[name])
        if not isinstance(io_layer, AwkwardInputLayer):
            raise TypeError(
                f"Expected layer {name} to be of type `dask_awkward.layers.AwkwardInputLayer`, got {type(io_layer)}"
            )
        projected_layer = io_layer.project_manually(columns=cols)

        # explicitely disable 'project-ability' now, since we did this manually just now
        # Is there a better way to do this? Because this disables the possibility to chain call `dak.manual.optimize_columns`
        projected_layer.is_projectable = False

        layers[name] = projected_layer

    new_dsk = HighLevelGraph(layers, deps)
    return array._rebuild(dsk=new_dsk)
