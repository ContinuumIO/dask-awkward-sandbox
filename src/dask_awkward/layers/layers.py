from __future__ import annotations

import copy
import math
import operator
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, Union, cast

import dask
import toolz
from dask.blockwise import Blockwise, BlockwiseDepDict, blockwise_token
from dask.highlevelgraph import MaterializedLayer
from dask.layers import Layer
from typing_extensions import TypeAlias

from dask_awkward.utils import LazyInputsDict

_dask_uses_tasks = hasattr(dask.blockwise, "Task")

if _dask_uses_tasks:
    from dask.blockwise import Task, TaskRef

if TYPE_CHECKING:
    from awkward import Array as AwkwardArray
    from awkward._nplikes.typetracer import TypeTracerReport


BackendT: TypeAlias = Union[Literal["cpu"], Literal["jax"], Literal["cuda"]]


class AwkwardBlockwiseLayer(Blockwise):
    """Just like upstream Blockwise, except we override pickling"""

    has_been_unpickled: bool = False

    @classmethod
    def from_blockwise(cls, layer: Blockwise) -> AwkwardBlockwiseLayer:
        ob = object.__new__(cls)
        ob.__dict__.update(layer.__dict__)
        return ob

    def mock(self) -> AwkwardBlockwiseLayer:
        """Mock this layer without evaluating it"""
        layer = copy.copy(self)
        nb = layer.numblocks
        layer.numblocks = {k: tuple(1 for _ in v) for k, v in nb.items()}
        layer.__dict__.pop("_dims", None)
        return layer

    def __getstate__(self) -> dict:
        # Indicator that this layer has been serialised
        state = self.__dict__.copy()
        state["has_been_unpickled"] = True
        return state

    def __repr__(self) -> str:
        return "Awkward" + super().__repr__()


class ImplementsIOFunction(Protocol):
    def __call__(self, *args, **kwargs): ...


T = TypeVar("T")


class ImplementsMocking(ImplementsIOFunction, Protocol):
    def mock(self) -> AwkwardArray: ...


class ImplementsMockEmpty(ImplementsIOFunction, Protocol):
    def mock_empty(self, backend: BackendT) -> AwkwardArray: ...


class ImplementsReport(ImplementsIOFunction, Protocol):
    @property
    def return_report(self) -> bool: ...


class ImplementsProjection(ImplementsMocking, Protocol[T]):
    def prepare_for_projection(self) -> tuple[AwkwardArray, TypeTracerReport, T]: ...

    def project(self, report: TypeTracerReport, state: T) -> ImplementsIOFunction: ...

    # `project_manually` is typically an alias to `project_columns`. Some IO functions
    # might implement this method with a different name, because their respective file format
    # uses different conventions. An example is ROOT, where the columns are called "keys".
    # In this case, the method should be aliased to `project_keys`.
    def project_manually(self, columns: frozenset[str]) -> ImplementsIOFunction: ...


class ImplementsNecessaryColumns(ImplementsProjection[T], Protocol):
    def necessary_columns(
        self, report: TypeTracerReport, state: T
    ) -> frozenset[str]: ...


class IOFunctionWithMocking(ImplementsMocking, ImplementsIOFunction):
    def __init__(self, meta: AwkwardArray, io_func: ImplementsIOFunction):
        self._meta = meta
        self._io_func = io_func

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_meta"] = None
        return state

    def __call__(self, *args, **kwargs):
        return self._io_func(*args, **kwargs)

    def mock(self) -> AwkwardArray:
        assert self._meta is not None
        return self._meta


def io_func_implements_projection(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "prepare_for_projection")


def io_func_implements_mocking(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "mock")


def io_func_implements_mock_empty(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "mock_empty")


def io_func_implements_columnar(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "necessary_columns")


def io_func_implements_report(func: ImplementsIOFunction) -> bool:
    return hasattr(func, "return_report")


class AwkwardTokenizable:

    def __init__(self, ret_val, parent_name):
        self.parent_name = parent_name
        self.ret_val = ret_val

    def __dask_tokenize__(self):
        return ("AwkwardTokenizable", self.parent_name)

    def __call__(self, *_, **__):
        return self.ret_val


class AwkwardInputLayer(AwkwardBlockwiseLayer):
    """A layer known to perform IO and produce Awkward arrays

    We specialise this so that we have a way to prune column selection on load
    """

    def __init__(
        self,
        *,
        name: str,
        inputs: Any,
        io_func: ImplementsIOFunction,
        label: str | None = None,
        produces_tasks: bool = False,
        creation_info: dict | None = None,
        annotations: Mapping[str, Any] | None = None,
        is_projectable: bool | None = None,
    ) -> None:
        self.name = name
        self.inputs = inputs
        self.io_func = io_func
        self.label = label
        self.produces_tasks = produces_tasks
        self.annotations = annotations
        self.creation_info = creation_info
        self._is_projectable = is_projectable

        io_arg_map = BlockwiseDepDict(
            mapping=LazyInputsDict(self.inputs),  # type: ignore
            produces_tasks=self.produces_tasks,
        )

        super_kwargs: dict[str, Any] = {
            "output": self.name,
            "output_indices": "i",
            "indices": [(io_arg_map, "i")],
            "numblocks": {},
            "annotations": None,
        }

        if _dask_uses_tasks:
            super_kwargs["task"] = Task(name, self.io_func, TaskRef(blockwise_token(0)))
        else:
            super_kwargs["dsk"] = {name: (self.io_func, blockwise_token(0))}

        super().__init__(**super_kwargs)

    def __repr__(self) -> str:
        return f"AwkwardInputLayer<{self.output}>"

    @property
    def is_projectable(self) -> bool:
        # isinstance(self.io_func, ImplementsProjection)
        if self._is_projectable is None:
            return (
                io_func_implements_projection(self.io_func)
                and not self.has_been_unpickled
            )
        return self._is_projectable

    @is_projectable.setter
    def is_projectable(self, value: bool) -> None:
        self._is_projectable = value

    @property
    def is_mockable(self) -> bool:
        # isinstance(self.io_func, ImplementsMocking)
        return io_func_implements_mocking(self.io_func)

    @property
    def is_columnar(self) -> bool:
        return io_func_implements_columnar(self.io_func)

    def mock(self) -> AwkwardInputLayer:
        assert self.is_mockable
        return AwkwardInputLayer(
            name=self.name,
            inputs=[None][: int(list(self.numblocks.values())[0][0])],
            io_func=lambda *_, **__: cast(ImplementsMocking, self.io_func).mock(),
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )

    def prepare_for_projection(self) -> tuple[AwkwardInputLayer, TypeTracerReport, T]:
        """Mock the input layer as starting with a data-less typetracer.
        This method is used to create new dask task graphs that
        operate purely on typetracer Arrays (that is, array with
        Awkward structure but without real data buffers). This allows
        us to test which parts of a real Awkward array will be used in
        a real computation. We do this by running a graph which starts
        with mocked AwkwardInputLayers.

        We mock an AwkwardInputLayer in these steps:
        1. Ask the IO function to prepare a new meta Array, and return
           any transient state.
        2. Build a new AwkwardInputLayer whose IO function just returns
           this meta (typetracer) Array
        3. Return the new input layer and the transient state

        When this new layer is added to a dask task graph, and that
        graph is computed, the report object will be mutated.
        Inspecting the report object after the compute tells us which
        buffers from the original form would be required for a real
        compute with the same graph.

        Returns
        -------
        AwkwardInputLayer
            Copy of the input layer with data-less input.
        TypeTracerReport
            The report object used to track touched buffers.
        Any
            The black-box state object returned by the IO function.
        """
        assert self.is_projectable
        new_meta_array, report, state = cast(
            ImplementsProjection, self.io_func
        ).prepare_for_projection()

        new_return = new_meta_array
        if io_func_implements_report(self.io_func):
            if cast(ImplementsReport, self.io_func).return_report:
                new_return = (new_meta_array, type(new_meta_array)([]))

        new_input_layer = AwkwardInputLayer(
            name=self.name,
            inputs=[None][: int(list(self.numblocks.values())[0][0])],
            io_func=AwkwardTokenizable(new_return, self.name),
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )
        return new_input_layer, report, state

    def project(
        self,
        report: TypeTracerReport,
        state: T,
    ) -> AwkwardInputLayer:
        assert self.is_projectable
        io_func = cast(ImplementsProjection, self.io_func).project(
            report=report, state=state
        )
        return AwkwardInputLayer(
            name=self.name,
            inputs=self.inputs,
            io_func=io_func,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )

    def necessary_columns(self, report: TypeTracerReport, state: T) -> frozenset[str]:
        """Report the necessary _columns_ implied by a given buffer optimisation state.

        Each IO source usually has the notion of a "column". For uproot, that's a TTree key,
        whilst Parquet has "fields". Awkward operates at the _buffer_ level, which is nearly-always
        a lower-level representation. As such, when users want to answer the question "which IO-columns"
        does this graph require, we need to map between buffer names and the IO-source columns.

        This routine asks the IO function to perform that remapping, without knowing anything about what it does.
        """
        assert self.is_columnar
        return cast(ImplementsNecessaryColumns, self.io_func).necessary_columns(
            report=report, state=state
        )

    def project_manually(self, columns: frozenset[str]) -> AwkwardInputLayer:
        """Project the necessary _columns_ to the AwkwardInputLayer."""
        assert self.is_projectable
        io_func = cast(ImplementsProjection, self.io_func).project_manually(columns)
        return AwkwardInputLayer(
            name=self.name,
            inputs=self.inputs,
            io_func=io_func,
            label=self.label,
            produces_tasks=self.produces_tasks,
            creation_info=self.creation_info,
            annotations=self.annotations,
        )


class AwkwardMaterializedLayer(MaterializedLayer):
    def __init__(
        self,
        mapping: dict,
        *,
        previous_layer_names: list[str],
        fn: Callable | None = None,
        **kwargs: Any,
    ):
        self.previous_layer_names: list[str] = previous_layer_names
        self.fn = fn
        super().__init__(mapping, **kwargs)

    def mock(self) -> MaterializedLayer:
        mapping = copy.copy(self.mapping)
        if not mapping:
            # no partitions at all
            return self
        name = next(iter(mapping))[0]

        npln = len(self.previous_layer_names)
        # one previous layer name
        #
        # this case is used for mocking repartition or slicing where
        # we maybe have multiple partitions that need to be included
        # in a task.
        if npln == 1:
            prev_name: str = self.previous_layer_names[0]
            if (name, 0) in mapping:
                task = mapping[(name, 0)]
                task = tuple(
                    (
                        (prev_name, 0)
                        if isinstance(v, tuple) and len(v) == 2 and v[0] == prev_name
                        else v
                    )
                    for v in task
                )

                # when using Array.partitions we need to mock that we
                # just want the first partition.
                if len(task) == 2 and isinstance(task[1], int) and task[1] > 0:
                    task = (task[0], 0)
                return MaterializedLayer({(name, 0): task})
            return self

        # zero previous layers; this is likely a known scalar.
        #
        # we just use the existing mapping
        elif npln == 0:
            return MaterializedLayer({(name, 0): mapping[(name, 0)]})

        # more than one previous_layer_names
        #
        # this case is needed for dak.concatenate on axis=0; we need
        # the first partition of _each_ of the previous layer names!
        else:
            if self.fn is None:
                raise ValueError(
                    "For multiple previous layers the fn argument cannot be None."
                )
            name0s = tuple((name, 0) for name in self.previous_layer_names)
            task = (self.fn, *name0s)
            return MaterializedLayer({(name, 0): task})


class AwkwardTreeReductionLayer(Layer):
    """Awkward Tree-Reduction Layer
    Parameters
    ----------
    name : str
        Name to use for the constructed layer.
    name_input : str
        Name of the input layer that is being reduced.
    npartitions_input : str
        Number of partitions in the input layer.
    concat_func : callable
        Function used by each tree node to reduce a list of inputs
        into a single output value. This function must accept only
        a list as its first positional argument.
    tree_node_func : callable
        Function used on the output of ``concat_func`` in each tree
        node. This function must accept the output of ``concat_func``
        as its first positional argument.
    finalize_func : callable, optional
        Function used in place of ``tree_node_func`` on the final tree
        node(s) to produce the final output for each split. By default,
        ``tree_node_func`` will be used.
    split_every : int, optional
        This argument specifies the maximum number of input nodes
        to be handled by any one task in the tree. Defaults to 32.
    split_out : int, optional
        This argument specifies the number of output nodes in the
        reduction tree. If ``split_out`` is set to an integer >=1, the
        input tasks must contain data that can be indexed by a ``getitem``
        operation with a key in the range ``[0, split_out)``.
    output_partitions : list, optional
        List of required output partitions. This parameter is used
        internally by Dask for high-level culling.
    tree_node_name : str, optional
        Name to use for intermediate tree-node tasks.
    """

    name: str
    name_input: str
    npartitions_input: int
    concat_func: Callable
    tree_node_func: Callable
    finalize_func: Callable | None
    split_every: int
    split_out: int
    output_partitions: list[int]
    tree_node_name: str
    widths: list[int]
    height: int

    def __init__(
        self,
        name: str,
        name_input: str,
        npartitions_input: int,
        concat_func: Callable,
        tree_node_func: Callable,
        finalize_func: Callable | None = None,
        split_every: int = 32,
        split_out: int | None = None,
        output_partitions: list[int] | None = None,
        tree_node_name: str | None = None,
        annotations: dict[str, Any] | None = None,
    ):
        super().__init__(annotations=annotations)
        self.name = name
        self.name_input = name_input
        self.npartitions_input = npartitions_input
        self.concat_func = concat_func
        self.tree_node_func = tree_node_func
        self.finalize_func = finalize_func
        self.split_every = split_every
        self.split_out = split_out  # type: ignore
        self.output_partitions = (
            list(range(self.split_out or 1))
            if output_partitions is None
            else output_partitions
        )
        self.tree_node_name = tree_node_name or "tree_node-" + self.name

        # Calculate tree widths and height
        # (Used to get output keys without materializing)
        parts = self.npartitions_input
        self.widths = [parts]
        while parts > 1:
            parts = math.ceil(parts / self.split_every)
            self.widths.append(int(parts))
        self.height = len(self.widths)

    def _make_key(self, *name_parts, split=0):
        # Helper function construct a key
        # with a "split" element when
        # bool(split_out) is True
        return name_parts + (split,) if self.split_out else name_parts

    def _define_task(self, input_keys, final_task=False):
        # Define nested concatenation and func task
        if final_task and self.finalize_func:
            outer_func = self.finalize_func
        else:
            outer_func = self.tree_node_func
        return (toolz.pipe, input_keys, self.concat_func, outer_func)

    def _construct_graph(self):
        """Construct graph for a tree reduction."""

        dsk = {}
        if not self.output_partitions:
            return dsk

        # Deal with `bool(split_out) == True`.
        # These cases require that the input tasks
        # return a type that enables getitem operation
        # with indices: [0, split_out)
        # Therefore, we must add "getitem" tasks to
        # select the appropriate element for each split
        name_input_use = self.name_input
        if self.split_out:
            name_input_use += "-split"
            for s in self.output_partitions:
                for p in range(self.npartitions_input):
                    dsk[self._make_key(name_input_use, p, split=s)] = (
                        operator.getitem,
                        (self.name_input, p),
                        s,
                    )

        if self.height >= 2:
            # Loop over output splits
            for s in self.output_partitions:
                # Loop over reduction levels
                for depth in range(1, self.height):
                    # Loop over reduction groups
                    for group in range(self.widths[depth]):
                        # Calculate inputs for the current group
                        p_max = self.widths[depth - 1]
                        lstart = self.split_every * group
                        lstop = min(lstart + self.split_every, p_max)
                        if depth == 1:
                            # Input nodes are from input layer
                            input_keys = [
                                self._make_key(name_input_use, p, split=s)
                                for p in range(lstart, lstop)
                            ]
                        else:
                            # Input nodes are tree-reduction nodes
                            input_keys = [
                                self._make_key(
                                    self.tree_node_name, p, depth - 1, split=s
                                )
                                for p in range(lstart, lstop)
                            ]

                        # Define task
                        if depth == self.height - 1:
                            # Final Node (Use fused `self.tree_finalize` task)
                            assert (
                                group == 0
                            ), f"group = {group}, not 0 for final tree reduction task"
                            dsk[(self.name, s)] = self._define_task(
                                input_keys, final_task=True
                            )
                        else:
                            # Intermediate Node
                            dsk[
                                self._make_key(
                                    self.tree_node_name, group, depth, split=s
                                )
                            ] = self._define_task(input_keys, final_task=False)
        else:
            # Deal with single-partition case
            for s in self.output_partitions:
                input_keys = [self._make_key(name_input_use, 0, split=s)]
                dsk[(self.name, s)] = self._define_task(input_keys, final_task=True)

        return dsk

    def __repr__(self):
        return "DataFrameTreeReduction<name='{}', input_name={}, split_out={}>".format(
            self.name, self.name_input, self.split_out
        )

    def _output_keys(self):
        return {(self.name, s) for s in self.output_partitions}

    def get_output_keys(self):
        if hasattr(self, "_cached_output_keys"):
            return self._cached_output_keys
        else:
            output_keys = self._output_keys()
            self._cached_output_keys = output_keys
        return self._cached_output_keys

    def is_materialized(self):
        return hasattr(self, "_cached_dict")

    @property
    def _dict(self):
        """Materialize full dict representation"""
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        # Start with "base" tree-reduction size
        tree_size = (sum(self.widths[1:]) or 1) * (self.split_out or 1)
        if self.split_out:
            # Add on "split-*" tasks used for `getitem` ops
            return tree_size + self.npartitions_input * len(self.output_partitions)
        return tree_size

    def _keys_to_output_partitions(self, keys):
        """Simple utility to convert keys to output partition indices."""
        splits = set()
        for key in keys:
            try:
                _name, _split = key
            except ValueError:
                continue
            if _name != self.name:
                continue
            splits.add(_split)
        return splits

    def _cull(self, output_partitions):
        return AwkwardTreeReductionLayer(
            self.name,
            self.name_input,
            self.npartitions_input,
            self.concat_func,
            self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            split_out=self.split_out,
            output_partitions=output_partitions,
            tree_node_name=self.tree_node_name,
            annotations=self.annotations,
        )

    def cull(self, keys, all_keys):
        """Cull a DataFrameTreeReduction HighLevelGraph layer"""
        deps = {
            (self.name, 0): {
                (self.name_input, i) for i in range(self.npartitions_input)
            }
        }
        output_partitions = self._keys_to_output_partitions(keys)
        if output_partitions != set(self.output_partitions):
            culled_layer = self._cull(output_partitions)
            return culled_layer, deps
        else:
            return self, deps

    def mock(self) -> AwkwardTreeReductionLayer:
        return AwkwardTreeReductionLayer(
            name=self.name,
            name_input=self.name_input,
            npartitions_input=1,
            concat_func=self.concat_func,
            tree_node_func=self.tree_node_func,
            finalize_func=self.finalize_func,
            split_every=self.split_every,
            tree_node_name=self.tree_node_name,
        )
