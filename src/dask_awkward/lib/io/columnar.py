from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import awkward as ak
from awkward import Array as AwkwardArray
from awkward.forms import Form
from awkward.typetracer import typetracer_from_form, typetracer_with_report

from dask_awkward.layers.layers import (
    BackendT,
    ImplementsIOFunction,
    ImplementsNecessaryColumns,
)
from dask_awkward.lib.utils import (
    METADATA_ATTRIBUTES,
    FormStructure,
    buffer_keys_required_to_compute_shapes,
    form_with_unique_keys,
    parse_buffer_key,
    render_buffer_key,
    trace_form_structure,
    walk_graph_depth_first,
)

if TYPE_CHECKING:
    from awkward._nplikes.typetracer import TypeTracerReport

log = logging.getLogger(__name__)

T = TypeVar("T")


class ImplementsColumnProjectionMixin(ImplementsNecessaryColumns, Protocol):
    @property
    def form(self) -> Form: ...

    @property
    def behavior(self) -> dict | None: ...

    @property
    def attrs(self) -> dict | None: ...

    def project_columns(self: T, columns: frozenset[str]) -> T: ...

    def project_manually(self: T, columns: frozenset[str]) -> ImplementsIOFunction: ...

    def __call__(self, *args, **kwargs): ...


S = TypeVar("S", bound=ImplementsColumnProjectionMixin)


class ColumnProjectionMixin(ImplementsNecessaryColumns[FormStructure]):
    """A mixin to add column-centric buffer projection to an IO function.

    Classes that inherit from this mixin are assumed to be able to read at the
    granularity of _fields_ in a form. As such, the buffer projection is performed
    such that the smallest possible number of fields (columns) are read, even
    when only metadata buffers are required.
    """

    def mock(self: S) -> AwkwardArray:
        return cast(
            AwkwardArray,
            typetracer_from_form(self.form, behavior=self.behavior, attrs=self.attrs),
        )

    def mock_empty(self: S, backend: BackendT = "cpu") -> AwkwardArray:
        return cast(
            AwkwardArray,
            ak.to_backend(
                self.form.length_zero_array(highlevel=False, behavior=self.behavior),
                backend,
                highlevel=True,
            ),
        )

    def prepare_for_projection(
        self: S,
    ) -> tuple[AwkwardArray, TypeTracerReport, FormStructure]:
        form = form_with_unique_keys(self.form, "@")

        # Build typetracer and associated report object
        (meta, report) = typetracer_with_report(
            form,
            highlevel=True,
            behavior=self.behavior,
            buffer_key=render_buffer_key,
        )

        return (
            cast(AwkwardArray, meta),
            report,
            trace_form_structure(form, buffer_key=render_buffer_key),
        )

    def necessary_columns(
        self: S,
        report: TypeTracerReport,
        state: FormStructure,
    ) -> frozenset[str]:
        ## Read from stash
        # Form hierarchy information
        form_key_to_parent_form_key = state["form_key_to_parent_form_key"]
        form_key_to_child_form_keys: dict[str, list[str]] = {}
        for child_key, parent_key in form_key_to_parent_form_key.items():
            form_key_to_child_form_keys.setdefault(parent_key, []).append(child_key)  # type: ignore
        form_key_to_form = state["form_key_to_form"]
        # Buffer hierarchy information
        form_key_to_buffer_keys = state["form_key_to_buffer_keys"]
        # Column hierarchy information
        form_key_to_path = state["form_key_to_path"]

        # Require the data of metadata buffers above shape-only requests
        data_buffers = {
            *report.data_touched,
            *buffer_keys_required_to_compute_shapes(
                parse_buffer_key,
                report.shape_touched,
                form_key_to_parent_form_key,
                form_key_to_buffer_keys,
            ),
        }

        # We can't read buffers directly, but if we encounter a metadata
        # buffer, then we should be able to pick any child.
        paths = set()
        wildcard_form_key = set()
        for buffer_key in data_buffers:
            form_key, attribute = parse_buffer_key(buffer_key)
            if attribute in METADATA_ATTRIBUTES:
                wildcard_form_key.add(form_key)
            else:
                paths.add(form_key_to_path[form_key])

        # Select the most appropriate column for each wildcard
        for form_key in wildcard_form_key:
            # Find (DFS) any non-empty record form in any child
            recursive_child_forms = (
                form_key_to_form[k]
                for k in walk_graph_depth_first(form_key, form_key_to_child_form_keys)
            )
            record_form_keys_with_contents = (
                f.form_key
                for f in recursive_child_forms
                if isinstance(f, ak.forms.RecordForm) and f.contents
            )
            # Now find the deepest of such records
            try:
                last_record_form_key = next(record_form_keys_with_contents)
            except StopIteration:
                # This is a leaf! Therefore, we read this column
                paths.add(form_key_to_path[form_key])
                continue
            else:
                # Ensure we get the "actual" last form key
                for last_record_form_key in record_form_keys_with_contents:
                    ...

            # First see if any child is already included
            for any_child_form_key in form_key_to_child_form_keys[last_record_form_key]:
                any_child_path = form_key_to_path[any_child_form_key]
                if any_child_path in paths:
                    break
            # Otherwise, add the last child
            else:
                paths.add(any_child_path)
        return frozenset({".".join(p) for p in paths if p})

    def project(
        self: S,
        report: TypeTracerReport,
        state: FormStructure,
    ) -> ImplementsIOFunction:
        if not self.use_optimization:  # type: ignore[attr-defined]
            return self

        return self.project_columns(self.necessary_columns(report, state))

    def project_manually(self: S, columns: frozenset[str]) -> ImplementsIOFunction:
        return self.project_columns(columns)
