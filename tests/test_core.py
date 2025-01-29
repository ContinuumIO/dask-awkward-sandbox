from __future__ import annotations

import copy
import json
import operator
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import awkward as ak
import dask.array as da
import dask.config
import fsspec
import numpy as np
import pytest
from dask.delayed import delayed

import dask_awkward as dak
from dask_awkward.lib.core import (
    Record,
    Scalar,
    calculate_known_divisions,
    compute_typetracer,
    empty_typetracer,
    is_typetracer,
    map_partitions,
    meta_or_identity,
    new_array_object,
    new_known_scalar,
    new_record_object,
    new_scalar_object,
    normalize_single_outer_inner_index,
    to_meta,
    typetracer_array,
)
from dask_awkward.lib.testutils import assert_eq
from dask_awkward.utils import ConcretizationTypeError, IncompatiblePartitions

if TYPE_CHECKING:
    from dask_awkward.lib.core import Array


def test_clear_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    assert daa.known_divisions
    daa.clear_divisions()
    assert len(daa.divisions) == daa.npartitions + 1
    assert not daa.known_divisions


def test_dunder_str(caa: ak.Array) -> None:
    daa = dak.from_awkward(caa, npartitions=2)
    assert str(daa) == "dask.awkward<from-awkward, npartitions=2>"


def test_calculate_known_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    target = (0, 5, 10, 15)
    assert calculate_known_divisions(daa) == target
    assert calculate_known_divisions(daa.points) == target
    assert calculate_known_divisions(daa.points.x) == target
    assert calculate_known_divisions(daa["points"]["y"]) == target
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    assert daa.known_divisions
    assert calculate_known_divisions(daa) == target


def test_fields(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    # records fields same as array of records fields
    assert daa[0].points.fields == daa.points.fields
    aa = daa.compute()
    assert daa.fields == aa.fields
    daa.reset_meta()
    assert daa.fields == []


def test_form(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    assert daa.form
    daa.reset_meta()

    from awkward.forms.emptyform import EmptyForm

    assert daa.form == EmptyForm()


@pytest.mark.parametrize("nparts", [2, 3, 4])
def test_from_awkward(caa: ak.Array, nparts: int) -> None:
    daa = dak.from_awkward(caa, npartitions=nparts)
    assert_eq(caa, daa, check_forms=False)
    assert_eq(daa, daa, check_forms=False)


def test_compute_typetracer(daa: Array) -> None:
    tt = compute_typetracer(daa.dask, daa.name)
    daa2 = new_array_object(daa.dask, daa.name, meta=tt, divisions=daa.divisions)
    assert_eq(daa, daa2)


def test_len(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert not daa.known_divisions
    with pytest.raises(
        TypeError,
        match=(
            "Cannot determine length of collection with unknown partition sizes without executing the graph.\\n"
            "Use `dask_awkward.num\\(\\.\\.\\., axis=0\\)` if you want a lazy Scalar of the length.\\n"
            "If you want to eagerly compute the partition sizes to have the ability to call `len` on the collection"
            ", use `\\.eager_compute_divisions\\(\\)` on the collection."
        ),
    ):
        assert len(daa) == 10
    daa.eager_compute_divisions()
    assert daa.known_divisions
    assert len(daa) == 10


def test_meta_exists(daa: Array) -> None:
    assert daa._meta is not None
    assert daa["points"]._meta is not None


def test_ndim(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert daa.ndim == daa.compute().ndim


def test_new_array_object_raises(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    name = daa.name
    hlg = daa.dask
    with pytest.raises(
        ValueError, match="One of either divisions or npartitions must be defined."
    ):
        new_array_object(hlg, name, meta=None, npartitions=None, divisions=None)
    with pytest.raises(
        ValueError, match="Only one of either divisions or npartitions can be defined."
    ):
        new_array_object(hlg, name, meta=None, npartitions=3, divisions=(0, 2, 4, 7))


def test_partitions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 4)
    for i in range(daa.npartitions):
        part = daa.partitions[i]
        assert part.npartitions == 1


def test_partitions_divisions(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 3)
    daa.eager_compute_divisions()
    divs = daa.divisions
    t1 = daa.partitions[1:3]
    assert not t1.known_divisions
    t2 = daa.partitions[1]
    assert t2.known_divisions
    assert t2.divisions == (0, divs[2] - divs[1])


def test_array_rebuild(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    x = daa.compute()
    daa = daa._rebuild(daa.dask)
    y = daa.compute()
    assert x.tolist() == y.tolist()


def test_type(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file] * 2)
    assert dak.type(daa) is not None
    daa._meta = None
    assert dak.type(daa) is None


def test_short_typestr(daa: Array) -> None:
    ts = daa._shorttypestr(max=12)
    assert len(ts) == 12


def test_typestr(daa: Array) -> None:
    aa = daa.compute()
    assert str(aa.layout.form.type) in daa._typestr()
    extras = len("var *  ... }")
    assert len(daa._typestr(max=20)) == 20 + extras


def test_head(daa: Array) -> None:
    out = daa.head(1)
    assert out.tolist() == daa.compute()[:1].tolist()

    out = daa.head(6)  # first partition only has 5 rows
    assert out.tolist() == daa.compute()[:5].tolist()

    out = daa.head(1, compute=False)
    assert isinstance(out, dak.lib.Array)
    assert out.divisions == (None, None)  # since where not known

    daa.eager_compute_divisions()
    out = daa.head(1, compute=False)
    assert isinstance(out, dak.lib.Array)
    assert out.divisions == (0, 1)


def test_record_collection(daa: Array) -> None:
    assert type(daa[0]) is Record
    aa = daa.compute()
    assert_eq(daa[0], aa[0])
    # assert daa[0].compute().tolist() == aa[0].tolist()


def test_scalar_collection(daa: Array) -> None:
    assert type(daa["points", "x"][0][0]) is Scalar


def test_known_scalar() -> None:
    i = 5
    s = new_known_scalar(5)
    assert s.compute() == 5
    with pytest.raises(AttributeError, match="should be done after converting"):
        s.denominator.compute()
    assert s.to_delayed().denominator.compute() == i.denominator


@pytest.mark.parametrize("op", [operator.add, operator.truediv, operator.mul])
def test_scalar_binary_ops(op: Callable, daa: Array, caa: ak.Array) -> None:
    a1 = dak.max(daa.points.x, axis=None)
    b1 = dak.min(daa.points.y, axis=None)
    a2 = ak.max(caa.points.x, axis=None)
    b2 = ak.min(caa.points.y, axis=None)
    assert_eq(op(a1, b1), op(a2, b2))


@pytest.mark.parametrize("op", [operator.add, operator.truediv, operator.mul])
def test_scalar_binary_ops_other_not_dak(
    op: Callable, daa: Array, caa: ak.Array
) -> None:
    a1 = dak.max(daa.points.x, axis=None)
    a2 = ak.max(caa.points.x, axis=None)
    assert_eq(op(a1, 5), op(a2, 5))


@pytest.mark.parametrize("op", [operator.abs])
def test_scalar_unary_ops(op: Callable, daa: Array, caa: ak.Array) -> None:
    a1 = dak.max(daa.points.x, axis=None)
    a2 = ak.max(caa.points.x, axis=None)
    assert_eq(op(-a1), op(-a2))


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.pow])
def test_array_broadcast_scalar(op: Callable, daa: Array, caa: Array) -> None:
    s1 = new_known_scalar(3)
    s2 = 3
    r1 = op(daa.points.x, s1)
    r2 = op(caa.points.x, s2)
    assert_eq(r1, r2)

    s3 = dak.min(daa.points.x, axis=None)
    s4 = ak.min(caa.points.x, axis=None)
    r3 = op(daa.points.y, s3)
    r4 = op(caa.points.y, s4)
    assert_eq(r3, r4)

    s5 = dak.max(daa.points.y, axis=None)
    s6 = ak.max(caa.points.y, axis=None)
    r5 = op(daa.points.x, s5)
    r6 = op(caa.points.x, s6)
    assert_eq(r5, r6)


@pytest.mark.parametrize(
    "where",
    [
        slice(0, 10),
        slice(0, 11),
        slice(1, 10),
        slice(1, 11),
        slice(1, 3),
        slice(6, 12),
        slice(0, 10, 2),
        slice(0, 11, 2),
        slice(1, 14, 2),
        slice(1, 11, 2),
        slice(1, 3, 3),
        slice(None, None, 3),
    ],
)
def test_getitem_zero_slice_single(daa: Array, where: slice) -> None:
    out = daa[where]
    assert out.compute().tolist() == daa.compute()[where].tolist()
    assert len(out) == len(daa.compute()[where])


@pytest.mark.parametrize(
    "where",
    [
        slice(0, 10),
        slice(0, 11),
        slice(1, 10),
        slice(1, 11),
        slice(1, 3),
        slice(6, 12),
        slice(0, 10, 2),
        slice(0, 11, 2),
        slice(1, 14, 2),
        slice(1, 11, 2),
        slice(1, 3, 3),
        slice(None, None, 3),
    ],
)
@pytest.mark.parametrize("rest", [slice(None, None, None), slice(0, 1)])
def test_getitem_zero_slice_tuple(
    daa: Array,
    where: slice,
    rest: slice,
) -> None:
    out = daa[where, rest]
    assert out.compute().tolist() == daa.compute()[where, rest].tolist()
    assert len(out) == len(daa.compute()[where, rest])


@pytest.mark.parametrize(
    "where",
    [
        slice(None, 10, None),
        slice(-30, None, None),
        slice(10, 68, 5),
        slice(None, 5, 2),
        slice(None, 15, 3),
        slice(15, None, 6),
        slice(62, None, None),
        slice(35, 70, None),
        slice(None, None, 3),
    ],
)
def test_getitem_zero_slice_divisions(where):
    concrete = ak.Array([[1, 2, 3], [4], [5, 6, 7], [8, 9]] * 25)
    lazy = dak.from_awkward(concrete, npartitions=4)

    conc_sliced = concrete[where]
    lazy_sliced = lazy[where]
    assert_eq(conc_sliced, lazy_sliced, check_forms=False)

    divs = [0]
    for i in range(lazy_sliced.npartitions):
        divs.append(len(lazy_sliced.partitions[i].compute()) + divs[i])
    assert lazy_sliced.divisions == tuple(divs)
    assert len(lazy_sliced) == len(conc_sliced)


def test_is_typetracer(daa: Array) -> None:
    assert not is_typetracer(daa)
    assert not is_typetracer(daa[0])
    assert not is_typetracer(daa["points"])
    assert not is_typetracer(daa.compute())
    assert is_typetracer(daa._meta)
    assert is_typetracer(daa[0]._meta)
    assert is_typetracer(daa["points"]._meta)
    assert is_typetracer(daa["points"][0]["x"][0]._meta)


def test_meta_or_identity(daa: Array) -> None:
    assert is_typetracer(meta_or_identity(daa))
    assert meta_or_identity(daa) is daa._meta
    assert meta_or_identity(5) == 5


def test_to_meta(daa: Array) -> None:
    x1 = daa["points"]["x"]
    x1_0 = x1[0]
    metad = to_meta([x1, 5, "ok", x1_0])
    assert isinstance(metad, tuple)
    for a, b in zip(metad, (x1._meta, 5, "ok", x1_0._meta)):
        if is_typetracer(a):
            assert a is b
        else:
            assert a == b


def test_record_str(daa: Array) -> None:
    r = daa[0]
    assert type(r) is dak.Record
    assert str(r) == "dask.awkward<getitem, type=Record>"


def test_record_to_delayed(daa: Array) -> None:
    r = daa[0]
    assert type(r) is dak.Record
    d = r.to_delayed()
    x = r.compute().tolist()
    y = d.compute().tolist()
    assert x == y


def test_record_fields(daa: Array) -> None:
    r = daa[0]
    assert type(r) is dak.Record
    r._meta = None
    with pytest.raises(TypeError, match="metadata is missing"):
        assert not r.fields


def test_record_dir(daa: Array) -> None:
    r = daa["points"][0][0]
    assert type(r) is dak.Record
    d = dir(r)
    for f in r.fields:
        assert f in d


# @pytest.mark.xfail(reason="ak.Record typetracer fails to pickle")
# def test_record_pickle(daa: Array) -> None:
#     import pickle

#     r = daa[0]
#     assert type(r) is dak.Record
#     assert isinstance(r._meta, ak.Record)

#     dumped = pickle.dumps(r)
#     new = pickle.loads(dumped)
#     assert_eq(dumped, new)


def test_array_dir(daa: Array) -> None:
    a = daa["points"]
    d = dir(a)
    for f in a.fields:
        assert f in d


def test_typetracer_function(daa: Array) -> None:
    aa = daa.compute()
    assert typetracer_array(daa) is not None
    assert typetracer_array(daa) is daa._meta
    tta = typetracer_array(aa)
    assert tta is not None
    assert tta.layout.form == aa.layout.form
    with pytest.raises(TypeError, match="Got type <class 'int'>"):
        typetracer_array(3)


def test_single_partition(ndjson_points_file: str) -> None:
    daa = dak.from_json([ndjson_points_file])
    with fsspec.open(ndjson_points_file, "r") as f:
        caa = ak.from_iter([json.loads(line) for line in f])

    assert daa.npartitions == 1
    assert_eq(daa, caa)
    assert_eq(caa, daa)


def test_new_known_scalar() -> None:
    s1 = 5
    c = new_known_scalar(s1)
    assert c.compute() == s1
    assert c._meta is not None
    s2 = 5.5
    c = new_known_scalar(s2)
    assert c.compute() == 5.5
    assert c._meta is not None

    c = dak.Scalar.from_known(s1)
    assert c.known_value == s1
    assert c.compute() == s1


def test_scalar_dtype() -> None:
    s = 2
    c = new_known_scalar(s)
    assert c.dtype == np.dtype(type(s))


def test_scalar_pickle(daa: Array) -> None:
    import cloudpickle as pickle

    s = 2
    c1 = new_known_scalar(s)
    s_dumped = pickle.dumps(c1)
    c2 = pickle.loads(s_dumped)
    assert_eq(c1, c2)
    s1 = dak.sum(daa["points"]["y"], axis=None)
    s_dumped = pickle.dumps(s1)
    s2 = pickle.loads(s_dumped)

    # TODO: workaround since dask un/pack disappeared
    for lay2, lay1 in zip(s2.dask.layers.values(), s1.dask.layers.values()):
        if hasattr(lay1, "_meta"):
            lay2._meta = lay1._meta
    assert_eq(s1.compute(), s2.compute())

    assert s1.known_value is None


@pytest.mark.parametrize("optimize_graph", [True, False])
def test_scalar_to_delayed(daa: Array, optimize_graph: bool) -> None:
    s1 = dak.sum(daa["points", "x"], axis=None)
    d1 = s1.to_delayed(optimize_graph=optimize_graph)
    s1c = s1.compute()
    assert d1.compute() == s1c


def test_defined_divisions_exception(ndjson_points1):
    jsds = dak.from_json([ndjson_points1] * 3)
    with pytest.raises(ValueError, match="defined_divisions only works"):
        jsds.defined_divisions


def test_compatible_partitions(ndjson_points_file: str) -> None:
    daa1 = dak.from_json([ndjson_points_file] * 5)
    daa2 = dak.from_awkward(daa1.compute(), npartitions=4)
    assert dak.compatible_partitions(daa1, daa1)
    assert dak.compatible_partitions(daa1, daa1, daa1)
    assert not dak.compatible_partitions(daa1, daa2)
    daa1.eager_compute_divisions()
    assert dak.compatible_partitions(daa1, daa1)
    x = ak.Array([[1, 2, 3], [1, 2, 3], [3, 4, 5]])
    y = ak.Array([[1, 2, 3], [3, 4, 5]])
    x = dak.from_awkward(x, npartitions=2)
    y = dak.from_awkward(y, npartitions=2)
    assert not dak.compatible_partitions(x, y)
    assert not dak.compatible_partitions(x, x, y)
    assert dak.compatible_partitions(y, y)


def test_compatible_partitions_after_slice() -> None:
    a = [[1, 2, 3], [4, 5]]
    b = [[5, 6, 7, 8], [], [9]]
    lazy = dak.from_lists([a, b])
    ccrt = ak.Array(a + b)

    # sanity
    assert_eq(lazy, ccrt)

    # sanity
    assert dak.compatible_partitions(lazy, lazy + 2)
    assert dak.compatible_partitions(lazy, dak.num(lazy, axis=1) > 2)

    assert not dak.compatible_partitions(lazy[:-2], lazy)
    assert not dak.compatible_partitions(lazy[:-2], dak.num(lazy, axis=1) != 3)

    with pytest.raises(IncompatiblePartitions, match="incompatibly partitioned"):
        (lazy[:-2] + lazy).compute()


def test_compatible_partitions_mixed() -> None:
    a = ak.Array([[1, 2, 3], [0, 0, 0, 0], [5, 6, 7, 8, 9], [0, 0, 0, 0]])
    b = dak.from_awkward(a, npartitions=2)
    assert b.known_divisions
    c = b[dak.num(b, axis=1) == 4]
    d = b[dak.num(b, axis=1) >= 3]
    assert not c.known_divisions
    # compatible partitions is going to get called in the __add__ ufunc
    e = b + c
    f = b + d
    with pytest.raises(ValueError):
        e.compute()
    assert_eq(f, a + a)


def test_compatible_partitions_all_unknown() -> None:
    a = ak.Array([[1, 2, 3], [0, 0, 0, 0], [5, 6, 7, 8, 9], [0, 0, 0, 0]])
    b = dak.from_awkward(a, npartitions=2)
    c = b[dak.sum(b, axis=1) == 0]
    d = b[dak.sum(b, axis=1) == 6]
    # this will pass compatible partitions which gets called in the
    # __add__ ufunc; both have unknown divisions but equal number of
    # partitions. the unknown divisions are going to materialize to be
    # incompatible so an exception will get raised at compute time.
    e = c + d
    with pytest.raises(ValueError):
        e.compute()


def test_partition_compatiblity() -> None:
    a = ak.Array([[1, 2, 3], [0, 0, 0, 0], [5, 6, 7, 8, 9], [0, 0, 0, 0]])
    b = dak.from_awkward(a, npartitions=2)
    c = b[dak.sum(b, axis=1) == 0]
    d = b[dak.sum(b, axis=1) == 6]
    assert dak.partition_compatibility(c, d) == dak.PartitionCompatibility.MAYBE
    assert dak.partition_compatibility(b, c, d) == dak.PartitionCompatibility.MAYBE
    assert (
        dak.partition_compatibility(b, dak.num(b, axis=1))
        == dak.PartitionCompatibility.YES
    )
    c.eager_compute_divisions()
    assert dak.partition_compatibility(b, c) == dak.PartitionCompatibility.NO


def test_partition_compat_with_strictness() -> None:
    a = ak.Array([[1, 2, 3], [0, 0, 0, 0], [5, 6, 7, 8, 9], [0, 0, 0, 0]])
    b = dak.from_awkward(a, npartitions=2)
    c = b[dak.sum(b, axis=1) == 0]
    d = b[dak.sum(b, axis=1) == 6]

    assert dak.compatible_partitions(c, d, how_strict=1)
    assert dak.compatible_partitions(
        c,
        d,
        how_strict=dak.PartitionCompatibility.MAYBE,
    )

    assert not dak.compatible_partitions(c, d, how_strict=2)
    assert not dak.compatible_partitions(
        c,
        d,
        how_strict=dak.PartitionCompatibility.YES,
    )


def test_bad_meta_backend_array(daa):
    with pytest.raises(TypeError, match="meta Array must have a typetracer backend"):
        daa.points.x.map_partitions(lambda x: x**2, meta=ak.Array([]))


def test_bad_meta_backend_record(daa):
    with pytest.raises(TypeError, match="meta Record must have a typetracer backend"):
        a = daa.points[0]
        new_record_object(a.dask, a.name, meta=ak.Record({"x": 1}))


def test_bad_meta_backend_scalar(daa):
    with pytest.raises(TypeError, match="meta Scalar must have a typetracer backend"):
        a = daa.points.x[0][0]
        new_scalar_object(a.dask, a.name, meta=5)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="skip if windows")
def test_scalar_repr(daa: Array) -> None:
    s = dak.max(daa.points.y)
    sstr = str(s)
    assert "type=Scalar" in sstr
    s = new_known_scalar(5)
    sstr = str(s)
    assert (
        sstr == r"dask.awkward<known-scalar, type=Scalar, dtype=int64, known_value=5>"
    )


def test_scalar_divisions(daa: Array) -> None:
    s = dak.max(daa.points.x, axis=None)
    assert s.divisions == (None, None)


def test_scalar_binop_inv() -> None:
    # GH #515
    x = dak.from_lists([[1]])
    y = x[0]  # scalar
    assert (0 - y) == -1
    assert (y - 0) == 1


def test_array_persist(daa: Array) -> None:
    daa2 = daa["points"]["x"].persist()
    assert_eq(daa["points"]["x"], daa2)
    daa2 = daa.persist()
    assert_eq(daa2, daa)


def test_scalar_persist(daa: Array) -> None:
    coll = daa["points"][0]["x"][0]
    coll2 = coll.persist()
    assert_eq(coll, coll2)


def test_array_rename_when_rebuilding(daa: Array) -> None:
    name = daa.name
    new_name = "foobar"
    assert daa._rebuild(dsk=daa.dask, rename={name: new_name}).name == new_name


def test_output_divisions(daa: Array) -> None:
    assert dak.max(daa.points.y, axis=1).divisions == daa.divisions
    assert daa["points"][["x", "y"]].divisions == daa.divisions
    assert daa["points"].divisions == daa.divisions


def test_record_npartitions(daa: Array) -> None:
    analysis0 = daa[0]
    assert analysis0.npartitions == 1


def test_iter(daa: Array) -> None:
    with pytest.raises(
        NotImplementedError,
        match="Iteration over a Dask Awkward collection is not supported",
    ):
        for a in daa:
            pass


def test_normalize_single_outer_inner_index() -> None:
    divisions = (0, 12, 14, 20, 23, 24)
    indices = [0, 1, 2, 8, 12, 13, 14, 15, 17, 20, 21, 22]
    results = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 8),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    for i, r in zip(indices, results):
        res = normalize_single_outer_inner_index(divisions, i)
        assert r == res

    divisions = (0, 12)  # type: ignore
    indices = [0, 2, 3, 6, 8, 11]
    results = [
        (0, 0),
        (0, 2),
        (0, 3),
        (0, 6),
        (0, 8),
        (0, 11),
    ]
    for i, r in zip(indices, results):
        res = normalize_single_outer_inner_index(divisions, i)
        assert r == res


def test_optimize_chain_single(daa):
    import dask

    from dask_awkward.lib.optimize import rewrite_layer_chains

    arr = ((daa.points.x + 1) + 6).map_partitions(lambda x: x + 1)

    # first a simple test by calling the one optimisation directly
    dsk2 = rewrite_layer_chains(arr.dask, arr.keys)
    (out,) = dask.compute(arr, optimize_graph=False)
    arr._dask = dsk2
    (out2,) = dask.compute(arr, optimize_graph=False)
    assert out.tolist() == out2.tolist()

    # and now with optimise as part of the usual pipeline
    arr = ((daa.points.x + 1) + 6).map_partitions(lambda x: x + 1)
    out = arr.compute()
    assert out.tolist() == out2.tolist()


def test_optimize_chain_multiple(daa):
    result = (daa.points.x**2 - daa.points.y) + 1

    assert len(result.compute()) > 0


def test_make_unknown_length():
    from dask_awkward.lib.core import make_unknown_length

    arr = ak.Array(
        [
            {"a": [1, 2, 3], "b": 5},
            {"a": [], "b": -1},
            {"a": [9, 8, 7, 6], "b": 0},
        ]
    )
    tt1 = ak.Array(arr.layout.to_typetracer())

    # sanity checks
    assert ak.backend(tt1) == "typetracer"
    assert len(tt1) == 3

    ul_arr = make_unknown_length(arr)
    ul_tt1 = make_unknown_length(tt1)

    assert ul_arr.layout.form == ul_tt1.layout.form

    with pytest.raises(TypeError, match="cannot interpret unknown lengths"):
        len(ul_arr)

    with pytest.raises(TypeError, match="cannot interpret unknown lengths"):
        len(ul_tt1)


def my_power(arg_x, *, kwarg_y=None):
    return arg_x**kwarg_y


def structured_function(*, inputs={}):
    return inputs["x"] + inputs["y"] * inputs["z"]


def scaled_structured_function(scale, *, inputs={}):
    return scale * (inputs["x"] + inputs["y"] * inputs["z"])


def mix_arg_and_kwarg_with_scalar_broadcasting(aaa, bbb, *, ccc=None, ddd=None):
    return (aaa + bbb) ** ccc - ddd


def test_map_partitions_args_and_kwargs_have_collection():
    xc = ak.Array([[1, 2, 3], [4, 5], [6, 7, 8]])
    yc = ak.Array([0, 1, 2])
    xl = dak.from_awkward(xc, npartitions=3)
    yl = dak.from_awkward(yc, npartitions=3)

    zc = my_power(xc, kwarg_y=yc)
    zl = dak.map_partitions(my_power, xl, kwarg_y=yl)

    assert_eq(zc, zl)

    zd = structured_function(inputs={"x": xc, "y": xc, "z": yc})
    zm = dak.map_partitions(structured_function, inputs={"x": xl, "y": xl, "z": yl})

    assert_eq(zd, zm)

    ze = scaled_structured_function(2.0, inputs={"x": xc, "y": xc, "z": yc})
    zn = dak.map_partitions(
        scaled_structured_function, 2.0, inputs={"x": xl, "y": xl, "z": yl}
    )

    assert_eq(ze, zn)

    zf = scaled_structured_function(2.0, inputs={"x": xc, "y": xc, "z": 4.0})
    zo = dak.map_partitions(
        scaled_structured_function, 2.0, inputs={"x": xl, "y": xl, "z": 4.0}
    )

    assert_eq(zf, zo)

    zg = my_power(xc, kwarg_y=2.0)
    zp = dak.map_partitions(my_power, xl, kwarg_y=2.0)

    assert_eq(zg, zp)

    a = ak.Array(
        [
            [
                1,
                2,
                3,
            ],
            [4, 5],
            [6, 7, 8],
        ]
    )
    b = ak.Array([[-10, -10, -10], [-10, -10], [-10, -10, -10]])
    c = ak.Array([0, 1, 2])
    d = 1

    aa = dak.from_awkward(a, npartitions=2)
    bb = dak.from_awkward(b, npartitions=2)
    cc = dak.from_awkward(c, npartitions=2)
    dd = d

    res1 = mix_arg_and_kwarg_with_scalar_broadcasting(a, b, ccc=c, ddd=d)
    res2 = dak.map_partitions(
        mix_arg_and_kwarg_with_scalar_broadcasting,
        aa,
        bb,
        ccc=cc,
        ddd=dd,
    )
    assert_eq(res1, res2)


def test_dask_array_in_map_partitions(daa, caa):
    x1 = dak.zeros_like(daa.points.x)
    y1 = da.ones(len(x1), chunks=x1.divisions[1])
    z1 = x1 + y1
    x2 = ak.zeros_like(caa.points.x)
    y2 = np.ones(len(x2))
    z2 = x2 + y2
    assert_eq(z1, z2)


def test_dask_awkward_Array_copy(daa):
    c = copy.copy(daa)
    assert_eq(daa, c)


def test_map_partitions_no_dask_collections_passed(caa):
    with pytest.raises(
        TypeError,
        match="map_partitions expects at least one Dask collection",
    ):
        dak.num(caa.points.x, axis=1)


@pytest.mark.parametrize("fn", [dak.count, dak.zeros_like, dak.ones_like])
def test_shape_only_ops(fn: Callable, tmp_path_factory: pytest.TempPathFactory) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    a = ak.Array([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    p = tmp_path_factory.mktemp("zeros-like-flat")
    ak.to_parquet(a, str(p / "file.parquet"))
    lazy = dak.from_parquet(str(p))
    result = fn(lazy.b)
    with dask.config.set({"awkward.optimization.enabled": True}):
        result.compute()


def test_assign_behavior() -> None:
    behavior = {"test": "hello"}
    x = ak.Array([{"a": 1, "b": 2}, {"a": 3, "b": 4}], behavior=behavior, attrs={})
    dx = dak.from_awkward(x, 3)
    with pytest.raises(
        TypeError, match="'mappingproxy' object does not support item assignment"
    ):
        dx.behavior["should_fail"] = None
    assert dx.behavior == behavior


def test_assign_attrs() -> None:
    attrs = {"test": "hello"}
    x = ak.Array([{"a": 1, "b": 2}, {"a": 3, "b": 4}], behavior={}, attrs=attrs)
    dx = dak.from_awkward(x, 3)
    with pytest.raises(
        TypeError, match="'mappingproxy' object does not support item assignment"
    ):
        dx.attrs["should_fail"] = None
    assert dx.attrs == attrs


@delayed
def a_delayed_array():
    return ak.Array([2, 4])


def test_partitionwise_op_with_delayed():
    array = ak.Array([[1, 2, 3], [4], [5, 6, 7], [8]])
    dak_array = dak.from_awkward(array, npartitions=2)
    result = map_partitions(
        operator.mul,
        dak_array,
        a_delayed_array(),
        meta=dak_array._meta,
        output_divisions=1,
    )
    concrete_result = ak.concatenate(
        [
            array[:2] * a_delayed_array().compute(),
            array[2:] * a_delayed_array().compute(),
        ],
    )
    assert_eq(result, concrete_result)

    result = map_partitions(
        operator.mul,
        a_delayed_array(),
        dak_array,
        meta=dak_array._meta,
    )
    assert_eq(result, concrete_result)


def multiply(a, b, c):
    return a * b * c


def test_map_partitions_bad_arguments():
    array1 = ak.Array([[1, 2, 3], [4], [5, 6, 7], [8]])
    array2 = ak.Array([4, 5, 6, 7])
    with pytest.raises(TypeError, match="at least one"):
        map_partitions(
            multiply,
            a_delayed_array(),
            array1,
            array2,
            meta=empty_typetracer(),
        )


def test_array__bool_nonzero_long_int_float_complex_index():
    import operator

    dask_arr = dak.from_awkward(ak.Array([1]), npartitions=1)

    for fun in bool, int, float, complex, operator.index:
        with pytest.raises(
            ConcretizationTypeError,
            match=r"A dask_awkward.Array is encountered in a computation where a concrete value is expected. If you intend to convert the dask_awkward.Array to a concrete value, use the `.compute\(\)` method. The .+ method was called on .+.",
        ):
            fun(dask_arr)


def test_map_partitions_deterministic_token():
    dask_arr = dak.from_awkward(ak.Array([1]), npartitions=1)

    def f(x):
        return x[0] + 1

    assert (
        map_partitions(f, {0: dask_arr}).name == map_partitions(f, {0: dask_arr}).name
    )
