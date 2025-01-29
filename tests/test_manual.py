from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

import dask_awkward as dak


def test_optimize_columns():
    pytest.importorskip("pyarrow")
    pytest.importorskip("requests")
    pytest.importorskip("aiohttp")

    array = dak.from_parquet(
        "https://github.com/scikit-hep/awkward/raw/main/tests/samples/nullable-record-primitives-simple.parquet"
    )

    needs = dak.inspect.report_necessary_columns(array.u4)
    only_u4_array = dak.manual.optimize_columns(array, needs)

    assert only_u4_array.fields == ["u4", "u8"]

    materialized_only_u4_array = only_u4_array.compute()

    # u4 is materialized, u8 is not
    assert isinstance(
        materialized_only_u4_array.layout.content("u4").content.data, np.ndarray
    )
    assert isinstance(
        materialized_only_u4_array.layout.content("u8").content.data,
        ak._nplikes.placeholder.PlaceholderArray,
    )

    # now again, but we add 'u8' by hand to the columns
    key, cols = needs.popitem()
    cols |= {"u8"}

    needs = {key: cols}

    u4_and_u8_array = dak.manual.optimize_columns(array, needs)

    assert u4_and_u8_array.fields == ["u4", "u8"]

    materialized_u4_and_u8_array = u4_and_u8_array.compute()

    # now u4 and u8 are materialized
    assert isinstance(
        materialized_u4_and_u8_array.layout.content("u4").content.data, np.ndarray
    )
    assert isinstance(
        materialized_u4_and_u8_array.layout.content("u8").content.data, np.ndarray
    )
