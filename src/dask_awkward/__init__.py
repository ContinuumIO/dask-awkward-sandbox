from dask_awkward import config  # isort:skip; load awkward config

import dask_awkward.lib.core as core
import dask_awkward.lib.describe as describe
import dask_awkward.lib.inspect as inspect
import dask_awkward.lib.operations as operations
import dask_awkward.lib.optimize as optimize
import dask_awkward.lib.reducers as reducers
import dask_awkward.lib.str as str
import dask_awkward.lib.structure as structure
import dask_awkward.lib.utils as utils
from dask_awkward.lib.core import Array, PartitionCompatibility, Record, Scalar
from dask_awkward.lib.core import _type as type
from dask_awkward.lib.core import (
    compatible_partitions,
    dask_method,
    dask_property,
    map_partitions,
    partition_compatibility,
)
from dask_awkward.lib.describe import backend, fields
from dask_awkward.lib.inspect import (
    report_necessary_buffers,
    report_necessary_columns,
    sample,
)

necessary_columns = report_necessary_columns  # Export for backwards compatibility.

import dask_awkward.manual as manual
from dask_awkward.lib.io.io import (
    from_awkward,
    from_dask_array,
    from_delayed,
    from_lists,
    from_map,
    to_dask_array,
    to_dask_bag,
    to_dataframe,
    to_delayed,
)
from dask_awkward.lib.io.json import from_json, layout_to_jsonschema, to_json
from dask_awkward.lib.io.parquet import from_parquet, to_parquet
from dask_awkward.lib.io.text import from_text
from dask_awkward.lib.operations import concatenate
from dask_awkward.lib.reducers import (
    all,
    any,
    argmax,
    argmin,
    corr,
    count,
    count_nonzero,
    covar,
    linear_fit,
    max,
    mean,
    min,
    moment,
    prod,
    ptp,
    softmax,
    std,
    sum,
    var,
)
from dask_awkward.lib.structure import (
    argcartesian,
    argcombinations,
    argsort,
    broadcast_arrays,
    cartesian,
    combinations,
    copy,
    drop_none,
    fill_none,
    firsts,
    flatten,
    from_regular,
    full_like,
    is_none,
    isclose,
    local_index,
    mask,
    nan_to_num,
    num,
    ones_like,
    pad_none,
    ravel,
    run_lengths,
    singletons,
    sort,
    strings_astype,
    to_list,
    to_packed,
    to_regular,
    unflatten,
    unzip,
    values_astype,
    where,
    with_field,
    with_name,
    with_parameter,
    without_field,
    without_parameters,
    zeros_like,
    zip,
)
from dask_awkward.version import __version__
