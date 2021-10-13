import awkward as ak  # noqa

import dask_awkward as dak  # noqa
import dask_awkward.core as dakc  # noqa
import dask_awkward.data as dakd

daa = dak.from_json(dakd.json_data(kind="numbers"))
aa = daa.compute()
a0 = daa.partitions[0].compute()
a1 = daa.partitions[1].compute()
a2 = daa.partitions[2].compute()
