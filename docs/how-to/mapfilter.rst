mapfilter
---------

:func:`dask_awkward.mapfilter` is a function that applies a function to each partition of
dask-awkward collections (:class:`dask_awkward.Array`). It maps the given function
over each partition in the provided collections in an embarrassingly parallel way. The input collections
must have the same number of partitions.

An example is shown below:

.. code-block:: python

    import dask_awkward as dak
    import awkward as ak

    # Create a dask-awkward array
    x = ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dak_array = dak.from_awkward(x, npartitions=2)


    # Define a function to apply to each partition
    @dak.mapfilter
    def add_one(array):
        return array + 1


    # Apply the function to each partition
    result = add_one(dak_array)

    # Compute the result
    result.compute()
    # <Array [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] type='10 * int64'>


Here, the ``dak_array`` has two partitions, and :func:`dask_awkward.mapfilter` will
apply the ``add_one`` function to each partition in parallel - resulting in two tasks in total (for the low-level graph).

.. warning::

    Since the mapped function is applied to each partition, the function must use eager awkward-array operations.
    It is not possible to use (lazy) dask-awkward operations inside.


Collapsing Lazy Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^

The main purpose of :func:`dask_awkward.mapfilter` is to merge nodes into a single node
in the highlevel dask graph. This is useful to keep the graph small and avoid unnecessary scheduling overhead.

*Any* function that is given to :func:`dask_awkward.mapfilter` will become a *single* node in the highlevel dask graph,
no matter how many operations are performed inside.

An example is given in the following:

.. code-block:: python

    import dask_awkward as dak
    import awkward as ak

    # Create a dask-awkward array
    x = ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dak_array = dak.from_awkward(x, npartitions=2)


    # Define a function to apply to each partition
    @dak.mapfilter
    def fun(array):
        return np.sin(array**2 + 1)


    # Apply the function to each partition
    result = fun(dak_array)

    # Inspect the graph
    result.dask
    # HighLevelGraph with 2 layers.
    # <dask.highlevelgraph.HighLevelGraph object at 0x104e68c40>
    # 0. from-awkward-25967e11ca4677388b80cfb6f556d752
    # 1. <dask-awkward.lib.core.ArgsKwargsPackedFunction ob-66ae0a4a59e17e64e96c9b1ee8c18f51

Here, one can see that the graph consists of 2 layers:

1. The first layer is the creation of the dask-awkward array (``dak.from_awkward``).
2. The second layer is the application of the function to each partition (``dak.mapfilter(fun)``).

In contrast, *without* using :func:`dask_awkward.mapfilter`, the graph would consist of 4 layers:

1. The first layer is the creation of the dask-awkward array (``dak.from_awkward``).
2. Power of 2 (``**``) operation.
3. Addition of 1 (``+ 1``) operation.
4. Sine (``np.sin``) operation.

:func:`dask_awkward.mapfilter` merges operations 2-4 into a single node in the highlevel dask graph.


Multiple Return Values
^^^^^^^^^^^^^^^^^^^^^^

:func:`dask_awkward.mapfilter` allows to return multiple values from the mapped function. This is useful if one wants to return
multiple arrays or even metadata from the function. The return values must be provided as a tuple, :func:`dask_awkward.mapfilter` will not
recurse into the return values.

Any returned :class:`awkward.Array` will be automatically converted to a :class:`dask_awkward.Array` collection.
Any other type will be wrapped by a :class:`dask.bag.Bag` collection.

An example is given in the following:

.. code-block:: python

    import dask_awkward as dak
    import awkward as ak

    # Create a dask-awkward array
    x = ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dak_array = dak.from_awkward(x, npartitions=2)


    class some: ...


    # Define a function to apply to each partition
    @dak.mapfilter
    def fun(array):
        return array + 1, array * 2, some()


    # Apply the function to each partition
    a, b, c = fun(dak_array)

    # Compute the result
    a.compute()
    # <Array [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] type='10 * int64'>
    b.compute()
    # <Array [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] type='10 * int64'>
    c.compute()
    # (<__main__.some at 0x10b5819c0>, <__main__.some at 0x10b580dc0>)


Untraceable Functions
^^^^^^^^^^^^^^^^^^^^^

Sometimes one needs to leave the awkward-array world and use some operations that are not traceable
by awkward's typetracer. In this case :func:`dask_awkward.mapfilter` can be used to apply the function
to each partition nevertheless. One needs to provide the ``meta`` and ``needs`` arguments to :func:`dask_awkward.mapfilter`
to enable this:

* ``meta``: The meta information of the output values
* ``needs``: A mapping that specifies an iterable of columns mapped to :class:`dask_awkward.Array` input arguments

An example is given in the following:

.. code-block:: python

    ak_array = ak.zip(
        {
            "x": ak.zip({"foo": [10, 20, 30, 40], "bar": [10, 20, 30, 40]}),
        }
    )
    dak_array = dak.from_awkward(ak_array, 2)


    def untraceable_fun(array):
        foo = ak.to_numpy(array.x.foo)
        return ak.Array([np.sum(foo)])


    dak.mapfilter(untraceable_fun)(dak_array)
    # ...
    # TypeError: Converting from an nplike without known data to an nplike with known data is not supported
    #
    # This error occurred while calling
    #
    #    ak.to_numpy(
    #        <Array-typetracer [...] type='## * int64'>
    #    )
    #
    # The above exception was the direct cause of the following exception:
    # ...

    # Now let's add `meta` and `needs` arguments
    from functools import partial

    mapf = partial(dak.mapfilter, needs={"array": [("x", "foo")]}, meta=ak.Array([0, 0]))

    # It works now!
    mapf(untraceable_fun)(dak_array).compute()
    # <Array [30, 70] type='2 * int64'>

In fact, providing ``meta`` and ``needs`` is entirely skipping the tracing step as both arguments provide all necessary information already.
In cases where the function is much more complex and not traceable it can be helpful to run the tracing step manually:

.. code-block:: python

    meta, needs = dak.prerun(untraceable_fun, array=dak_array)
    # ...
    # UntraceableFunctionError: '<function untraceable_fun at 0x10536d240>' is not traceable, an error occurred at line 9. 'dak.mapfilter' can circumvent this by providing 'needs' and 'meta' arguments to it.
    #
    # - 'needs': mapping where the keys point to input argument dask_awkward arrays and the values to columns that should be touched explicitly. The typetracing step could determine the following necessary columns until the exception occurred:
    #
    # needs={'array': [('x', 'foo')]}
    #
    # - 'meta': value(s) of what the wrapped function would return. For arrays, only the shape and type matter.

Here, :func:`dask_awkward.prerun` will try to trace the function once and return the necessary information (``meta`` and ``needs``) to provide to :func:`dask_awkward.mapfilter`.
In this case the function is untraceable, so :func:`dask_awkward.prerun` will report at least ``needs`` to the point where the function is not traceable anymore.

.. tip::

    For traceable but long-running functions (e.g. if the contain the evaluation of a neural network), it is recommended to use :func:`dask_awkward.prerun` to infer ``meta`` and ``needs`` once,
    and provide it to all consecutive :func:`dask_awkward.mapfilter` calls. This way, the tracing step is only performed once.
