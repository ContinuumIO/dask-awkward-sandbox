.. dask-awkward documentation master file, created by
   sphinx-quickstart on Fri Sep 24 13:43:52 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dask-awkward
============

*Connecting Dask and Awkward Array.*

The dask-awkward project implements a native Dask collection for
representing partitioned Awkward arrays. Read more about Awkward
arrays at `their website <awkwardarray_>`_.


Table of Contents
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting-started/install.rst
   getting-started/intro.rst
   getting-started/terminology.rst
   getting-started/limitations.rst

.. toctree::
   :maxdepth: 1
   :caption: How Tos

   how-to/configuration.rst
   how-to/io.rst
   how-to/behaviors.rst
   how-to/mapfilter.rst

.. toctree::
   :maxdepth: 1
   :caption: Deeper Explanation

   more/optimization.rst
   more/faq.rst

.. toctree::
   :maxdepth: 1
   :caption: API

   api/collections.rst
   api/inspect.rst
   api/io.rst
   api/mapfilter.rst
   api/reducers.rst
   api/structure.rst
   api/behavior.rst
   api/utils.rst

.. toctree::
   :maxdepth: 1
   :caption: Development

   dev/contributing.rst
   dev/releasing.rst

------------

Support for this work was provided by NSF grant `OAC-2103945 <nsf_>`_.

.. _awkwardarray: https://awkward-array.org/
.. _nsf: https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945


.. raw:: html

   <script data-goatcounter="https://dask-awkward.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
