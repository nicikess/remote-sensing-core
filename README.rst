.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/remote-sensing-core.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/remote-sensing-core
    .. image:: https://readthedocs.org/projects/remote-sensing-core/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://remote-sensing-core.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/remote-sensing-core/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/remote-sensing-core
    .. image:: https://img.shields.io/pypi/v/remote-sensing-core.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/remote-sensing-core/
    .. image:: https://img.shields.io/conda/vn/conda-forge/remote-sensing-core.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/remote-sensing-core
    .. image:: https://pepy.tech/badge/remote-sensing-core/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/remote-sensing-core
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/remote-sensing-core

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===================
remote-sensing-core
===================


    Project focused on pytorch integrations for BEN-GE data set


+----------------------+----------------+----------------+-------------------+--------------------+-------------------------+
| Modality             + Index          + Channel        + Pixel-Resolution  + Spatial Resolution | Value Range (min, max)  |
+----------------------+----------------+----------------+-------------------+--------------------+-------------------------+
|  Sentinel 1          | 0              | VV             | 120x120           | 10m/px             |                         |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 1              | VH             | 120x120           | 10m/px             |                         |
+----------------------+----------------+----------------+-------------------+--------------------+-------------------------+
| Sentinel 2           | 0              | Aerosol        | 120x120           | 10m/px (60m)       | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 1              | Blue           | 120x120           | 10m/px             | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 2              | Green          | 120x120           | 10m/px             | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 3              | Red            | 120x120           | 10m/px             | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 4              | Red Edge       | 120x120           | 10m/px             | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 5              | Red Edge 2     | 120x120           | 10m/px (20m)       | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 6              | Red Edge 3     | 120x120           | 10m/px (20m)       | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 7              | NIR            | 120x120           | 10m/px             | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 8              | Narrow NIR     | 120x120           | 10m/px (20m)       | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 9              | Water Vapor    | 120x120           | 10m/px (60m)       | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 10             | SWIR           | 120x120           | 10m/px (20m)       | (0, 10.000)             |
+                      +----------------+----------------+-------------------+--------------------+-------------------------+
|                      | 11             | SWIR           | 120x120           | 10m/px (20m)       | (0, 10.000)             |
+----------------------+----------------+----------------+-------------------+--------------------+-------------------------+
| Altitude Model       | 0              |                |                   |                    |                         |
+----------------------+----------------+----------------+-------------------+--------------------+-------------------------+
| ESA World-Cover      | 0              | Land Use       | 120x120           | 10m/px             | (0, 100)                |
+----------------------+----------------+----------------+-------------------+--------------------+-------------------------+


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
