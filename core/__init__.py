"""SachiSLAM core: sensor-agnostic SLAM pipeline components.

Nothing in this package imports a sensor SDK (pyorbbecsdk) or a middleware
(ROS). Sensor I/O lives in ``drivers/`` and visualization in ``viz/``. Core
speaks only in the data contract defined by ``core.types``.
"""
