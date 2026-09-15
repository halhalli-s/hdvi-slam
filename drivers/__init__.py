"""SachiSLAM drivers: the only place a sensor SDK is imported.

Drivers repackage raw hardware output into the ``core.types`` data contract
(``CloudFrame`` / ``ImuSample``) so the rest of the system never sees a
vendor SDK type.
"""
