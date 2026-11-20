"""Phase 1 setup for the discretized route (NumPy only).

``continuous_setup`` builds the bases on the compact data frame, ``design``
the exact indexed designs and ``continuous_design`` their descriptors from
the compact indices. ``continuous_metadata`` is the row-free record of a
fitted discretization that a result keeps, and ``working_penalties`` holds
the penalty transforms of the working-model REML step.
"""
