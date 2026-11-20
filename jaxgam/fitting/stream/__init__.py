"""JAX kernels for the streamed route.

Per-batch reductions for the coefficient fits (``kernels``, ``regular``,
``nb_kernels``, ``nb_theta``), coefficient-space derivatives of the exact
REML score (``reml``, ``nb_reml``) and the Fellner-Schall contractions
(``efs_factor``). The host drivers are in ``jaxgam.execution.stream``.
"""
