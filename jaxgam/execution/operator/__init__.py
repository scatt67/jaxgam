"""Host drivers for the operator route (``FitControl(execution="operator")``).

Matrix-free fitting of models with indexed random effects. ``system`` holds
the frozen scan system and its conjugate-gradient solve; ``fit`` is the
coefficient fit at fixed smoothing parameters. The ``efs_*`` and ``score*``
modules, with ``logdet``, ``penalty_determinant`` and ``arrowhead``, make up
Fellner-Schall smoothing selection and its score error bound. ``variance``
and ``bulk_variance`` give the approximate Fisher variances. The JAX kernels
they call are in ``jaxgam.fitting.operator``.
"""
