"""Host drivers for the streamed route (``FitControl(execution="stream")``).

Coefficient fits from a replayable row source at fixed smoothing parameters
(``pirls``, and the source-ordered ``regular`` and ``nb`` fits), and the
drivers that select smoothing parameters around them: exact REML (``reml``,
``nb_reml``) and Fellner-Schall (``efs``, ``efs_provider``, ``efs_start``).
The JAX kernels they call are in ``jaxgam.fitting.stream``.
"""
