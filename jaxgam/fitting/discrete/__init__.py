"""JAX kernels for the discretized route.

``ops`` applies the compact indexed design, ``working`` computes the
per-batch ``bgam.fitd`` working statistics, and ``working_model`` makes one
working-model REML proposal. The host drivers are in
``jaxgam.execution.discrete``.
"""
