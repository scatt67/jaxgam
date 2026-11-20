"""Host drivers for the discretized route (``FitControl(execution="discrete")``).

The analogue of ``bam(discrete=TRUE)``. ``gaussian`` holds the shared setup
and the additive Gaussian fit; ``regular`` fits the other families by the
``bgam.fitd`` performance iteration. ``working_model`` and
``regular_working_model`` control the working-model REML steps, and
``working_theta`` ports ``estimate.theta``. The JAX kernels they call are in
``jaxgam.fitting.discrete``.
"""
