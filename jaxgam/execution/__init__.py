"""Host-side execution adapters.

Execution code coordinates compiled fitting kernels but does not construct
formula bases or perform post-estimation work.

Layout: one sub-package per route (``stream``, ``discrete``, ``operator``,
``gaussian_qr``). The modules at this level are the Fellner-Schall loop that
the dense and streamed routes share (``efs``, ``efs_provider``) and the
reductions several routes use (``qr``, ``null_coefficient``).
"""

from jaxgam.execution.stream.pirls import StreamPIRLSControl, fit_streamed_pirls

__all__ = ["StreamPIRLSControl", "fit_streamed_pirls"]
