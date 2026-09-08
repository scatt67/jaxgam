"""Host-side execution adapters.

Execution code coordinates compiled fitting kernels but does not construct
formula bases or perform post-estimation work.
"""

from jaxgam.execution.stream import StreamPIRLSControl, fit_streamed_pirls

__all__ = ["StreamPIRLSControl", "fit_streamed_pirls"]
