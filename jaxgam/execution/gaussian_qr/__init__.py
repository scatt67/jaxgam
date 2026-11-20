"""Host drivers for Gaussian QR compression (``gaussian_compression=True``).

``preparation`` reduces a Gaussian identity-link model to frozen QR
statistics, from a dense design or from a row source; ``optimizer`` selects
the smoothing parameters on those statistics. The score kernel is in
``jaxgam.fitting.gaussian_qr``.
"""
