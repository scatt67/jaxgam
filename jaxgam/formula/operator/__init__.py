"""Phase 1 setup for the operator route (NumPy only).

``prepare`` builds the prepared operator model from a row source. ``modest``
and ``radial`` set up its small dense block (``radial`` the thin plate and
Gaussian process smooths), and ``workspace`` bounds the memory of compact
prediction.
"""
