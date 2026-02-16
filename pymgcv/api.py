"""Top-level fitting orchestration for pymgcv."""


def gam(formula, data, family="gaussian", method="REML", **kwargs):
    """Fit a Generalized Additive Model.

    Parameters
    ----------
    formula : str
        Model formula in R-style Wilkinson notation.
    data : pandas.DataFrame
        Data frame containing the variables in the formula.
    family : str or ExponentialFamily
        Distribution family. One of 'gaussian', 'binomial', 'poisson', 'gamma'.
    method : str
        Smoothing parameter estimation method. One of 'REML', 'ML'.
    **kwargs
        Additional arguments passed to the fitting engine.

    Returns
    -------
    GAMResult
        Fitted model object.
    """
    raise NotImplementedError(
        "gam() is not yet implemented. See IMPLEMENTATION_PLAN.md."
    )


def predict(model, newdata=None, type="response", se_fit=False):
    """Predict from a fitted GAM.

    Parameters
    ----------
    model : GAMResult
        Fitted model from gam().
    newdata : pandas.DataFrame, optional
        New data for prediction. If None, uses the training data.
    type : str
        Type of prediction: 'response' or 'link'.
    se_fit : bool
        Whether to return standard errors.

    Returns
    -------
    numpy.ndarray or tuple
        Predictions, or (predictions, standard_errors) if se_fit=True.
    """
    raise NotImplementedError(
        "predict() is not yet implemented. See IMPLEMENTATION_PLAN.md."
    )


def summary(model):
    """Print summary of a fitted GAM.

    Parameters
    ----------
    model : GAMResult
        Fitted model from gam().
    """
    raise NotImplementedError(
        "summary() is not yet implemented. See IMPLEMENTATION_PLAN.md."
    )


def plot(model, select=None, pages=0, rug=True, se=True, shade=True):
    """Plot smooth components of a fitted GAM.

    Parameters
    ----------
    model : GAMResult
        Fitted model from gam().
    select : int or list, optional
        Which smooth terms to plot.
    pages : int
        Number of pages for multi-panel plots.
    rug : bool
        Whether to add rug plots.
    se : bool
        Whether to plot standard error bands.
    shade : bool
        Whether to shade standard error bands.
    """
    raise NotImplementedError(
        "plot() is not yet implemented. See IMPLEMENTATION_PLAN.md."
    )
