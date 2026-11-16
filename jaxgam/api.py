"""Top-level fitting orchestration for jaxgam.

Provides the ``GAM`` class (sklearn-style API) that wires together:
- Phase 1: ``parse_formula()`` → ``ModelSetup.build()``
- Phase 2: ``FittingData.from_setup()`` → ``newton_optimize()`` / ``pirls_loop()``
- Phase 3: ``GAMResults._from_fit()`` materializes the selected result type

``GAM.fit()`` returns a full ``GAMResults`` by default or a lean
``GAMInferenceResult`` when requested.

Design doc reference: docs/refactor_gam_api/design.md §3.3
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from jaxgam.control import FitControl
from jaxgam.data.source import RowSource
from jaxgam.families.base import ExponentialFamily
from jaxgam.families.extended import ExtendedFamily
from jaxgam.families.registry import get_family
from jaxgam.fitting.data import FittingData, PreparedFittingMetadata
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.newton import NewtonResult, OuterScoreProvider, newton_optimize
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.fitting.reml import REMLCriterion
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.results import (
    GAMInferenceResult,
    GAMPredictionResult,
    GAMResults,
    StreamExactOptimizerDiagnostics,
)

if TYPE_CHECKING:
    import jax
    import pandas as pd


# ---------------------------------------------------------------------------
# GAM class
# ---------------------------------------------------------------------------


class GAM:
    """Generalized Additive Model specification.

    This class holds model specification parameters and orchestrates the
    fit pipeline. Calling ``fit()`` returns a frozen full or lean result,
    selected by the keyword-only ``result`` argument.

    Parameters
    ----------
    formula : str
        Model formula in R-style Wilkinson notation, e.g. ``"y ~ s(x)"``.
    family : str or ExponentialFamily
        Distribution family. One of ``'gaussian'``, ``'binomial'``,
        ``'poisson'``, ``'gamma'``, or an ``ExponentialFamily`` instance.
    method : str
        Smoothing parameter estimation method. Only ``'REML'`` is supported
        in v1.0; ``'ML'`` raises ``NotImplementedError`` (see notes below).
    sp : np.ndarray or list, optional
        Fixed smoothing parameters. If provided, skips Newton optimization.
    device : str, optional
        Target device: ``'cpu'``, ``'gpu'``, or ``None`` (auto-detect).
        GPU requires ``jax[cuda12]`` (NVIDIA) or ``jax-metal`` (Apple).
    **kwargs
        Additional arguments. ``optimizer`` accepts ``"newton"`` (default)
        or the dense ``"efs"`` route. The remaining accepted scope-guard keys
        are ``device``, ``backend``, ``select``, ``gamma``, and ``knots``;
        other keywords raise ``TypeError``.

    Examples
    --------
    >>> model = GAM("y ~ s(x)", family="gaussian")
    >>> full = model.fit(data, result="full")  # default; model.fit(data) is equivalent
    >>> full.summary()
    >>> lean = model.fit(data, result="inference")
    >>> lean.predict(newdata)  # already lean; no to_predictor() call required
    array([...])

    Design doc reference: docs/refactor_gam_api/design.md §3.3
    """

    def __init__(
        self,
        formula: str,
        family: str | ExponentialFamily = "gaussian",
        method: str = "REML",
        sp: np.ndarray | list | None = None,
        control: FitControl | None = None,
        **kwargs,
    ) -> None:
        _check_scope_guards(method, kwargs)
        self.formula = formula
        self.family = family
        self.method = method.upper()
        self.sp = sp
        self.device = kwargs.get("device")
        self.optimizer = kwargs.get("optimizer") or "newton"
        if control is not None and not isinstance(control, FitControl):
            raise TypeError("control must be a FitControl instance or None.")
        self.control = FitControl() if control is None else control

    @overload
    def fit(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["full"] = "full",
    ) -> GAMResults: ...

    @overload
    def fit(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["inference"],
    ) -> GAMInferenceResult: ...

    @overload
    def fit(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["prediction"],
    ) -> GAMPredictionResult: ...

    def fit(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["full", "inference", "prediction"] = "full",
    ) -> GAMResults | GAMInferenceResult | GAMPredictionResult:
        """Fit the GAM to data.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Data frame containing the variables in the formula. A replayable
            ``RowSource`` is accepted with ``execution='stream'``,
            ``execution='operator'`` or the explicitly configured
            ``execution='discrete'`` route. The operator route requires
            ``linear_solver='pcg'`` and prediction-only retention. It accepts
            explicit fixed ``sp`` or calibrated indexed Poisson EFS.
        weights : np.ndarray, optional
            Prior weights, shape ``(n,)``.
        offset : np.ndarray, optional
            Offset vector, shape ``(n,)``.
        result : {"full", "inference", "prediction"}
            Result materialization mode. ``"full"`` retains training-backed
            diagnostics; ``"inference"`` returns a lean result that is already
            directly usable for new-data prediction. No ``to_predictor()`` call
            is required unless a narrower prediction-only handoff is desired.

        Returns
        -------
        GAMResults or GAMInferenceResult
            Frozen full-diagnostic or lean-inference result. Streamed fitting
            returns ``result='prediction'`` and supports fixed smoothing
            parameters across the family execution contracts or EFS with QR.
            Discrete fitting accepts fixed ``sp`` or
            ``smoothing_strategy='working_model_reml'`` and also returns
            ``result='prediction'``.

        Design doc reference: docs/refactor_gam_api/design.md §3.3
        """
        if result not in ("full", "inference", "prediction"):
            raise ValueError(
                f"result must be 'full', 'inference', or 'prediction', got {result!r}"
            )

        family_obj = get_family(self.family)

        # Extended families have mutable theta state that is synced by
        # _build_result.put_theta after fitting.  Copy the instance so
        # the registry singleton is never mutated.
        if hasattr(family_obj, "n_theta") and family_obj.n_theta > 0:
            import copy

            family_obj = copy.deepcopy(family_obj)

        if self.control.execution == "operator":
            return self._fit_operator(
                data, weights, offset, result=result, family=family_obj
            )
        if self.control.execution == "stream":
            return self._fit_stream(
                data, weights, offset, result=result, family=family_obj
            )
        if self.control.execution == "discrete":
            return self._fit_discrete(
                data, weights, offset, result=result, family=family_obj
            )
        if isinstance(data, RowSource):
            raise TypeError(
                "A RowSource requires FitControl execution 'operator', 'stream' or "
                "'discrete'; the dense route accepts only a DataFrame or dict."
            )
        if self.control.linear_solver != "cholesky":
            route = "operator" if self.control.linear_solver == "pcg" else "stream"
            raise NotImplementedError(
                f"linear_solver={self.control.linear_solver!r} currently requires "
                f"execution={route!r}."
            )

        # Phase 1: parse + build model setup
        spec = parse_formula(self.formula)
        setup = ModelSetup.build(spec, data, weights, offset)

        # Phase 1→2: transfer to JAX device
        jax_device = _resolve_device(self.device)
        fd = FittingData.from_setup(setup, family_obj, device=jax_device)

        score_provider: OuterScoreProvider | None = None
        execution_route = "dense"
        execution_fallback_reason = None
        if self.control.gaussian_compression:
            if self.optimizer == "efs" and self.sp is None:
                execution_fallback_reason = (
                    "Gaussian QR compression requires Newton smoothing selection."
                )
            else:
                from jaxgam.execution.gaussian_optimizer import (
                    GaussianQRIneligible,
                    build_dense_gaussian_qr_provider,
                )

                try:
                    score_provider = build_dense_gaussian_qr_provider(fd)
                except GaussianQRIneligible as error:
                    execution_fallback_reason = str(error)
                else:
                    execution_route = "gaussian_qr"

        # Phase 2: fit. Fixed smoothing and parametric models retain their
        # established dispatch before estimated smoothing selection.
        if self.sp is not None:
            fit_result = _fit_fixed_sp(
                fd, self.sp, self.method, score_provider=score_provider
            )
            lambda_strategy = "fixed"
        elif fd.n_penalties == 0:
            fit_result = newton_optimize(fd, self.method)
            lambda_strategy = f"newton_{self.method.lower()}"
        elif self.optimizer == "efs":
            from jaxgam.execution.efs import (
                _known_scale_family_supported,
                _unknown_scale_family_supported,
                dense_efs_known_scale,
                dense_efs_unknown_scale,
                efs_initial_log_lambda,
                efs_initial_log_scale,
            )
            from jaxgam.families.negative_binomial import NegativeBinomial
            from jaxgam.links.links import IdentityLink, SqrtLink

            if (
                isinstance(family_obj, NegativeBinomial)
                and family_obj.n_theta > 0
                and isinstance(family_obj.link, (IdentityLink, SqrtLink))
            ):
                null_eta = (
                    np.zeros(fd.n_obs, dtype=np.float64)
                    if fd.offset is None
                    else np.asarray(fd.offset, dtype=np.float64)
                )
                null_mu = np.asarray(family_obj.link.inverse(null_eta))
                if not (
                    np.all(family_obj.valid_eta(null_eta))
                    and np.all(family_obj.valid_mu(null_mu))
                ):
                    raise ValueError(
                        "Estimated NB identity/sqrt EFS has no valid default "
                        "zero-coefficient recovery anchor for this offset; pinned "
                        "mgcv rejects the same input. Supply an offset whose "
                        "zero-coefficient means are positive."
                    )

            if family_obj.scale_known and not _known_scale_family_supported(fd):
                raise NotImplementedError(
                    "optimizer='efs' does not support this known-scale family/link"
                )
            if not family_obj.scale_known and not _unknown_scale_family_supported(fd):
                raise NotImplementedError(
                    "optimizer='efs' does not support this unknown-scale family/link"
                )
            initial_log_lambda = efs_initial_log_lambda(setup, family_obj)
            if family_obj.scale_known:
                fit_result = dense_efs_known_scale(
                    fd,
                    initial_log_lambda=initial_log_lambda,
                    control=self.control.efs,
                )
            else:
                fit_result = dense_efs_unknown_scale(
                    fd,
                    initial_log_lambda=initial_log_lambda,
                    initial_log_scale=efs_initial_log_scale(setup, family_obj),
                    control=self.control.efs,
                )
            lambda_strategy = "efs_reml"
        else:
            use_lbfgs = False
            if score_provider is not None:
                from jaxgam.execution.gaussian_optimizer import (
                    optimize_gaussian_qr_lbfgs,
                    use_gaussian_qr_lbfgs,
                )

                use_lbfgs = use_gaussian_qr_lbfgs(fd.n_penalties)
            if use_lbfgs:
                fit_result = optimize_gaussian_qr_lbfgs(fd, score_provider)
                lambda_strategy = f"lbfgs_{self.method.lower()}"
            else:
                fit_result = newton_optimize(
                    fd, self.method, score_provider=score_provider
                )
                lambda_strategy = f"newton_{self.method.lower()}"

        if (
            self.optimizer == "efs"
            and self.sp is None
            and fd.n_penalties > 0
            and fit_result.theta is not None
            and family_obj.n_theta > 0
        ):
            family_obj.put_theta(np.log(np.asarray([fit_result.theta])))

        # Phase 2→3: construct the selected result materialization
        return GAMResults._from_fit(
            fit_result=fit_result,
            setup=setup,
            spec=spec,
            data=data,
            family=family_obj,
            fd=fd,
            lambda_strategy=lambda_strategy,
            formula=self.formula,
            method=self.method,
            result_mode=result,
            control=self.control,
            execution_route=execution_route,
            execution_fallback_reason=execution_fallback_reason,
            supplied_smoothing_params=self.sp,
        )

    def _fit_operator(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None,
        offset: np.ndarray | None,
        *,
        result: Literal["full", "inference", "prediction"],
        family: ExponentialFamily,
    ) -> GAMPredictionResult:
        """Run CPU operator fitting with compact prediction retention."""
        import numbers

        from jaxgam.execution.operator_fit import fit_operator_fixed_sp
        from jaxgam.formula.operator_prepare import (
            OperatorPreparationControl,
            prepare_operator_model,
        )

        if self.control.linear_solver != "pcg":
            raise ValueError("execution='operator' requires linear_solver='pcg'.")
        if result != "prediction":
            raise NotImplementedError(
                "Operator full/inference results need planned score/EDF/variance "
                "providers; this route currently requires result='prediction'."
            )
        if self.control.uncertainty != "none":
            raise NotImplementedError(
                "Operator uncertainty needs a planned selected-variance provider."
            )
        estimated_efs = self.sp is None and self.optimizer == "efs"
        if self.sp is None and not estimated_efs:
            raise NotImplementedError("Operator fitting requires explicit fixed sp.")
        snapshot = family.execution_parameter_snapshot()
        if snapshot.theta_mode == "estimated":
            raise NotImplementedError(
                "Operator fitting currently requires fixed theta."
            )
        if len(snapshot.log_theta) > 1:
            raise NotImplementedError("Operator result theta currently has one scalar.")
        if self.device not in (None, "cpu"):
            raise NotImplementedError("Frozen-system operator PCG currently uses CPU.")
        if not isinstance(data, RowSource):
            raise TypeError("Operator fitting requires a replayable RowSource.")
        if weights is not None or offset is not None:
            raise ValueError("Operator weights/offset must be owned by the RowSource.")
        if estimated_efs:
            return self._fit_operator_efs(data, family)
        spec = parse_formula(self.formula)
        expected = (
            len(spec.smooth_terms)
            if all(t.by is None and t.smooth_type == "s" for t in spec.smooth_terms)
            else None
        )
        values = [self.sp] if isinstance(self.sp, numbers.Real) else self.sp
        if expected is not None and len(values) != expected:
            raise ValueError(f"sp must have {expected} elements for this model.")
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Real)
            for value in values
        ):
            raise ValueError("Operator sp must be a one-dimensional real vector.")
        if 16 * len(values) > self.control.memory_budget_bytes:
            raise MemoryError("Operator sp ownership exceeds its memory budget.")
        sp = np.asarray(values, dtype=np.float64)
        if np.any(~np.isfinite(sp) | (sp <= 0)):
            raise ValueError("Operator sp must contain positive finite values.")
        model = prepare_operator_model(
            spec,
            data,
            control=OperatorPreparationControl(
                self.control.batch_rows,
                self.control.memory_budget_bytes,
                self.control.operator.max_modest_coefficients,
            ),
        )
        expected = model.penalties.n_penalties
        if len(sp) != expected:
            raise ValueError(f"sp must have {expected} elements for this model.")
        fitted = fit_operator_fixed_sp(
            model,
            data,
            family,
            sp,
            control=self.control.operator,
            memory_budget_bytes=self.control.memory_budget_bytes,
        )
        return GAMPredictionResult._from_operator_fit(
            fitted=fitted,
            sp=sp,
            formula=self.formula,
            method=self.method,
            control=self.control,
        )

    def _fit_operator_efs(
        self, source: RowSource, family: ExponentialFamily
    ) -> GAMPredictionResult:
        """Materialize calibrated indexed and modest-indexed Poisson EFS fits."""
        from jaxgam.execution.operator_efs_controller import (
            CALIBRATED_OPERATOR_EFS_DOMAIN,
            CALIBRATED_OPERATOR_EFS_MAX_MODEST,
            SourceOperatorEFSFitProvider,
            run_operator_efs_known_scale,
        )
        from jaxgam.execution.operator_efs_start import (
            prepare_indexed_operator_efs_start,
        )
        from jaxgam.execution.operator_logdet import OperatorLogDetControl
        from jaxgam.families.standard import Poisson
        from jaxgam.formula.operator_prepare import (
            OperatorPreparationControl,
            prepare_operator_model,
        )

        if not isinstance(family, Poisson) or not family.is_canonical:
            raise NotImplementedError(
                "Operator EFS currently supports calibrated Poisson/log indexed fits."
            )
        domain = CALIBRATED_OPERATOR_EFS_DOMAIN
        if not (
            domain.minimum_batch_rows
            <= self.control.batch_rows
            <= domain.maximum_batch_rows
        ):
            raise NotImplementedError(
                "Calibrated operator EFS requires batch_rows "
                f"{domain.minimum_batch_rows}-{domain.maximum_batch_rows}."
            )
        spec = parse_formula(self.formula)
        total = self.control.memory_budget_bytes
        preparation_budget = total // 8
        model = prepare_operator_model(
            spec,
            source,
            control=OperatorPreparationControl(
                self.control.batch_rows,
                preparation_budget,
                self.control.operator.max_modest_coefficients,
            ),
        )
        indexed = (
            model.random_effects[0].n_coef if len(model.random_effects) == 1 else 0
        )
        modest = len(model.modest_columns)
        if (
            not domain.admits(indexed, model.n_obs, self.control.batch_rows)
            or modest > CALIBRATED_OPERATOR_EFS_MAX_MODEST
            or model.n_coef != indexed + modest
        ):
            raise NotImplementedError(
                f"Calibrated operator EFS requires {domain.describe()}, and at "
                f"most {CALIBRATED_OPERATOR_EFS_MAX_MODEST} modest coefficients."
            )
        start = prepare_indexed_operator_efs_start(
            model, source, family, memory_budget_bytes=preparation_budget
        )
        # Final prediction geometry and coefficients coexist with the accepted
        # score/trace state during materialization. Keep that ownership outside
        # the controller's aggregate fit/score reservation.
        result_reserve = 2 * model.metadata_bound_bytes + 32 * model.n_coef + 8192 + 16
        if result_reserve + start.log_lambda.nbytes >= total:
            raise MemoryError("Operator EFS result ownership exceeds its budget.")
        provider = SourceOperatorEFSFitProvider(
            model,
            source,
            family,
            fit_control=self.control.operator,
            fit_budget_bytes=total // 4,
            components_budget_bytes=total // 2,
            aggregate_budget_bytes=total - result_reserve - start.log_lambda.nbytes,
            max_y=32,
        )
        execution = run_operator_efs_known_scale(
            provider,
            start.log_lambda,
            control=self.control.efs,
            initial_accuracy=OperatorLogDetControl(probes=4, lanczos_depth=4),
        )
        selected = execution.accepted
        base = GAMPredictionResult._from_operator_fit(
            fitted=selected.fit,
            sp=np.exp(selected.fit.log_lambda),
            formula=self.formula,
            method=self.method,
            control=self.control,
        )
        return replace(
            base,
            score=selected.components.score.value,
            score_error_bound=selected.score_error_bound,
            score_bound_method=selected.score_bound_method,
            scale=selected.reported_phi,
            score_scale=selected.score_phi,
            edf_total=selected.components.edf_total,
            converged=execution.converged,
            n_iter=execution.outer_iterations,
            convergence_info=execution.stop_reason,
            lambda_strategy="efs_reml",
            execution_route="operator",
            operator_diagnostics=replace(
                selected.fit.diagnostics,
                efs_indeterminate_comparisons=execution.indeterminate_comparisons,
            ),
            # Fit work only (the EFS start reduction and every refit and
            # score replay), as on the streamed routes. Phase-1 preparation is
            # reported in operator_diagnostics.preparation_*.
            source_scans=start.source_scans + provider.completed_source_scans,
            batches_scanned=start.source_batches + provider.completed_source_batches,
            known_workspace_bytes=max(
                start.peak_bound_bytes,
                result_reserve + provider.aggregate_peak_bound_bytes,
            ),
        )

    def _fit_stream(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None,
        offset: np.ndarray | None,
        *,
        result: Literal["full", "inference", "prediction"],
        family: ExponentialFamily,
    ) -> GAMPredictionResult:
        """Run the replayable-source route, including opt-in Gaussian QR.

        Route precedence: an explicit ``gaussian_compression=True`` request
        for an exact Gaussian identity Newton/fixed-sp fit takes the frozen
        Gaussian QR route. Every other fit, and an opted-in fit found
        ineligible, takes the published route it would take without the
        request; an unmet request is reported in ``execution_fallback_reason``.
        """
        from jaxgam.execution.stream import StreamPIRLSControl, fit_streamed_pirls

        if self.control.linear_solver == "pcg":
            raise ValueError("linear_solver='pcg' requires execution='operator'.")
        if not isinstance(data, RowSource):
            raise TypeError(
                "FitControl(execution='stream') requires a replayable RowSource."
            )
        if weights is not None or offset is not None:
            raise ValueError(
                "weights and offset must be owned by the RowSource for "
                "execution='stream'; do not pass separate arrays."
            )
        if result != "prediction":
            raise NotImplementedError(
                "Streamed fitting supports result='prediction' only; "
                "full and inference results retain unsupported row diagnostics."
            )
        estimated_efs = self.optimizer == "efs" and self.sp is None
        estimated_exact = self.optimizer == "newton" and self.sp is None
        execution_fallback_reason = _gaussian_qr_request_fallback(
            self.control, family, estimated_efs=estimated_efs
        )
        gaussian_qr_candidate = (
            self.control.gaussian_compression and execution_fallback_reason is None
        )
        if estimated_efs:
            if self.control.linear_solver != "qr":
                raise NotImplementedError(
                    "Streamed optimizer='efs' requires linear_solver='qr'."
                )
            _check_stream_efs_family(family)
        elif estimated_exact:
            # The frozen Gaussian QR route owns its own solves; only the
            # exact streamed REML route requires the QR coefficient solver.
            if self.control.linear_solver != "qr" and not gaussian_qr_candidate:
                raise NotImplementedError(
                    "Exact streamed REML requires linear_solver='qr'."
                )
            _check_stream_fixed_family(family, "qr")
        else:
            _check_stream_fixed_family(family, self.control.linear_solver)
        spec = parse_formula(self.formula)
        prepared = prepare_model(
            spec,
            data,
            family=family,
            batch_rows=self.control.batch_rows,
            maximum_bytes=self.control.memory_budget_bytes,
        )
        stream = StreamDesign(prepared, data)
        gaussian_qr_workspace = None
        if gaussian_qr_candidate:
            from jaxgam.execution.gaussian import (
                preflight_streamed_gaussian_qr_workspace,
            )

            assert prepared.fitting is not None
            response = prepared.fitting.response
            if response.weight_min < 1e-10 or response.weight_max > 1e10:
                execution_fallback_reason = (
                    "Gaussian QR currently requires unclipped finite positive weights"
                )
            else:
                # Reject an over-budget QR route before device placement.
                gaussian_qr_workspace = preflight_streamed_gaussian_qr_workspace(
                    prepared,
                    self.control.batch_rows,
                    self.control.memory_budget_bytes,
                    pin_lambda=self.sp is not None,
                )
        jax_device = _resolve_device(self.device)
        if estimated_efs:
            if prepared.penalties is None or prepared.penalties.n_penalties == 0:
                return _with_execution_fallback_reason(
                    _fit_stream_parametric_bypass(
                        stream,
                        family,
                        formula=self.formula,
                        method=self.method,
                        control=self.control,
                        device=jax_device,
                    ),
                    execution_fallback_reason,
                )
            from jaxgam.execution.efs_stream import fit_streamed_efs

            execution = fit_streamed_efs(
                stream,
                family,
                maximum_bytes=self.control.memory_budget_bytes,
                batch_rows=self.control.batch_rows,
                control=self.control.efs,
                device=jax_device,
            )
            return _with_execution_fallback_reason(
                GAMPredictionResult._from_stream_fit(
                    stream_state=execution.fit.pirls_result,
                    prepared=prepared,
                    metadata=execution.metadata,
                    family=execution.family,
                    formula=self.formula,
                    method=self.method,
                    control=self.control,
                    execution_route="stream_efs_qr",
                    selected_fit=execution.fit,
                    lambda_strategy="efs_reml",
                ),
                execution_fallback_reason,
            )
        metadata = None
        if gaussian_qr_workspace is not None:
            from jaxgam.execution.gaussian import GaussianQRIneligible
            from jaxgam.execution.gaussian_optimizer import (
                build_stream_gaussian_qr_provider,
            )

            metadata = PreparedFittingMetadata.from_prepared(
                prepared, family, jax_device
            )
            try:
                provider = build_stream_gaussian_qr_provider(
                    stream, metadata, batch_rows=self.control.batch_rows
                )
            except GaussianQRIneligible as error:
                execution_fallback_reason = str(error)
            else:
                return _fit_stream_gaussian_qr(
                    stream,
                    provider,
                    metadata,
                    family,
                    self.sp,
                    formula=self.formula,
                    method=self.method,
                    control=self.control,
                    known_workspace_bytes=gaussian_qr_workspace.required_bytes,
                )
        if estimated_exact and self.control.linear_solver != "qr":
            # Only an opted-in Gaussian QR candidate reaches this point with
            # another solver; its exact-REML fallback needs the QR solver.
            raise NotImplementedError(
                "Estimated-sp RowSource fitting requires an eligible Gaussian QR "
                f"route ({execution_fallback_reason}) or linear_solver='qr' for "
                "exact streamed REML."
            )
        known_workspace_bytes = _preflight_stream_workspace(
            prepared.n_coef,
            self.control.batch_rows,
            self.control.memory_budget_bytes,
            linear_solver=self.control.linear_solver,
        )
        if metadata is None:
            metadata = PreparedFittingMetadata.from_prepared(
                prepared, family, jax_device
            )
        if estimated_exact:
            return _with_execution_fallback_reason(
                _fit_stream_exact_reml(
                    stream,
                    family,
                    metadata,
                    formula=self.formula,
                    method=self.method,
                    control=self.control,
                    device=jax_device,
                ),
                execution_fallback_reason,
            )
        log_lambda = _fixed_log_smoothing_parameters(self.sp, metadata.n_penalties)
        from jaxgam.families.negative_binomial import NegativeBinomial

        if isinstance(family, NegativeBinomial):
            return _with_execution_fallback_reason(
                _fit_stream_fixed_nb(
                    stream,
                    family,
                    log_lambda,
                    metadata,
                    formula=self.formula,
                    method=self.method,
                    control=self.control,
                    device=jax_device,
                    supplied_sp=self.sp,
                ),
                execution_fallback_reason,
            )
        if not _stream_legacy_fisher_eligible(family) or (
            self.control.linear_solver == "qr" and family.scale_known
        ):
            return _with_execution_fallback_reason(
                _fit_stream_fixed_regular(
                    stream,
                    family,
                    log_lambda,
                    metadata,
                    formula=self.formula,
                    method=self.method,
                    control=self.control,
                    device=jax_device,
                    supplied_sp=self.sp,
                ),
                execution_fallback_reason,
            )
        stream_state = fit_streamed_pirls(
            stream,
            family,
            log_lambda,
            control=StreamPIRLSControl(
                batch_rows=self.control.batch_rows,
                solver_policy=self.control.linear_solver,
            ),
            device=jax_device,
        )
        return _with_execution_fallback_reason(
            GAMPredictionResult._from_stream_fit(
                stream_state=stream_state,
                prepared=prepared,
                metadata=metadata,
                family=family,
                formula=self.formula,
                method=self.method,
                control=self.control,
                execution_route=(
                    "stream_qr" if self.control.linear_solver == "qr" else "stream"
                ),
                known_workspace_bytes=known_workspace_bytes,
                supplied_smoothing_params=self.sp,
            ),
            execution_fallback_reason,
        )

    def _fit_discrete(
        self,
        data: pd.DataFrame | dict | RowSource,
        weights: np.ndarray | None,
        offset: np.ndarray | None,
        *,
        result: Literal["full", "inference", "prediction"],
        family: ExponentialFamily,
    ) -> GAMPredictionResult:
        """Run continuous-discrete fitting with fixed or working-model sp.

        Additive Gaussian (identity) models use the exact fixed-sp solve or
        the frozen-statistic working model. Families exposing the
        ``regular_performance_iteration`` capability (Poisson, Binomial and
        Gamma with the links pinned ``bam`` accepts for them) use pinned
        ``bgam.fitd``'s Fisher performance iteration; the negative binomial
        (``extended_performance_iteration``) uses its ``dDeta`` Newton
        working model with fixed or ``estimate.theta``-updated theta.

        Prior weights and the offset come from the ``RowSource``. Its
        ``has_explicit_weights``/``has_explicit_offset`` say whether the
        caller supplied them; an undeclared attribute counts as supplied, as
        pinned ``bam`` treats supplied ``weights=``/``offset=`` arguments
        (they enter ``discrete.mf``'s seeded padding whatever their values).

        The route is taken only on an explicit ``execution='discrete'``
        request and owns its solves: ``linear_solver='qr'`` is rejected, and a
        ``gaussian_compression`` request is reported unmet in
        ``execution_fallback_reason`` while the discrete fit runs unchanged.
        Smoothing parameters are fixed or selected by
        ``smoothing_strategy='working_model_reml'``; the dense
        ``optimizer='efs'`` selection is not a discrete strategy.
        """
        from jaxgam.execution.discrete import fit_discrete_gaussian
        from jaxgam.execution.discrete_regular import fit_discrete_regular
        from jaxgam.families.standard import Gaussian

        if not isinstance(data, RowSource):
            raise TypeError(
                "FitControl(execution='discrete') requires a replayable RowSource."
            )
        if weights is not None or offset is not None:
            raise ValueError(
                "weights and offset must be owned by the RowSource for "
                "execution='discrete'; do not pass separate arrays."
            )
        if result != "prediction":
            raise NotImplementedError(
                "Discrete fitting currently supports result='prediction' only; "
                "full and inference results retain unsupported row diagnostics."
            )
        if self.control.linear_solver != "cholesky":
            route = "operator" if self.control.linear_solver == "pcg" else "stream"
            raise NotImplementedError(
                f"linear_solver={self.control.linear_solver!r} currently requires "
                f"execution={route!r}."
            )
        if self.sp is None and self.control.smoothing_strategy != "working_model_reml":
            raise NotImplementedError(
                "Discrete fitting requires explicit fixed sp or opt-in "
                "smoothing_strategy='working_model_reml'."
            )
        if self.sp is None and self.optimizer != "newton":
            raise NotImplementedError(
                f"optimizer={self.optimizer!r} does not apply to discrete fitting; "
                "smoothing_strategy='working_model_reml' selects its smoothing "
                "parameters."
            )
        capabilities = family.execution_capabilities()
        if (
            isinstance(family, ExtendedFamily)
            and not capabilities.extended_performance_iteration
        ):
            raise NotImplementedError(
                f"Discrete fitting of the {family!r} family is not implemented: "
                "bgam.fitd's extended-family route is ported for the negative "
                "binomial family with its log, identity and sqrt links."
            )
        additive = isinstance(family, Gaussian) and family.is_canonical
        if not (
            additive
            or capabilities.regular_performance_iteration
            or capabilities.extended_performance_iteration
        ):
            raise NotImplementedError(
                "Discrete fitting supports Gaussian identity, the Poisson "
                "(log, identity, sqrt), Binomial (logit, probit, cloglog, log) "
                "and Gamma (inverse, identity, log) families, and the negative "
                f"binomial (log, identity, sqrt); got {family!r}."
            )
        spec = parse_formula(self.formula)
        device = _resolve_device(self.device)
        state = (
            fit_discrete_gaussian(
                spec, data, self.sp, control=self.control, device=device
            )
            if additive
            else fit_discrete_regular(
                spec, data, family, self.sp, control=self.control, device=device
            )
        )
        return _with_execution_fallback_reason(
            GAMPredictionResult._from_discrete(
                fit_state=state,
                family=family,
                formula=self.formula,
                method=self.method,
                control=self.control,
                supplied_smoothing_params=self.sp,
            ),
            _discrete_gaussian_qr_request_fallback(self.control),
        )


def _gaussian_qr_request_fallback(
    control: FitControl, family: ExponentialFamily, *, estimated_efs: bool
) -> str | None:
    """Return why an explicit Gaussian QR request cannot apply to this route."""
    from jaxgam.families.standard import Gaussian
    from jaxgam.links.links import IdentityLink

    if not control.gaussian_compression:
        return None
    if estimated_efs:
        return "Gaussian QR compression requires Newton smoothing selection."
    if type(family) is not Gaussian or not isinstance(family.link, IdentityLink):
        return "Gaussian QR requires the exact Gaussian family with identity link"
    return None


def _discrete_gaussian_qr_request_fallback(control: FitControl) -> str | None:
    """Return why a Gaussian QR request cannot apply to the discrete route."""
    if not control.gaussian_compression:
        return None
    return "Gaussian QR compression requires execution 'dense' or 'stream'."


def _with_execution_fallback_reason(
    result: GAMPredictionResult, reason: str | None
) -> GAMPredictionResult:
    """Record an unmet Gaussian QR request on a published-route result."""
    if reason is None:
        return result
    return replace(result, execution_fallback_reason=reason)


def _fit_stream_gaussian_qr(
    stream: StreamDesign,
    provider: object,
    metadata: PreparedFittingMetadata,
    family: ExponentialFamily,
    sp: np.ndarray | list | None,
    *,
    formula: str,
    method: str,
    control: FitControl,
    known_workspace_bytes: int,
) -> GAMPredictionResult:
    """Fit the opt-in frozen Gaussian QR route through the published seam.

    The selected Newton or large-penalty L-BFGS-B result carries the outer
    score and convergence provenance; the row-free provider state owns the
    coefficient factor.
    """
    fixed = sp is not None
    log_lambda = (
        _fixed_log_smoothing_parameters(sp, metadata.n_penalties)
        if fixed
        else metadata.log_lambda_init
    )
    from jaxgam.execution.gaussian_optimizer import (
        optimize_gaussian_qr_lbfgs,
        use_gaussian_qr_lbfgs,
    )

    if use_gaussian_qr_lbfgs(metadata.n_penalties):
        fit_result = optimize_gaussian_qr_lbfgs(
            metadata,
            provider,
            log_lambda_init=log_lambda,
            pin_lambda=fixed,
        )
        optimized_strategy = f"lbfgs_{method.lower()}"
    else:
        fit_result = newton_optimize(
            metadata,
            method,
            log_lambda_init=log_lambda,
            pin_lambda=fixed,
            score_provider=provider,
        )
        optimized_strategy = f"newton_{method.lower()}"
    lambda_strategy = "fixed" if fixed else optimized_strategy
    state = provider.to_stream_state(fit_result)
    selected = _selected_stream_fit(
        state,
        float(np.asarray(fit_result.score)),
        converged=bool(fit_result.converged),
        n_iter=int(fit_result.n_iter),
        convergence_info=fit_result.convergence_info,
    )
    return GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=stream.prepared,
        metadata=metadata,
        family=family,
        formula=formula,
        method=method,
        control=control,
        execution_route="stream_gaussian_qr",
        selected_fit=selected,
        lambda_strategy=lambda_strategy,
        source_scans=provider.source_scans,
        batches_scanned=provider.batches_scanned,
        known_workspace_bytes=known_workspace_bytes,
        supplied_smoothing_params=sp,
    )


def _stream_legacy_fisher_eligible(family: ExponentialFamily) -> bool:
    """Match the existing bounded Fisher controller's declared contract."""
    capabilities = family.execution_capabilities()
    policy = family.stream_reduction_policy()
    return (
        capabilities.coefficient_system == "fisher"
        and capabilities.fisher_equals_observed_for_score
        and policy.reported_scale in {"known_one", "gaussian_fisher_edf_deviance"}
        and policy.score_scale != "unsupported"
    )


def _check_stream_fixed_family(family: ExponentialFamily, linear_solver: str) -> None:
    """Separate family contract, parameter mode and signed-solver policy."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.links.links import Link

    capabilities = family.execution_capabilities()
    policy = family.stream_reduction_policy()
    if not all(
        (
            capabilities.row_separable,
            capabilities.direct_deviance,
            capabilities.saturated_loglikelihood,
            capabilities.fisher_working_system,
            capabilities.observed_information,
        )
    ):
        raise NotImplementedError(
            "Streamed fixed-sp fitting requires row-separable direct deviance, "
            "saturated likelihood, and Fisher/observed information primitives."
        )
    if policy.reported_scale == "unsupported":
        raise NotImplementedError(
            "Streamed fixed-sp fitting requires an explicit reported-scale policy."
        )
    if type(family.link).second_derivative is Link.second_derivative:
        raise NotImplementedError(
            "Streamed observed-information initialization requires a link "
            "second_derivative implementation."
        )
    if isinstance(family, NegativeBinomial):
        from jaxgam.execution.nb_stream import _link_name

        _link_name(family)
        if linear_solver != "qr":
            raise NotImplementedError(
                "Streamed NB observed-information fitting requires linear_solver='qr'."
            )
        if family.n_theta and not capabilities.dynamic_theta:
            raise NotImplementedError("Estimated NB lacks dynamic-theta capability.")
        return
    if capabilities.dynamic_theta or family.n_theta:
        raise NotImplementedError(
            "Streamed fixed-sp dynamic family parameters require an explicit "
            "exact-parameter driver."
        )
    if linear_solver != "qr" and not _stream_legacy_fisher_eligible(family):
        raise NotImplementedError(
            "Streamed observed-information fitting requires linear_solver='qr'."
        )
    if not family.scale_known and not capabilities.differentiable_deviance:
        raise NotImplementedError(
            "Unknown-scale streamed fixed-sp fitting requires differentiable deviance."
        )


def _selected_stream_fit(
    state,
    score: float,
    *,
    converged: bool,
    n_iter: int,
    convergence_info: str,
    theta: float | None = None,
) -> NewtonResult:
    """Pass reviewed scalar provenance to compact stream materialization."""
    import jax.numpy as jnp

    return NewtonResult(
        log_lambda=state.log_lambda,
        smoothing_params=jnp.exp(state.log_lambda),
        converged=converged,
        n_iter=n_iter,
        score=jnp.asarray(score),
        gradient=jnp.zeros_like(state.log_lambda),
        edf=state.edf,
        scale=state.scale,
        pirls_result=state,
        convergence_info=convergence_info,
        theta=theta,
    )


def _fit_stream_fixed_regular(
    stream: StreamDesign,
    family: ExponentialFamily,
    log_lambda: np.ndarray,
    metadata: PreparedFittingMetadata,
    *,
    formula: str,
    method: str,
    control: FitControl,
    device: jax.Device | None,
    supplied_sp: np.ndarray | list | float | None = None,
) -> GAMPredictionResult:
    from jaxgam.execution.regular_stream import fit_regular_streamed_pirls
    from jaxgam.execution.reml import optimize_regular_stream_reml
    from jaxgam.execution.stream import StreamPIRLSControl

    coefficient_control = StreamPIRLSControl(
        batch_rows=control.batch_rows,
        solver_policy="qr",
        # Canonical and unknown-scale cells retain the tighter final state.
        # Noncanonical known-scale boundary cells use the convergent 1e-9
        # source control; forcing 1e-11 there falsely exhausts recovery.
        tol=1e-9 if family.scale_known and not family.is_canonical else 1e-11,
    )
    selected = None
    if family.scale_known:
        fitted = fit_regular_streamed_pirls(
            stream,
            family,
            log_lambda,
            maximum_bytes=control.memory_budget_bytes,
            control=coefficient_control,
            device=device,
        )
        state = fitted.state
        source_scans = state.source_scans
        batches_scanned = state.batches_scanned
        known_workspace_bytes = fitted.workspace.required_bytes
    else:
        optimized = optimize_regular_stream_reml(
            stream,
            family,
            np.r_[log_lambda, 0.0],
            maximum_bytes=control.memory_budget_bytes,
            pin_lambda=True,
            pirls_control=coefficient_control,
            device=device,
        )
        state = optimized.trial.fit_result.state
        source_scans = optimized.cumulative_source_scans
        batches_scanned = optimized.cumulative_batches_scanned
        known_workspace_bytes = optimized.required_bytes
        selected = _selected_stream_fit(
            state,
            float(np.asarray(optimized.trial.score)),
            converged=optimized.converged,
            n_iter=optimized.n_iter,
            convergence_info=optimized.message,
        )
    return GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=stream.prepared,
        metadata=metadata,
        family=family,
        formula=formula,
        method=method,
        control=control,
        execution_route="stream_qr",
        selected_fit=selected,
        source_scans=source_scans,
        batches_scanned=batches_scanned,
        known_workspace_bytes=known_workspace_bytes,
        supplied_smoothing_params=supplied_sp,
    )


def _fit_stream_fixed_nb(
    stream: StreamDesign,
    family: ExponentialFamily,
    log_lambda: np.ndarray,
    metadata: PreparedFittingMetadata,
    *,
    formula: str,
    method: str,
    control: FitControl,
    device: jax.Device | None,
    supplied_sp: np.ndarray | list | float | None = None,
) -> GAMPredictionResult:
    from jaxgam.execution.nb_reml import optimize_nb_stream_reml
    from jaxgam.execution.nb_stream import fit_nb_streamed_pirls
    from jaxgam.execution.reml import StreamREMLControl
    from jaxgam.execution.stream import StreamPIRLSControl

    coefficient_control = StreamPIRLSControl(
        batch_rows=control.batch_rows, solver_policy="qr", tol=1e-11
    )
    if family.n_theta:
        optimized = optimize_nb_stream_reml(
            stream,
            family,
            np.r_[log_lambda, family.get_theta(transformed=False)],
            maximum_bytes=control.memory_budget_bytes,
            pin_lambda=True,
            pirls_control=coefficient_control,
            control=StreamREMLControl(gtol=1e-9, ftol=0.0),
            device=device,
        )
        fitted = optimized.trial.fit_result
        state = fitted.state
        source_scans = optimized.cumulative_source_scans
        batches_scanned = optimized.cumulative_batches_scanned
        known_workspace_bytes = optimized.required_bytes
        theta = float(np.exp(np.asarray(optimized.trial.params)[-1]))
        family.put_theta(np.log(np.asarray([theta])))
        selected = _selected_stream_fit(
            state,
            float(np.asarray(optimized.trial.score)),
            converged=optimized.converged,
            n_iter=optimized.n_iter,
            convergence_info=optimized.message,
            theta=theta,
        )
    else:
        fitted = fit_nb_streamed_pirls(
            stream,
            family,
            log_lambda,
            maximum_bytes=control.memory_budget_bytes,
            control=coefficient_control,
            device=device,
        )
        state = fitted.state
        source_scans = state.source_scans
        batches_scanned = state.batches_scanned
        known_workspace_bytes = fitted.workspace.required_bytes
        selected = _selected_stream_fit(
            state,
            fitted.reml_score,
            converged=state.converged,
            n_iter=state.n_iter,
            convergence_info=(
                "fixed sp streamed PIRLS"
                if state.converged
                else "fixed sp streamed PIRLS did not converge"
            ),
        )
    return GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=stream.prepared,
        metadata=metadata,
        family=family,
        formula=formula,
        method=method,
        control=control,
        execution_route="stream_qr",
        selected_fit=selected,
        source_scans=source_scans,
        batches_scanned=batches_scanned,
        known_workspace_bytes=known_workspace_bytes,
        supplied_smoothing_params=supplied_sp,
    )


def _fit_stream_exact_reml(
    stream: StreamDesign,
    family: ExponentialFamily,
    metadata: PreparedFittingMetadata,
    *,
    formula: str,
    method: str,
    control: FitControl,
    device: jax.Device | None,
) -> GAMPredictionResult:
    """Expose one accepted exact source trial through the public result seam."""
    from jaxgam.execution.efs_stream_start import prepare_stream_reml_start
    from jaxgam.execution.nb_reml import optimize_nb_stream_reml
    from jaxgam.execution.reml import (
        StreamREMLControl,
        optimize_regular_stream_reml,
    )
    from jaxgam.execution.stream import StreamPIRLSControl
    from jaxgam.families.negative_binomial import NegativeBinomial

    n_lambda = metadata.n_penalties
    if n_lambda == 0 and family.n_theta == 0 and family.scale_known:
        # Pinned gam.outer uses a plain coefficient fit when there is no
        # smoothing, dispersion, or theta parameter to select.
        empty = np.empty(0, dtype=np.float64)
        result = (
            _fit_stream_fixed_nb(
                stream,
                family,
                empty,
                metadata,
                formula=formula,
                method=method,
                control=control,
                device=device,
            )
            if isinstance(family, NegativeBinomial)
            else _fit_stream_fixed_regular(
                stream,
                family,
                empty,
                metadata,
                formula=formula,
                method=method,
                control=control,
                device=device,
            )
        )
        return replace(
            result,
            lambda_strategy="exact_reml_no_parameters",
            execution_route="stream_exact_reml_qr",
        )

    if n_lambda:
        startup = prepare_stream_reml_start(
            stream,
            family,
            maximum_bytes=control.memory_budget_bytes,
            batch_rows=control.batch_rows,
        )
        initial_log_lambda = startup.log_lambda
        initial_score_phi = startup.score_phi
        startup_source_scans = startup.source_scans
        startup_batches_scanned = startup.batches_scanned
        startup_workspace_bytes = startup.workspace_bytes
        retained_startup_bytes = 8 * (
            startup.log_lambda.size + startup.null_coefficients.size + 64
        )
    else:
        # Pinned mgcv.r:estimate.gam skips initial.spg when length(G$sp)==0.
        # The free dispersion/theta coordinate still takes the exact joint
        # optimizer; it must not enter the fixed-theta parametric bypass.
        initial_log_lambda = np.empty(0, dtype=np.float64)
        initial_score_phi = 1.0
        startup_source_scans = 0
        startup_batches_scanned = 0
        startup_workspace_bytes = 0
        retained_startup_bytes = 0
    if startup_workspace_bytes + retained_startup_bytes > control.memory_budget_bytes:
        raise MemoryError("Exact REML startup exceeds the known workspace budget.")
    trial_budget = control.memory_budget_bytes - retained_startup_bytes
    coefficient_control = StreamPIRLSControl(
        batch_rows=control.batch_rows,
        solver_policy="qr",
    )
    if isinstance(family, NegativeBinomial):
        nb_coefficient_control = replace(coefficient_control, tol=1e-11)
        initial_params = np.r_[
            initial_log_lambda,
            family.get_theta(transformed=False) if family.n_theta else [],
        ]
        optimized = optimize_nb_stream_reml(
            stream,
            family,
            initial_params,
            maximum_bytes=trial_budget,
            pirls_control=nb_coefficient_control,
            control=StreamREMLControl(gtol=1e-9, ftol=0.0),
            max_step=5.0,
            device=device,
        )
        state = optimized.trial.fit_result.state
        theta = None
        if family.n_theta:
            theta = float(np.exp(np.asarray(optimized.trial.params)[-1]))
            family.put_theta(np.log(np.asarray([theta])))
    else:
        initial_params = np.r_[
            initial_log_lambda,
            [] if family.scale_known else [np.log(initial_score_phi)],
        ]
        optimized = optimize_regular_stream_reml(
            stream,
            family,
            initial_params,
            maximum_bytes=trial_budget,
            pirls_control=coefficient_control,
            control=StreamREMLControl(ftol=0.0),
            max_step=5.0,
            device=device,
        )
        state = optimized.trial.fit_result.state
        theta = None
    selected = _selected_stream_fit(
        state,
        float(np.asarray(optimized.trial.score)),
        converged=optimized.converged,
        n_iter=optimized.n_iter,
        convergence_info=optimized.message,
        theta=theta,
    )
    return GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=stream.prepared,
        metadata=metadata,
        family=family,
        formula=formula,
        method=method,
        control=control,
        execution_route="stream_exact_reml_qr",
        selected_fit=selected,
        lambda_strategy="exact_reml",
        source_scans=startup_source_scans + optimized.cumulative_source_scans,
        batches_scanned=startup_batches_scanned + optimized.cumulative_batches_scanned,
        known_workspace_bytes=max(startup_workspace_bytes, optimized.required_bytes)
        + retained_startup_bytes,
        exact_optimizer_diagnostics=StreamExactOptimizerDiagnostics(
            status=optimized.status,
            n_evaluations=optimized.n_evaluations,
            n_accepted=optimized.n_accepted,
            projected_gradient_inf=optimized.projected_gradient_inf,
            accepted_score_history=optimized.accepted_score_history,
            startup_source_scans=startup_source_scans,
            startup_batches_scanned=startup_batches_scanned,
            cumulative_inner_iterations=optimized.cumulative_inner_iterations,
            cumulative_final_refit_solves=optimized.cumulative_final_refit_solves,
            cumulative_derivative_s=optimized.cumulative_derivative_s,
        ),
    )


def _check_stream_efs_family(family: ExponentialFamily) -> None:
    """Reject statically unsupported streamed-EFS families before preparation."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
    from jaxgam.links.links import IdentityLink, LogLink, SqrtLink

    if isinstance(family, NegativeBinomial):
        if not isinstance(family.link, (LogLink, IdentityLink, SqrtLink)):
            raise NotImplementedError(
                "Streamed EFS supports NB log, identity, and sqrt links only."
            )
        return
    if not isinstance(family, (Gaussian, Binomial, Poisson, Gamma)):
        raise NotImplementedError(
            "Streamed EFS supports registered Gaussian, Binomial, Poisson, "
            "Gamma, and NegativeBinomial families only."
        )
    if family.execution_parameter_snapshot().theta_mode != "none":
        raise NotImplementedError("Streamed regular EFS does not accept theta state.")


def _fit_stream_parametric_bypass(
    stream: StreamDesign,
    family: ExponentialFamily,
    *,
    formula: str,
    method: str,
    control: FitControl,
    device: jax.Device | None,
) -> GAMPredictionResult:
    """Preserve the ordinary no-penalty fit before streamed EFS dispatch."""
    from jaxgam.execution.nb_stream import fit_nb_streamed_pirls
    from jaxgam.execution.regular_stream import fit_regular_streamed_pirls
    from jaxgam.execution.stream import StreamPIRLSControl
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.fitting.family_execution import (
        FamilyExecutionLineage,
        FamilyExecutionParameters,
    )

    prepared = stream.prepared
    coefficient_control = StreamPIRLSControl(
        batch_rows=control.batch_rows,
        max_iter=control.efs.pirls_max_iter,
        tol=control.efs.pirls_tolerance,
        solver_policy="qr",
    )
    empty_rho = np.zeros(0, dtype=np.float64)
    if isinstance(family, NegativeBinomial):
        lineage = FamilyExecutionLineage.from_prepared(prepared, family)
        parameters = (
            FamilyExecutionParameters.from_snapshot(lineage.parameters)
            if family.n_theta
            else None
        )
        state = fit_nb_streamed_pirls(
            stream,
            family,
            empty_rho,
            maximum_bytes=control.memory_budget_bytes,
            parameters=parameters,
            control=coefficient_control,
            device=device,
            estimate_theta=False,
        ).state
    else:
        regular = fit_regular_streamed_pirls(
            stream,
            family,
            empty_rho,
            maximum_bytes=control.memory_budget_bytes,
            score_scale=1.0,
            control=coefficient_control,
            device=device,
        )
        state = regular.state
        if not family.scale_known:
            # Dense Newton's no-penalty shortcut scores at the reported
            # Fletcher scale. Coefficients are dispersion-invariant, so no
            # second coefficient fit or EFS outer loop is needed. The
            # saturated likelihood is scale-dependent and needs one bounded
            # replay at that selected scale.
            from jaxgam.execution.stream import _source_batches

            lineage = FamilyExecutionLineage.from_prepared(prepared, family)
            reported_scale = float(np.asarray(state.scale))
            saturated = 0.0
            batches = 0
            for _batch, y, weight, _offset, _valid in _source_batches(
                stream, family, lineage, coefficient_control.batch_rows
            ):
                saturated += float(
                    np.asarray(
                        family.saturated_loglik(y, weight, reported_scale, max_y=0)
                    )
                )
                batches += 1
            if not np.isfinite(saturated):
                raise FloatingPointError(
                    "parametric streamed saturated likelihood is nonfinite"
                )
            state = replace(
                state,
                score_scale=state.scale,
                saturated_loglik=np.asarray(saturated),
                source_scans=state.source_scans + 1,
                batches_scanned=state.batches_scanned + batches,
            )
    metadata = PreparedFittingMetadata.from_prepared(prepared, family, device)
    return GAMPredictionResult._from_stream_fit(
        stream_state=state,
        prepared=prepared,
        metadata=metadata,
        family=family,
        formula=formula,
        method=method,
        control=control,
        execution_route="stream_parametric_qr",
        lambda_strategy=f"newton_{method.lower()}",
    )


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------


def _preflight_stream_workspace(
    n_coef: int,
    batch_rows: int,
    memory_budget_bytes: int,
    *,
    linear_solver: Literal["cholesky", "qr"] = "cholesky",
) -> int:
    """Reject stream fits whose known live fit workspace exceeds its budget.

    This conservatively allows old and candidate coefficient reductions,
    penalty/system/factor buffers, and the two triangular-solve workspaces
    used for total Fisher EDF, alongside one real batch's design and working
    vectors. It is an estimate of known array workspaces, not an allocator or
    native-library hard cap. CPU-only basis preparation is separate.
    """
    itemsize = np.dtype(np.float64).itemsize
    if linear_solver not in ("cholesky", "qr"):
        raise ValueError("linear_solver must be 'cholesky' or 'qr'.")
    if linear_solver == "qr":
        # During a noninitial merge the source design, weighted design,
        # [old R; weighted X], and SciPy's packed reflector copy coexist.
        # The two stacked arrays each add p rows. Coefficient-space reporting,
        # factors, penalty roots, and final Fisher actions retain the same
        # conservative twelve-matrix allowance as the Cholesky route.
        matrix_bytes = 14 * n_coef * n_coef * itemsize
        batch_bytes = (
            4 * batch_rows * n_coef + 12 * batch_rows + 6 * n_coef
        ) * itemsize
    else:
        matrix_bytes = 12 * n_coef * n_coef * itemsize
        batch_bytes = (3 * batch_rows * n_coef + 8 * batch_rows + 4 * n_coef) * itemsize
    required = matrix_bytes + batch_bytes
    if required > memory_budget_bytes:
        solver_label = " QR" if linear_solver == "qr" else ""
        raise MemoryError(
            f"Known streamed{solver_label} PIRLS workspace requires "
            f"{required} bytes, exceeding FitControl.memory_budget_bytes="
            f"{memory_budget_bytes}. Phase-1 setup scans are budgeted separately."
        )
    return required


def _resolve_device(device: str | None) -> jax.Device | None:
    """Resolve a device string to a JAX device object."""
    if device is None:
        return None
    import jax

    if device == "cpu":
        return jax.devices("cpu")[0]
    if device == "gpu":
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        if not gpu_devices:
            raise RuntimeError(
                "device='gpu' requested but no GPU backend found. "
                "Install jax[cuda12] (NVIDIA) or jax-metal (Apple Silicon)."
            )
        return gpu_devices[0]
    # _check_scope_guards validates device before this is called, so this
    # line is unreachable.  Raise explicitly for defensive clarity.
    raise ValueError(f"Unrecognized device: {device!r}")


#: Keyword arguments accepted by ``GAM`` beyond the explicit positional params
#: (formula, family, method, sp). Anything else is a typo (e.g. ``tol=``,
#: ``max_iter=``) and must be rejected rather than silently ignored.
_ACCEPTED_KWARGS = frozenset(
    {"device", "backend", "optimizer", "select", "gamma", "knots"}
)


def _check_scope_guards(method: str, kwargs: dict) -> None:
    """Validate v1.0 scope guards."""
    unknown = set(kwargs) - _ACCEPTED_KWARGS
    if unknown:
        raise TypeError(
            f"GAM() got unexpected keyword argument(s) {sorted(unknown)}. "
            f"Accepted arguments: formula, family, method, sp, "
            f"{', '.join(sorted(_ACCEPTED_KWARGS))}."
        )

    method_upper = method.upper()
    if method_upper == "ML":
        raise NotImplementedError(
            "method='ML' is not supported. mgcv's ML criterion uses the "
            "penalty range-space projection of log|X'WX+S| (the C routine "
            "MLpenalty1 in gdi.c), which differs from REML's full-space "
            "determinant. Only method='REML' is available; ML is deferred "
            "until the range-space determinant is implemented. "
            "See docs/design.md Section 4.4."
        )
    if method_upper != "REML":
        raise ValueError(
            f"method must be 'REML', got {method!r}. "
            "ML is not available (see above); GCV/UBRE is planned for v1.1."
        )

    backend = kwargs.get("backend")
    if backend is not None and backend != "jax":
        raise NotImplementedError(
            f"backend={backend!r} is not supported in v1.0. "
            "Only 'jax' backend is available. See docs/design.md Section 10."
        )

    device = kwargs.get("device")
    if device is not None and device not in ("cpu", "gpu"):
        raise ValueError(
            f"device={device!r} is not recognized. Use 'cpu', 'gpu', or None."
        )

    optimizer = kwargs.get("optimizer")
    if optimizer is not None:
        if not isinstance(optimizer, str):
            raise ValueError("optimizer must be 'newton', 'efs', or None.")
        if optimizer not in ("newton", "efs"):
            raise NotImplementedError(
                f"optimizer={optimizer!r} is not supported. Use 'newton' or 'efs'."
            )

    if kwargs.get("select", False):
        raise NotImplementedError(
            "select=True (shrinkage smoothing) is planned for v1.1. "
            "See docs/design.md Section 4.6."
        )

    gamma = kwargs.get("gamma", 1.0)
    if gamma != 1.0:
        raise NotImplementedError(
            f"gamma={gamma} is not supported in v1.0. "
            "Only gamma=1.0 (standard REML) is available."
        )

    if kwargs.get("knots") is not None:
        raise NotImplementedError(
            "User-specified knots are planned for v1.1. See docs/design.md Section 5.2."
        )


def _fixed_log_smoothing_parameters(sp: np.ndarray | list, n_penalties: int):
    """Validate natural-scale fixed smoothing parameters and log-transform."""
    import jax.numpy as jnp

    sp_arr = np.atleast_1d(np.asarray(sp, dtype=np.float64))
    if sp_arr.shape[0] != n_penalties:
        raise ValueError(
            f"sp has {sp_arr.shape[0]} elements but model has "
            f"{n_penalties} penalty terms."
        )
    if np.any(sp_arr <= 0.0) or not np.all(np.isfinite(sp_arr)):
        bad = sp_arr[(sp_arr <= 0.0) | ~np.isfinite(sp_arr)]
        raise ValueError(
            f"sp must contain strictly positive, finite values (smoothing "
            f"parameters are on the natural scale and are log-transformed "
            f"internally); got invalid value(s) {bad.tolist()}."
        )
    return jnp.log(jnp.array(sp_arr))


def _fit_fixed_sp(
    fd: FittingData,
    sp: np.ndarray | list,
    method: str = "REML",
    *,
    score_provider: OuterScoreProvider | None = None,
) -> NewtonResult:
    """Fit with user-supplied fixed smoothing parameters.

    For standard families this runs a single PIRLS at the given lambda (no
    Newton optimization). For extended families with an estimated dispersion
    parameter (e.g. Negative Binomial theta), the smoothing parameters are
    pinned but theta is still estimated via the outer optimizer — matching
    mgcv, where fixing ``sp`` does not fix theta (``gam.fit4`` keeps family
    parameters in the outer optimization).

    Parameters
    ----------
    fd : FittingData
        Phase 1→2 boundary data.
    sp : array-like
        Smoothing parameters on the original scale, shape ``(n_penalties,)``.
        A scalar is accepted for single-penalty models.
    method : str
        ``"REML"`` — used only when theta is estimated. (ML is not
        supported in v1.0; the public API rejects it before this point.)

    Returns
    -------
    NewtonResult
        For known-scale standard families: ``n_iter=0``,
        ``convergence_info="fixed sp"``; a non-finite REML score is reported
        as not converged (``"fixed sp: non-finite REML score"``).
        For estimated-theta or unknown-scale families: the pinned
        outer-optimizer result, with the supplied smoothing parameters held
        bitwise (never clamped) and theta / scale estimated at them.
    """
    import jax.numpy as jnp

    log_lambda = _fixed_log_smoothing_parameters(sp, fd.n_penalties)

    if score_provider is not None:
        from jaxgam.execution.gaussian_optimizer import (
            optimize_gaussian_qr_lbfgs,
            use_gaussian_qr_lbfgs,
        )

        if use_gaussian_qr_lbfgs(fd.n_penalties):
            return optimize_gaussian_qr_lbfgs(
                fd,
                score_provider,
                log_lambda_init=log_lambda,
                pin_lambda=True,
            )

    # Families with an estimated dispersion parameter still co-optimize it with
    # the smoothing parameters held fixed, matching mgcv (gam.outer keeps theta
    # for extended families and appends log.scale as an extra "smoothing
    # parameter" for unknown-scale families, mgcv.r:2025-2039). Route both
    # through the pinned-Newton path so the reported REML score is evaluated at
    # the jointly-optimal (theta, phi) and the real sp stay fixed.
    if fd.family.n_theta > 0 or not fd.family.scale_known:
        return newton_optimize(
            fd,
            method,
            log_lambda_init=log_lambda,
            pin_lambda=True,
            score_provider=score_provider,
        )

    S_lam = fd.S_lambda(log_lambda)

    # Initialize beta and run PIRLS
    beta_init = fd.beta_init
    if beta_init is None:
        beta_init = initialize_beta(
            np.asarray(fd.X),
            np.asarray(fd.y),
            np.asarray(fd.wt),
            fd.family,
            np.asarray(fd.offset) if fd.offset is not None else None,
        )

    log_theta = None
    if fd.family.n_theta > 0:
        log_theta = jnp.asarray(fd.family.get_theta(transformed=False))

    pirls_result = pirls_loop(
        fd.X,
        fd.y,
        beta_init,
        S_lam,
        fd.family,
        wt=fd.wt,
        offset=fd.offset,
        log_theta=log_theta,
    )

    # Known-scale standard families (Poisson/Binomial): no free dispersion
    # parameter, so the REML score is a single criterion evaluation at the fixed
    # sp (mgcv's no.sps / gam2objective path, mgcv.r:1692-1716). Build the same
    # REMLCriterion the estimated-sp path uses and evaluate it — rather than
    # stubbing score=0.0 — so GAMResults.score is the real REML criterion (R's
    # gcv.ubre). REMLCriterion recomputes EDF/scale internally with the Fisher
    # weighting, identical to the values below.
    criterion = REMLCriterion(fd, pirls_result)
    edf = criterion.edf
    scale = criterion.scale
    score = criterion.score(log_lambda)
    # A non-finite REML score is a failed evaluation, never a converged fit.
    finite_score = bool(jnp.isfinite(score))

    return NewtonResult(
        log_lambda=log_lambda,
        smoothing_params=jnp.exp(log_lambda),
        converged=bool(pirls_result.converged) and finite_score,
        n_iter=0,
        score=score,
        gradient=jnp.zeros_like(log_lambda),
        edf=edf,
        scale=scale,
        pirls_result=pirls_result,
        convergence_info="fixed sp"
        if finite_score
        else "fixed sp: non-finite REML score",
    )
