from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..history import EvaluationHistory
from ..interfaces import DecisionModel
from ..types import GenerationState, SurrogateDecision
from .surrogate_bundle import SurrogateBundle


@dataclass(slots=True)
class DecisionController:
    """
    Controller responsible for preparing the decision-model input and selecting
    the surrogate bundle for the current generation.

    The controller may keep lightweight runtime statistics for logging or later
    analysis, but these are deliberately NOT exposed to the decision model input,
    to keep training-time and deployment-time representations aligned.
    """

    decision_model: DecisionModel

    last_selected_name: str | None = None
    last_goodness_by_bundle: dict[str, float] = field(default_factory=dict)
    selection_count_by_bundle: dict[str, int] = field(default_factory=dict)
    last_true_ratio_by_bundle: dict[str, float] = field(default_factory=dict)
    recent_improvement_by_bundle: dict[str, float] = field(default_factory=dict)

    def choose_bundle(
        self,
        bundles: list[SurrogateBundle],
        history: EvaluationHistory,
        candidate_x: np.ndarray,
        optimizer_state: dict,
        generation_index: int,
    ) -> tuple[SurrogateBundle, SurrogateDecision]:
        """
        Build the decision state, query the decision model, and return the
        selected surrogate bundle together with the raw decision output.

        Parameters
        ----------
        bundles:
            Candidate surrogate bundles available in the current generation.
        history:
            Evaluation history available before the current decision.
        candidate_x:
            Current candidate population proposed by the optimizer, shape [M, D].
        optimizer_state:
            Optimizer descriptors exposed to the decision model.
        generation_index:
            Index of the current generation.

        Returns
        -------
        chosen_bundle, decision
            The selected surrogate bundle and the raw decision model output.
        """
        if not bundles:
            raise ValueError("bundles must not be empty.")

        surrogate_names = [bundle.name for bundle in bundles]
        if len(set(surrogate_names)) != len(surrogate_names):
            raise ValueError(f"Bundle names must be unique, got {surrogate_names!r}")

        candidate_x = np.asarray(candidate_x, dtype=np.float32)
        if candidate_x.ndim != 2:
            raise ValueError(f"candidate_x must have shape [M, D], got {candidate_x.shape}.")
        candidate_x = np.nan_to_num(
            candidate_x,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        optimizer_state = dict(optimizer_state)

        history_x = np.asarray(history.x, dtype=np.float32)
        history_y = np.asarray(history.y, dtype=np.float32).reshape(-1)

        if history_x.ndim != 2:
            raise ValueError(f"history.x must have shape [N, D], got {history_x.shape}.")
        if len(history_x) != len(history_y):
            raise ValueError("history.x and history.y must have the same number of rows.")

        history_x = np.nan_to_num(
            history_x,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        history_y = np.nan_to_num(
            history_y,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        if history_x.shape[0] > 0 and history_x.shape[1] != candidate_x.shape[1]:
            raise ValueError(
                "history.x and candidate_x must have the same feature dimension. "
                f"Got {history_x.shape[1]} vs {candidate_x.shape[1]}."
            )

        inferred_dim = int(history_x.shape[1]) if history_x.shape[0] > 0 else int(candidate_x.shape[1])

        incumbent_x = np.asarray(history.incumbent_x, dtype=np.float32).reshape(-1)
        if incumbent_x.size == 0:
            incumbent_x = np.zeros(inferred_dim, dtype=np.float32)
        elif len(incumbent_x) != inferred_dim:
            raise ValueError(f"incumbent_x must have length {inferred_dim}, got {len(incumbent_x)}.")

        incumbent_x = np.nan_to_num(
            incumbent_x,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        incumbent_y = float(history.incumbent_y)
        if not np.isfinite(incumbent_y):
            if len(history_y) > 0:
                incumbent_y = float(np.min(history_y))
            else:
                incumbent_y = 0.0

        state = GenerationState(
            generation_index=int(generation_index),
            evaluated_history_x=history_x.copy(),
            evaluated_history_y=history_y.copy(),
            candidate_x=candidate_x.copy(),
            incumbent_x=incumbent_x.copy(),
            incumbent_y=float(incumbent_y),
            optimizer_state=optimizer_state,
            metadata={},
        )

        decision = self.decision_model.score(
            state=state,
            surrogate_names=surrogate_names,
        )

        if decision.chosen_surrogate_name not in surrogate_names:
            raise ValueError(
                "Decision model returned an unknown surrogate bundle name: "
                f"{decision.chosen_surrogate_name!r}. "
                f"Known bundles: {surrogate_names!r}"
            )

        chosen = next(bundle for bundle in bundles if bundle.name == decision.chosen_surrogate_name)

        # Enrich decision metadata with runtime diagnostics without changing the
        # decision-model input representation.
        metadata = dict(decision.metadata) if decision.metadata is not None else {}
        metadata.update(
            {
                "surrogate_names": surrogate_names,
                "history_size": int(len(history_y)),
                "candidate_count": int(len(candidate_x)),
                "generation_index": int(generation_index),
            }
        )
        decision = SurrogateDecision(
            goodness=dict(decision.goodness),
            chosen_surrogate_name=decision.chosen_surrogate_name,
            metadata=metadata,
        )

        return chosen, decision

    def update_statistics(
        self,
        *,
        chosen_bundle_name: str,
        decision: SurrogateDecision,
        true_ratio: float,
        improvement: float,
    ) -> None:
        """
        Update controller-side per-bundle statistics after the selected bundle
        has been executed.

        These statistics are kept only for controller-side bookkeeping and
        analysis. They are not injected into the decision-model input.

        Parameters
        ----------
        chosen_bundle_name:
            Name of the surrogate bundle used in the current generation.
        decision:
            Decision model output returned by `choose_bundle`.
        true_ratio:
            Fraction of points that were evaluated with the true objective
            in the selected bundle during this generation.
        improvement:
            Realized improvement attributable to this generation.
        """
        self.last_selected_name = chosen_bundle_name

        for name, goodness in decision.goodness.items():
            self.last_goodness_by_bundle[name] = float(goodness)

        self.selection_count_by_bundle[chosen_bundle_name] = (
            self.selection_count_by_bundle.get(chosen_bundle_name, 0) + 1
        )

        self.last_true_ratio_by_bundle[chosen_bundle_name] = float(true_ratio)
        self.recent_improvement_by_bundle[chosen_bundle_name] = float(improvement)
