"""Budget-aware sensor selection policy.

The Sunrise CPD experiment always operates on the same multi-sensor
network. The *budget regime* determines how many of those sensors the
sampling policy is allowed to activate. The current implementation
ranks sensors by the per-sensor Gaussian divergence ``D_i`` estimated on
the sunrise window:

    D_i = (mu_1 - mu_0)^2 / (2 sigma^2)

Following the paper, when sensing and communication costs ``C_i`` and
``T_i`` are not available we assume unit cost and rank by ``D_i``
directly, which is equivalent to using the cost-aware ratio
``D_i / (C_i + T_i)`` with ``C_i + T_i = 1``. The function
:func:`select_sensors_by_budget` keeps a hook for cost-aware ranking
once costs become available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class BudgetSelection:
    """Result of applying a budget policy to the candidate sensors."""

    selected_sensors: list[str]
    budget_regime: str
    selection_reason: str
    ranking_used: list[dict]
    unit_cost_assumption: bool
    k: int

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""
        return {
            "selected_sensors": list(self.selected_sensors),
            "budget_regime": self.budget_regime,
            "selection_reason": self.selection_reason,
            "ranking_used": self.ranking_used,
            "unit_cost_assumption": self.unit_cost_assumption,
            "k": self.k,
        }


def _rank_candidates(valid_sensors: Iterable[str],
                     informativeness: Mapping[str, float]
                     ) -> list[dict]:
    """Return valid sensors sorted by descending ``D_i``.

    Sensors without an informativeness entry are ranked last with score
    ``-inf`` so they are not preferred but remain available when the
    high-budget regime requests every valid sensor.
    """
    ranking = []
    for sid in valid_sensors:
        d = informativeness.get(str(sid))
        ranking.append({
            "sensor_id": str(sid),
            "D_i": d,
            "score": d if d is not None else float("-inf"),
        })
    ranking.sort(key=lambda r: r["score"], reverse=True)
    for rank, entry in enumerate(ranking, start=1):
        entry["rank"] = rank
    return ranking


def select_sensors_by_budget(valid_sensors: Iterable[str],
                             sensor_informativeness: list[dict],
                             regime: str,
                             k: int = 3
                             ) -> BudgetSelection:
    """Apply a budget policy to a candidate sensor set.

    Args:
        valid_sensors: Identifiers of the sensors that survived the
            preprocessing filters.
        sensor_informativeness: List of records produced by
            :mod:`src.informativeness` (each with ``sensor_id`` and
            ``D_i``).
        regime: ``"low"``, ``"medium"`` or ``"high"``.
        k: Subset size for the medium-budget regime.

    Returns:
        :class:`BudgetSelection` describing the applied policy.
    """
    regime = regime.lower()
    if regime not in {"low", "medium", "high"}:
        raise ValueError(f"Unknown budget regime: {regime!r}")

    info_map = {
        str(rec["sensor_id"]): rec.get("D_i")
        for rec in sensor_informativeness
    }
    valid_list = [str(s) for s in valid_sensors]
    ranking = _rank_candidates(valid_list, info_map)

    if regime == "low":
        n = 1 if ranking else 0
        selected = [r["sensor_id"] for r in ranking[:n]]
        if selected:
            top = ranking[0]
            reason = (
                f"Low-budget policy: selected sensor {selected[0]} as the "
                f"single most informative sensor among {len(valid_list)} "
                f"valid sensors (D_i={top['D_i']})."
            )
        else:
            reason = "No valid sensors available; selection is empty."
    elif regime == "medium":
        n = max(0, min(k, len(ranking)))
        selected = [r["sensor_id"] for r in ranking[:n]]
        reason = (
            f"Medium-budget policy: selected the top-{n} sensors by D_i "
            f"out of {len(valid_list)} valid sensors."
        )
    else:  # high
        selected = [r["sensor_id"] for r in ranking]
        reason = (
            f"High-budget policy: selected all {len(selected)} valid "
            f"sensors."
        )

    return BudgetSelection(
        selected_sensors=selected,
        budget_regime=regime,
        selection_reason=reason,
        ranking_used=ranking,
        unit_cost_assumption=True,
        k=k,
    )
