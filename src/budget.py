"""Budget-aware sensor selection policy.

The Sunrise CPD experiment always operates on the same multi-sensor
network. The *budget regime* fixes a numeric resource budget ``B`` that
limits how much sensing/communication cost the sampling policy may
spend per timestamp; the active subset ``S(t)`` is then selected
*dynamically at each timestamp* by greedy information-per-cost ranking
under the constraint

    sum_{i in S(t)} (C_i + T_i)  <=  B.

Per-sensor information is the global empirical Gaussian divergence

    D_i = (mu_1 - mu_0)^2 / (2 sigma^2)

estimated once per sensor from the dataset (see
:mod:`src.informativeness`). Because the SensorScope dataset does not
provide real sensing or communication costs, the project uses an
explicit unit-cost approximation by default:

    C_i = 1.0,  T_i = 0.0,  total_cost_i = 1.0

so that ``B`` reduces to a per-timestamp cardinality constraint while
still being represented as an explicit numeric budget. The previous
static helper :func:`select_sensors_by_budget` is kept only as a
preliminary diagnostic that picks a fixed top-`k` subset once per
scenario; the experiment runner uses the dynamic policy
:func:`select_sensors_at_time` instead.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from . import paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

DEFAULT_SENSING_COST = 1.0
DEFAULT_COMMUNICATION_COST = 0.0
DEFAULT_COST_SOURCE = "unit_cost_assumption"

SENSOR_COSTS_JSON = paths.JSON_DIR / "sensor_costs.json"


@dataclass
class SensorCost:
    """Per-sensor sensing and communication cost."""

    sensor_id: str
    sensing_cost: float = DEFAULT_SENSING_COST
    communication_cost: float = DEFAULT_COMMUNICATION_COST
    cost_source: str = DEFAULT_COST_SOURCE

    @property
    def total_cost(self) -> float:
        return float(self.sensing_cost) + float(self.communication_cost)

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "sensing_cost": float(self.sensing_cost),
            "communication_cost": float(self.communication_cost),
            "total_cost": float(self.total_cost),
            "cost_source": self.cost_source,
        }


def default_sensor_costs(valid_sensors: Iterable[str]
                         ) -> dict[str, SensorCost]:
    """Return default unit-cost ``SensorCost`` records for every sensor."""
    return {
        str(sid): SensorCost(sensor_id=str(sid))
        for sid in valid_sensors
    }


def write_sensor_costs(costs: Mapping[str, SensorCost],
                       out_path: Path = SENSOR_COSTS_JSON) -> Path:
    """Persist a sensor cost map to JSON."""
    paths.ensure_dirs()
    payload = {
        "cost_model": "unit_cost_assumption",
        "cost_model_description": (
            "Default unit costs are used because the SensorScope "
            "dataset does not provide per-sensor sensing or "
            "communication costs."
        ),
        "default_sensing_cost": DEFAULT_SENSING_COST,
        "default_communication_cost": DEFAULT_COMMUNICATION_COST,
        "sensors": [costs[sid].to_dict() for sid in sorted(costs)],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Sensor costs written to %s", out_path)
    return out_path


def load_sensor_costs(path: Path = SENSOR_COSTS_JSON
                      ) -> dict[str, SensorCost]:
    """Load a sensor cost map; return an empty dict if absent."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[str, SensorCost] = {}
    for rec in payload.get("sensors", []):
        sid = str(rec["sensor_id"])
        out[sid] = SensorCost(
            sensor_id=sid,
            sensing_cost=float(rec.get("sensing_cost",
                                       DEFAULT_SENSING_COST)),
            communication_cost=float(rec.get("communication_cost",
                                             DEFAULT_COMMUNICATION_COST)),
            cost_source=str(rec.get("cost_source", DEFAULT_COST_SOURCE)),
        )
    return out


# ---------------------------------------------------------------------------
# Budget policy configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetPolicyConfig:
    """Configuration of the dynamic budget-constrained sampling policy."""

    regime: str
    budget: float
    unit_cost_assumption: bool = True
    score_name: str = "D_i_per_total_cost"

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "budget": float(self.budget),
            "unit_cost_assumption": bool(self.unit_cost_assumption),
            "score_name": self.score_name,
        }


def regime_budget(regime: str, n_valid_sensors: int) -> float:
    """Return the default numeric budget for a regime under unit costs.

    With ``C_i = 1.0`` and ``T_i = 0.0`` the per-sensor total cost is
    ``1.0``; the regime budgets are therefore:

        low    -> 1.0
        medium -> 3.0
        high   -> n_valid_sensors  (any active subset fits)
    """
    regime = regime.lower()
    if regime == "low":
        return 1.0
    if regime == "medium":
        return 3.0
    if regime == "high":
        return float(max(n_valid_sensors, 1))
    raise ValueError(f"Unknown regime: {regime!r}")


# ---------------------------------------------------------------------------
# Dynamic per-timestamp selection
# ---------------------------------------------------------------------------

@dataclass
class DynamicBudgetSelection:
    """Result of applying the dynamic budget policy at a single time."""

    selected_sensors: list[str]
    available_sensors: list[str]
    rejected_sensors: list[str]
    total_cost: float
    budget: float
    selection_scores: list[dict] = field(default_factory=list)
    selection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "selected_sensors": list(self.selected_sensors),
            "available_sensors": list(self.available_sensors),
            "rejected_sensors": list(self.rejected_sensors),
            "total_cost": float(self.total_cost),
            "budget": float(self.budget),
            "selection_scores": list(self.selection_scores),
            "selection_reason": self.selection_reason,
        }


def _is_finite(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def select_sensors_at_time(available_sensors: Iterable[str],
                           sensor_informativeness: Mapping[str, float],
                           sensor_costs: Mapping[str, SensorCost],
                           budget: float,
                           ) -> DynamicBudgetSelection:
    """Greedy information-per-cost selection at a single timestamp.

    Args:
        available_sensors: Sensors that report a finite observation at
            the current timestamp ``t``.
        sensor_informativeness: Mapping ``sensor_id -> D_i`` (global
            empirical divergence).
        sensor_costs: Mapping ``sensor_id -> SensorCost``. Sensors
            without a cost entry receive default unit costs.
        budget: Numeric budget ``B`` available at this timestamp.

    Returns:
        A :class:`DynamicBudgetSelection` describing the active subset
        ``S(t)``, its total cost, and the scores that drove the choice.
    """
    available = [str(s) for s in available_sensors]
    scored: list[dict] = []
    for sid in available:
        d = sensor_informativeness.get(sid)
        cost = sensor_costs.get(sid) or SensorCost(sensor_id=sid)
        total_cost = cost.total_cost
        if (d is None or not _is_finite(d)
                or total_cost <= 0 or not _is_finite(total_cost)):
            score = float("-inf")
        else:
            score = float(d) / float(total_cost)
        scored.append({
            "sensor_id": sid,
            "D_i": (float(d) if d is not None and _is_finite(d) else None),
            "sensing_cost": float(cost.sensing_cost),
            "communication_cost": float(cost.communication_cost),
            "total_cost": float(total_cost),
            "score": score,
        })

    scored.sort(key=lambda r: r["score"], reverse=True)

    selected: list[str] = []
    used_cost = 0.0
    rejected: list[str] = []
    for entry in scored:
        if entry["score"] == float("-inf"):
            rejected.append(entry["sensor_id"])
            continue
        if used_cost + entry["total_cost"] <= budget + 1e-12:
            selected.append(entry["sensor_id"])
            used_cost += entry["total_cost"]
        else:
            rejected.append(entry["sensor_id"])

    if selected:
        reason = (
            f"Greedy information-per-cost selection: chose "
            f"{len(selected)}/{len(available)} available sensors using "
            f"budget {used_cost:.3f} of {budget:.3f}."
        )
    else:
        reason = (
            f"No sensors selected at this timestamp (available="
            f"{len(available)}, budget={budget:.3f})."
        )

    return DynamicBudgetSelection(
        selected_sensors=selected,
        available_sensors=available,
        rejected_sensors=rejected,
        total_cost=used_cost,
        budget=float(budget),
        selection_scores=scored,
        selection_reason=reason,
    )


def build_sensor_ranking(valid_sensors: Iterable[str],
                         sensor_informativeness: Mapping[str, float],
                         sensor_costs: Mapping[str, SensorCost],
                         ) -> list[dict]:
    """Return the static information-per-cost ranking of all valid sensors.

    Used for reporting only; the actual selection is performed
    dynamically at each timestamp by :func:`select_sensors_at_time`.
    """
    rows: list[dict] = []
    for sid in valid_sensors:
        sid = str(sid)
        d = sensor_informativeness.get(sid)
        cost = sensor_costs.get(sid) or SensorCost(sensor_id=sid)
        total = cost.total_cost
        if (d is None or not _is_finite(d)
                or total <= 0 or not _is_finite(total)):
            score = None
        else:
            score = float(d) / float(total)
        rows.append({
            "sensor_id": sid,
            "D_i": (float(d) if d is not None and _is_finite(d) else None),
            "sensing_cost": float(cost.sensing_cost),
            "communication_cost": float(cost.communication_cost),
            "total_cost": float(total),
            "score": score,
        })
    rows.sort(key=lambda r: (r["score"] is None,
                             -(r["score"] if r["score"] is not None else 0.0)))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


# ---------------------------------------------------------------------------
# Legacy preliminary static helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class BudgetSelection:
    """Result of the *preliminary* static budget approximation.

    .. deprecated::
        Kept only for backward compatibility. The main experiment uses
        the dynamic per-timestamp policy in
        :func:`select_sensors_at_time`.
    """

    selected_sensors: list[str]
    budget_regime: str
    selection_reason: str
    ranking_used: list[dict]
    unit_cost_assumption: bool
    k: int

    def to_dict(self) -> dict:
        return {
            "selected_sensors": list(self.selected_sensors),
            "budget_regime": self.budget_regime,
            "selection_reason": self.selection_reason,
            "ranking_used": self.ranking_used,
            "unit_cost_assumption": self.unit_cost_assumption,
            "k": self.k,
        }


def select_sensors_by_budget(valid_sensors: Iterable[str],
                             sensor_informativeness: list[dict],
                             regime: str,
                             k: int = 3,
                             ) -> BudgetSelection:
    """Static top-`k` selection (preliminary diagnostic helper).

    .. deprecated::
        Use :func:`select_sensors_at_time` inside a per-timestamp loop
        for the corrected dynamic budget policy. This helper picks a
        fixed subset once and is not the project's main budget
        implementation.
    """
    regime = regime.lower()
    if regime not in {"low", "medium", "high"}:
        raise ValueError(f"Unknown budget regime: {regime!r}")
    info_map = {
        str(rec["sensor_id"]): rec.get("D_i")
        for rec in sensor_informativeness
    }
    valid_list = [str(s) for s in valid_sensors]
    ranking: list[dict] = []
    for sid in valid_list:
        d = info_map.get(sid)
        ranking.append({
            "sensor_id": sid,
            "D_i": d,
            "score": d if d is not None else float("-inf"),
        })
    ranking.sort(key=lambda r: r["score"], reverse=True)
    for rank, entry in enumerate(ranking, start=1):
        entry["rank"] = rank

    if regime == "low":
        n = 1 if ranking else 0
    elif regime == "medium":
        n = max(0, min(k, len(ranking)))
    else:
        n = len(ranking)
    selected = [r["sensor_id"] for r in ranking[:n]]
    reason = (
        f"Preliminary static {regime}-budget selection: top-{n} sensors "
        f"by D_i out of {len(valid_list)} valid sensors. This is a "
        "diagnostic approximation; the main experiment uses the "
        "dynamic per-timestamp policy."
    )
    return BudgetSelection(
        selected_sensors=selected,
        budget_regime=regime,
        selection_reason=reason,
        ranking_used=ranking,
        unit_cost_assumption=True,
        k=k,
    )
