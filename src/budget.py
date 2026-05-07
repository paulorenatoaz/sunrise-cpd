"""Two-budget dynamic sensor selection policy.

The Sunrise CPD experiment always operates on the same multi-sensor
network. The budget policy enforces *two* separate resource constraints
that mirror the paper:

    sum_{i in S(t)} C_i  <=  B    (sensing/acquisition budget)
    sum_{i in S(t)} T_i  <=  C    (transmission/communication budget)

The active subset ``S(t)`` is selected dynamically at every timestamp.

Local/reference vs. cooperative sensors
---------------------------------------
At each timestamp ``t``, the first sensor selected by the greedy ranking
is treated as the local/reference sensor. It consumes sensing cost but
its *effective* transmission cost is zero because the local sensor does
not need to transmit to itself. Each additional selected sensor is a
cooperative/remote sensor that consumes both sensing cost and
transmission cost. The local/reference sensor is *not* fixed globally:
it depends on which sensors are available and on their D_i ranking at
that timestamp.

Cost model (this step: homogeneous synthetic costs)
---------------------------------------------------
Because the SensorScope dataset does not provide real sensing or
communication costs, every sensor uses the explicit homogeneous
synthetic costs

    C_i = 1.0,  T_i = 1.0,  cost_source = "homogeneous_synthetic_cost_assumption".

The local/reference sensor receives an effective transmission cost of
``0.0`` only inside :func:`select_sensors_at_time`; the persisted
sensor cost record always stores ``T_i = 1.0``.

Budget regimes
--------------
    low    -> B = 1.0,                   C = 0.0
    medium -> B = 3.0,                   C = 2.0
    high   -> B = number_of_valid,       C = number_of_valid - 1
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
DEFAULT_TRANSMISSION_COST = 1.0
DEFAULT_COST_SOURCE = "homogeneous_synthetic_cost_assumption"

SENSOR_COSTS_JSON = paths.JSON_DIR / "sensor_costs.json"


@dataclass
class SensorCost:
    """Per-sensor sensing and transmission cost.

    The transmission cost stored here is the *nominal* per-sensor cost.
    Whether the cost is paid in full at a given timestamp depends on the
    sensor's role at that timestamp (local/reference vs. cooperative).
    """

    sensor_id: str
    sensing_cost: float = DEFAULT_SENSING_COST
    transmission_cost: float = DEFAULT_TRANSMISSION_COST
    cost_source: str = DEFAULT_COST_SOURCE

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "sensing_cost": float(self.sensing_cost),
            "transmission_cost": float(self.transmission_cost),
            "cost_source": self.cost_source,
        }


def default_sensor_costs(valid_sensors: Iterable[str]
                         ) -> dict[str, SensorCost]:
    """Return default homogeneous-cost ``SensorCost`` records."""
    return {
        str(sid): SensorCost(sensor_id=str(sid))
        for sid in valid_sensors
    }


def write_sensor_costs(costs: Mapping[str, SensorCost],
                       out_path: Path = SENSOR_COSTS_JSON) -> Path:
    """Persist a sensor cost map to JSON."""
    paths.ensure_dirs()
    payload = {
        "cost_model": "homogeneous_synthetic_cost_assumption",
        "cost_model_description": (
            "Homogeneous synthetic per-sensor costs because the "
            "SensorScope dataset does not provide measured sensing or "
            "transmission costs. Every sensor uses C_i = 1.0 and "
            "T_i = 1.0. The local/reference sensor's effective "
            "transmission cost is set to 0.0 only dynamically inside "
            "the per-timestamp selection step; the stored T_i is the "
            "nominal cost incurred when the sensor is used as a "
            "cooperative sensor."
        ),
        "default_sensing_cost": DEFAULT_SENSING_COST,
        "default_transmission_cost": DEFAULT_TRANSMISSION_COST,
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
        # Accept the legacy "communication_cost" field name as a
        # fallback so old JSONs still load.
        t_cost = rec.get("transmission_cost",
                         rec.get("communication_cost",
                                 DEFAULT_TRANSMISSION_COST))
        out[sid] = SensorCost(
            sensor_id=sid,
            sensing_cost=float(rec.get("sensing_cost",
                                       DEFAULT_SENSING_COST)),
            transmission_cost=float(t_cost),
            cost_source=str(rec.get("cost_source", DEFAULT_COST_SOURCE)),
        )
    return out


# ---------------------------------------------------------------------------
# Budget policy configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetPolicyConfig:
    """Configuration of the two-budget dynamic sampling policy."""

    regime: str
    sensing_budget: float
    transmission_budget: float
    unit_cost_assumption: bool = False
    homogeneous_cost_assumption: bool = True
    score_name: str = "D_i"

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "sensing_budget": float(self.sensing_budget),
            "transmission_budget": float(self.transmission_budget),
            "unit_cost_assumption": bool(self.unit_cost_assumption),
            "homogeneous_cost_assumption": bool(
                self.homogeneous_cost_assumption),
            "score_name": self.score_name,
        }


def regime_budgets(regime: str, n_valid_sensors: int) -> tuple[float, float]:
    """Return ``(sensing_budget, transmission_budget)`` for a regime.

        low    -> (1.0, 0.0)
        medium -> (3.0, 2.0)
        high   -> (n_valid_sensors, n_valid_sensors - 1)
    """
    regime = regime.lower()
    n = max(int(n_valid_sensors), 1)
    if regime == "low":
        return 1.0, 0.0
    if regime == "medium":
        return 3.0, 2.0
    if regime == "high":
        return float(n), float(max(n - 1, 0))
    raise ValueError(f"Unknown regime: {regime!r}")


# ---------------------------------------------------------------------------
# Dynamic per-timestamp selection
# ---------------------------------------------------------------------------

@dataclass
class DynamicBudgetSelection:
    """Result of applying the dynamic two-budget policy at a single time."""

    timestamp: object = None
    available_sensors: list[str] = field(default_factory=list)
    selected_sensors: list[str] = field(default_factory=list)
    local_sensor: str | None = None
    cooperative_sensors: list[str] = field(default_factory=list)
    rejected_sensors: list[str] = field(default_factory=list)
    sensing_cost_used: float = 0.0
    transmission_cost_used: float = 0.0
    sensing_budget: float = 0.0
    transmission_budget: float = 0.0
    selection_scores: list[dict] = field(default_factory=list)
    selection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": (str(self.timestamp)
                          if self.timestamp is not None else None),
            "available_sensors": list(self.available_sensors),
            "selected_sensors": list(self.selected_sensors),
            "local_sensor": self.local_sensor,
            "cooperative_sensors": list(self.cooperative_sensors),
            "rejected_sensors": list(self.rejected_sensors),
            "sensing_cost_used": float(self.sensing_cost_used),
            "transmission_cost_used": float(self.transmission_cost_used),
            "sensing_budget": float(self.sensing_budget),
            "transmission_budget": float(self.transmission_budget),
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
                           sensing_budget: float,
                           transmission_budget: float,
                           timestamp=None,
                           ) -> DynamicBudgetSelection:
    """Greedy two-budget selection at a single timestamp.

    Sensors are ranked by ``D_i`` (which equals ``D_i / (C_i + T_i)``
    under the homogeneous synthetic cost assumption). The first
    selected sensor is the local/reference sensor at this timestamp and
    pays an effective transmission cost of ``0``; subsequent sensors
    pay the full transmission cost. Both constraints

        sum_{i in S(t)} C_i  <=  sensing_budget
        sum_{i in S(t)} T_i_eff  <=  transmission_budget

    are enforced separately.
    """
    available = [str(s) for s in available_sensors]
    scored: list[dict] = []
    for sid in available:
        d = sensor_informativeness.get(sid)
        cost = sensor_costs.get(sid) or SensorCost(sensor_id=sid)
        d_val = float(d) if d is not None and _is_finite(d) else None
        score = d_val if d_val is not None else float("-inf")
        scored.append({
            "sensor_id": sid,
            "D_i": d_val,
            "sensing_cost": float(cost.sensing_cost),
            "transmission_cost": float(cost.transmission_cost),
            "score": score,
        })
    scored.sort(key=lambda r: r["score"], reverse=True)

    selected: list[str] = []
    cooperative: list[str] = []
    local_sensor: str | None = None
    sensing_used = 0.0
    transmission_used = 0.0
    rejected: list[str] = []
    tol = 1e-12

    for entry in scored:
        if entry["score"] == float("-inf"):
            rejected.append(entry["sensor_id"])
            continue
        c_i = entry["sensing_cost"]
        # First selectable sensor is the local/reference sensor and
        # pays no transmission cost. Subsequent sensors are cooperative.
        if local_sensor is None:
            t_eff = 0.0
            new_sense = sensing_used + c_i
            new_trans = transmission_used + t_eff
            if (new_sense <= sensing_budget + tol
                    and new_trans <= transmission_budget + tol):
                selected.append(entry["sensor_id"])
                local_sensor = entry["sensor_id"]
                sensing_used = new_sense
                transmission_used = new_trans
                entry["effective_transmission_cost"] = t_eff
                entry["role"] = "local"
            else:
                rejected.append(entry["sensor_id"])
        else:
            t_eff = entry["transmission_cost"]
            new_sense = sensing_used + c_i
            new_trans = transmission_used + t_eff
            if (new_sense <= sensing_budget + tol
                    and new_trans <= transmission_budget + tol):
                selected.append(entry["sensor_id"])
                cooperative.append(entry["sensor_id"])
                sensing_used = new_sense
                transmission_used = new_trans
                entry["effective_transmission_cost"] = t_eff
                entry["role"] = "cooperative"
            else:
                rejected.append(entry["sensor_id"])

    if selected:
        reason = (
            f"Greedy two-budget selection: chose {len(selected)}/"
            f"{len(available)} available sensors "
            f"(1 local + {len(cooperative)} cooperative); "
            f"sensing cost {sensing_used:.3f}/{sensing_budget:.3f}, "
            f"transmission cost {transmission_used:.3f}/"
            f"{transmission_budget:.3f}."
        )
    else:
        reason = (
            f"No sensors selected (available={len(available)}, "
            f"sensing_budget={sensing_budget:.3f}, "
            f"transmission_budget={transmission_budget:.3f})."
        )

    return DynamicBudgetSelection(
        timestamp=timestamp,
        available_sensors=available,
        selected_sensors=selected,
        local_sensor=local_sensor,
        cooperative_sensors=cooperative,
        rejected_sensors=rejected,
        sensing_cost_used=sensing_used,
        transmission_cost_used=transmission_used,
        sensing_budget=float(sensing_budget),
        transmission_budget=float(transmission_budget),
        selection_scores=scored,
        selection_reason=reason,
    )


def build_sensor_ranking(valid_sensors: Iterable[str],
                         sensor_informativeness: Mapping[str, float],
                         sensor_costs: Mapping[str, SensorCost],
                         ) -> list[dict]:
    """Return the static D_i ranking of all valid sensors.

    Used for reporting only; the actual selection is performed
    dynamically at each timestamp by :func:`select_sensors_at_time`.
    Under the homogeneous synthetic cost assumption (``C_i = T_i = 1``),
    ranking by ``D_i`` is equivalent to ranking by information per cost.
    """
    rows: list[dict] = []
    for sid in valid_sensors:
        sid = str(sid)
        d = sensor_informativeness.get(sid)
        cost = sensor_costs.get(sid) or SensorCost(sensor_id=sid)
        d_val = float(d) if d is not None and _is_finite(d) else None
        rows.append({
            "sensor_id": sid,
            "D_i": d_val,
            "sensing_cost": float(cost.sensing_cost),
            "transmission_cost": float(cost.transmission_cost),
            "score": d_val,
        })
    rows.sort(key=lambda r: (r["score"] is None,
                             -(r["score"] if r["score"] is not None else 0.0)))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
