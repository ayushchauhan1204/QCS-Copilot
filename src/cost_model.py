"""
cost_model.py — Illustrative broke/downtime cost translation.

Turns off-spec risk and stabilization time into an estimated dollar figure,
so a non-technical reviewer can grasp business impact at a glance without
doing unit conversion in their head.

IMPORTANT: the price/width constants below are ILLUSTRATIVE INDUSTRY-TYPICAL
ASSUMPTIONS, not real mill data. They are deliberately exposed as adjustable
parameters (see dashboard/app.py's sidebar) specifically so this is never
presented as more precise than it is — a real deployment would plug in the
mill's actual price and web width.

CONSTITUTION RULES OBSERVED:
  Rule 4 — Dashboard contains only UI code; this pure-function logic lives here.
"""

from __future__ import annotations

# Illustrative defaults — NOT real mill data. Typical containerboard/kraft
# paper price and a typical web width, used only so the estimate has a
# believable order of magnitude out of the box.
DEFAULT_PRICE_PER_TON_USD: float = 800.0
DEFAULT_WEB_WIDTH_M: float = 5.0


def estimate_broke_cost(
    basis_weight_gsm: float,
    machine_speed_m_per_min: float,
    duration_seconds: float,
    price_per_ton_usd: float = DEFAULT_PRICE_PER_TON_USD,
    web_width_m: float = DEFAULT_WEB_WIDTH_M,
) -> float:
    """Estimate the dollar cost of producing off-spec paper for a given duration.

    Args:
        basis_weight_gsm: Paper basis weight (g/m^2) during the window.
        machine_speed_m_per_min: Machine speed (m/min) during the window.
        duration_seconds: How long the process is assumed to run off-spec (s).
            Pass 0 (or a non-positive value) for "no cost" — the function
            returns 0.0 rather than a negative or nonsensical figure.
        price_per_ton_usd: Assumed price per metric ton of finished paper.
        web_width_m: Assumed web width in meters.

    Returns:
        Estimated cost in USD of scrapping that duration's production as broke.
    """
    if duration_seconds <= 0:
        return 0.0
    production_kg_per_min = basis_weight_gsm * machine_speed_m_per_min * web_width_m / 1000.0
    production_kg = production_kg_per_min * (duration_seconds / 60.0)
    return production_kg / 1000.0 * price_per_ton_usd
