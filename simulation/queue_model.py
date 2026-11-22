"""
GridMind AI — Queue and Swap Dynamics Model
============================================
Models charging session queues and swap-pack fulfillment as discrete
inventory management problems. These metrics feed into reliability
calculations and loss-of-load probability estimates.

Key design decisions:
- Queue length and abandonment are modeled hourly using an M/D/C approximation
  (Poisson arrivals, deterministic service, C servers = charging bays).
- Swap fulfilment uses discrete pack counting \u2014 a request is either served
  (pack available) or deferred/abandoned.
- Pack returns are probabilistically modeled based on typical EV usage duration.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.assumptions import (
    CLUSTERS, AVG_KWH_PER_SESSION, MAX_QUEUE_WAIT_MIN,
    ABANDONMENT_RATE_PER_MIN, PACK_CAPACITY_KWH,
    PACK_RESERVE_FRACTION, CLUSTER_SWAP_PACKS,
)


def simulate_charging_queue(
    arrivals_per_hour: np.ndarray,  # shape (T,) - sessions arriving per hour
    capacity_mw: float,              # Available charging capacity (MW)
    avg_kwh_session: float = AVG_KWH_PER_SESSION,
    dt_hours: float = 1.0,
) -> dict:
    """
    Simulate charging queue dynamics over a horizon.

    Service capacity in sessions/hour:
      capacity_sessions = capacity_mw * 1000 / avg_kwh_session

    Queue update (discrete-time):
      Queue[t+1] = max(0, Queue[t] + Arrivals[t] - Served[t] - Abandoned[t])

    Returns arrays of queue length, served sessions, abandoned sessions, unserved energy.
    """
    T = len(arrivals_per_hour)
    capacity_sessions_hr = capacity_mw * 1000.0 / avg_kwh_session

    queue = np.zeros(T + 1)
    served = np.zeros(T)
    abandoned = np.zeros(T)
    unserved_energy_mwh = np.zeros(T)

    for t in range(T):
        # Total waiting = carry-over queue + new arrivals
        total_waiting = queue[t] + arrivals_per_hour[t]
        # Service capacity this hour
        can_serve = capacity_sessions_hr * dt_hours
        # Sessions served
        sessions_served = min(total_waiting, can_serve)
        # Excess waiting
        excess = max(0, total_waiting - sessions_served)
        # Abandonment: waiting sessions abandon at a rate based on expected wait
        avg_wait_min = (excess / max(capacity_sessions_hr, 1)) * 60 * dt_hours
        abandon_fraction = min(0.95, ABANDONMENT_RATE_PER_MIN * max(0, avg_wait_min - 10))
        sessions_abandoned = excess * abandon_fraction
        queue[t + 1] = max(0, excess - sessions_abandoned)

        served[t] = sessions_served
        abandoned[t] = sessions_abandoned
        unserved_energy_mwh[t] = sessions_abandoned * avg_kwh_session / 1000.0

    return {
        "queue_length": queue[:-1],
        "sessions_served": served,
        "sessions_abandoned": abandoned,
        "unserved_energy_mwh": unserved_energy_mwh,
        "total_served_energy_mwh": served * avg_kwh_session / 1000.0,
    }


def simulate_swap_inventory(
    swap_requests: np.ndarray,      # shape (T,) - pack requests per hour
    n_packs_initial: int,            # Initial charged pack count
    n_packs_total: int,              # Total fleet pack inventory
    return_rate: float = 0.85,       # Fraction of issued packs returned per hour
    charge_rate_packs_hr: float = None,  # Packs that can be charged per hour
) -> dict:
    """
    Simulate swap pack inventory dynamics.

    Pack inventory transition (hourly):
      Charged[t+1] = Charged[t] - Issued[t] + Recharged[t]

    Recharging: empty packs are recharged at the given rate.
    Reserve: 5% of total packs are held as priority reserve.

    Returns arrays for charged packs, fulfilment rate, stock-outs.
    """
    T = len(swap_requests)
    if charge_rate_packs_hr is None:
        charge_rate_packs_hr = n_packs_total * 0.05  # Charge 5% of fleet per hour

    reserve_packs = int(n_packs_total * PACK_RESERVE_FRACTION)

    charged_packs = np.zeros(T + 1)
    empty_packs = np.zeros(T + 1)
    issued = np.zeros(T, dtype=int)
    unfulfilled = np.zeros(T, dtype=int)
    recharged = np.zeros(T, dtype=int)

    charged_packs[0] = n_packs_initial
    empty_packs[0] = n_packs_total - n_packs_initial

    for t in range(T):
        # Available packs (above reserve threshold)
        available = max(0, int(charged_packs[t]) - reserve_packs)
        # Fulfill as many requests as possible
        requests = int(swap_requests[t])
        can_fulfill = min(requests, available)
        cant_fulfill = requests - can_fulfill

        issued[t] = can_fulfill
        unfulfilled[t] = cant_fulfill

        # Returns: some fraction of issued packs returned as empties
        returned_this_hour = int(issued[t] * return_rate)
        empty_now = empty_packs[t] + can_fulfill  # newly empty from issues
        # Recharge as many empty packs as the charging infrastructure allows
        packs_to_recharge = min(int(empty_now), int(charge_rate_packs_hr))
        recharged[t] = packs_to_recharge

        charged_packs[t + 1] = charged_packs[t] - can_fulfill + packs_to_recharge + returned_this_hour
        empty_packs[t + 1] = max(0, empty_now - packs_to_recharge - returned_this_hour)

        # Clamp to physical bounds
        total = charged_packs[t + 1] + empty_packs[t + 1]
        if total > n_packs_total:
            overflow = total - n_packs_total
            charged_packs[t + 1] -= overflow / 2
            empty_packs[t + 1] -= overflow / 2
        charged_packs[t + 1] = max(reserve_packs, charged_packs[t + 1])
        empty_packs[t + 1] = max(0, empty_packs[t + 1])

    energy_per_swap_mwh = PACK_CAPACITY_KWH / 1000.0
    stock_out = (issued < swap_requests.astype(int)).astype(int)

    return {
        "charged_packs": charged_packs[:-1],
        "empty_packs": empty_packs[:-1],
        "packs_issued": issued,
        "packs_unfulfilled": unfulfilled,
        "packs_recharged": recharged,
        "stock_outs": stock_out,
        "energy_served_mwh": issued * energy_per_swap_mwh,
        "energy_missed_mwh": unfulfilled * energy_per_swap_mwh,
        "fulfilment_rate": issued / (swap_requests + 1e-6),
    }
