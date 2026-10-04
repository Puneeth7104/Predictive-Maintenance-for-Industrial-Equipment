"""Create synthetic, run-to-failure sensor data for a fleet of machines.

Each machine runs in cycles. A cycle ends either with a *failure* (repaired
afterwards) or, sometimes, with a *preventive* maintenance stop. In the
hours before a failure the sensors drift away from normal (temperature and
vibration rise, pressure drops). A few failures are "sudden" (very short
warning) so the problem is not trivially easy.
"""
import argparse

import numpy as np
import pandas as pd

from .config import DATA_DIR, N_DAYS, N_MACHINES, SEED, START


def _new_cycle(rng):
    """Draw the length of the next cycle, its warning ramp and (maybe) a preventive stop."""
    next_fail = int(rng.uniform(350, 1000))                  # hours of life in this cycle
    ramp = 10 if rng.random() < 0.2 else int(rng.choice([48, 72, 96]))
    preventive_at = None
    if rng.random() < 0.25:                                  # 25 % of cycles end with a planned stop
        lo = 0.5 * max(next_fail - ramp, 20)
        hi = max(next_fail - ramp - 1, lo + 1)
        preventive_at = int(rng.uniform(lo, hi))
    return next_fail, ramp, preventive_at


def simulate_machine(machine_id, rng, n_hours, start):
    machine_type = str(rng.choice(["Pump", "Compressor", "Motor"]))
    base_temp = rng.normal(70, 3)
    base_vib = rng.normal(2.0, 0.15)
    base_pres = rng.normal(100, 4)

    rows, log = [], []
    t = hours_since = maint_count = op_hours = 0
    next_fail, ramp, prev_at = _new_cycle(rng)

    while t < n_hours:
        ts = start + pd.Timedelta(hours=t)
        ttf = next_fail - hours_since                         # hours left until failure
        d = 0.0 if ttf > ramp else ((ramp - ttf) / ramp) ** 1.5   # degradation 0..1
        load = 0.6 + 0.25 * np.sin(2 * np.pi * (t % 24) / 24) + rng.normal(0, 0.05)

        temp = base_temp + 6 * load + 22 * d + rng.normal(0, 1.2)
        vib = base_vib + 0.4 * load + 3.0 * d + abs(rng.normal(0, 0.12))
        if rng.random() < 0.01:                               # harmless random spikes
            vib += rng.uniform(0.3, 0.8)
        pres = base_pres - 3 * load - 14 * d + rng.normal(0, 1.5)

        reading = {"temperature": temp, "vibration": vib, "pressure": pres}
        for k in reading:                                     # occasional missing sensor value
            if rng.random() < 0.005:
                reading[k] = np.nan

        rows.append({
            "timestamp": ts, "machine_id": machine_id, "machine_type": machine_type,
            **{k: round(v, 3) for k, v in reading.items()},
            "operating_hours": op_hours,
            "hours_since_maintenance": hours_since,
            "maintenance_count": maint_count,
        })
        t += 1
        hours_since += 1
        op_hours += 1

        if hours_since >= next_fail:                          # failure + repair
            log.append({"timestamp": start + pd.Timedelta(hours=t),
                        "machine_id": machine_id, "event": "failure"})
            t += int(rng.integers(8, 17))
            hours_since, maint_count = 0, maint_count + 1
            next_fail, ramp, prev_at = _new_cycle(rng)
        elif prev_at is not None and hours_since >= prev_at:  # planned maintenance
            log.append({"timestamp": start + pd.Timedelta(hours=t),
                        "machine_id": machine_id, "event": "preventive"})
            t += 4
            hours_since, maint_count = 0, maint_count + 1
            next_fail, ramp, prev_at = _new_cycle(rng)
    return rows, log


def generate(n_machines=N_MACHINES, n_days=N_DAYS, seed=SEED):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(START)
    all_rows, all_log = [], []
    for i in range(1, n_machines + 1):
        rows, log = simulate_machine(f"M{i:02d}", rng, n_days * 24, start)
        all_rows += rows
        all_log += log
    sensors = pd.DataFrame(all_rows).sort_values(["timestamp", "machine_id"]).reset_index(drop=True)
    maintenance = pd.DataFrame(all_log).sort_values("timestamp").reset_index(drop=True)
    return sensors, maintenance


def main(n_machines=N_MACHINES, n_days=N_DAYS, seed=SEED):
    DATA_DIR.mkdir(exist_ok=True)
    sensors, maintenance = generate(n_machines, n_days, seed)
    sensors.to_csv(DATA_DIR / "sensor_data.csv", index=False)
    maintenance.to_csv(DATA_DIR / "maintenance_log.csv", index=False)
    n_fail = int((maintenance["event"] == "failure").sum())
    print(f"[data] {len(sensors):,} sensor rows, {n_machines} machines, {n_fail} failures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", type=int, default=N_MACHINES)
    ap.add_argument("--days", type=int, default=N_DAYS)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    main(a.machines, a.days, a.seed)
