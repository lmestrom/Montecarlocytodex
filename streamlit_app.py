# streamlit_app.py
# Seed Train Monte Carlo (Cytodex surface model)
# Fix: Day-0 comes from the model (no forced inoc point -> no artificial drops)
# Fix: simulate_surface_mc trajectories stored correctly per metric
# Option 1: Infectable = OCCUPIED + NEWLY_OCCUPIED + INHIBITED (MULTILAYER non-infectable)

import io
import math
import random
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# Cell states
# ----------------------------
UNOCCUPIED = 0
OCCUPIED = 1
NEWLY_OCCUPIED = 2
INHIBITED = 3
MULTILAYER = 4


# ----------------------------
# Core grid model (surface-only)
# ----------------------------
def update_state(grid: np.ndarray, multilayer_cycle_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One growth step on a periodic grid."""
    grid_size = grid.shape[0]
    new_grid = np.copy(grid)

    for i in range(grid_size):
        for j in range(grid_size):

            # growth attempt
            if grid[i, j] == OCCUPIED:
                neighbors = [
                    (x % grid_size, y % grid_size)
                    for x in range(i - 1, i + 2)
                    for y in range(j - 1, j + 2)
                    if (x, y) != (i, j)
                ]
                random.shuffle(neighbors)

                placed = False
                for x, y in neighbors:
                    if new_grid[x, y] == UNOCCUPIED:
                        new_grid[x, y] = NEWLY_OCCUPIED
                        placed = True
                        break

                if not placed:
                    new_grid[i, j] = INHIBITED

            # multilayer rule (only triggered from inhibited)
            if new_grid[i, j] == INHIBITED:
                neighbors = [
                    (x % grid_size, y % grid_size)
                    for x in range(i - 1, i + 2)
                    for y in range(j - 1, j + 2)
                    if (x, y) != (i, j)
                ]
                if any(new_grid[x, y] in (INHIBITED, MULTILAYER) for x, y in neighbors):
                    multilayer_cycle_grid[i, j] += 1
                    if multilayer_cycle_grid[i, j] >= 1:
                        new_grid[i, j] = MULTILAYER
                        multilayer_cycle_grid[i, j] = 0
                else:
                    multilayer_cycle_grid[i, j] = 0

    return new_grid, multilayer_cycle_grid


def count_states(grid: np.ndarray) -> Dict[str, int]:
    uno = int(np.count_nonzero(grid == UNOCCUPIED))
    occ = int(np.count_nonzero(grid == OCCUPIED))
    new = int(np.count_nonzero(grid == NEWLY_OCCUPIED))
    inh = int(np.count_nonzero(grid == INHIBITED))
    mul = int(np.count_nonzero(grid == MULTILAYER))

    occupied_total = occ + new + inh + mul
    infectable_total = occ + new + inh  # Option 1
    total = uno + occ + new + inh + mul

    return {
        "UNOCCUPIED": uno,
        "OCCUPIED": occ,
        "NEWLY_OCCUPIED": new,
        "INHIBITED": inh,
        "MULTILAYER": mul,
        "OCCUPIED_TOTAL": occupied_total,
        "INFECTABLE_TOTAL": infectable_total,
        "TOTAL": total,
    }


# ----------------------------
# Monte Carlo simulation (includes Day-0)  ✅ FIXED
# ----------------------------
def simulate_surface_mc(
    n_experiments: int,
    days: int,
    max_cells_setpoint: float,
    max_cells_sd: float,
    inoc_cells_per_mc_mean: float,
    inoc_cells_per_mc_sd: float,
    rng_seed: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte Carlo simulation of cells/MC over time (surface model).
    RETURNS days+1 points: day0..dayN
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    metrics = [
        "UNOCCUPIED",
        "OCCUPIED",
        "NEWLY_OCCUPIED",
        "INHIBITED",
        "MULTILAYER",
        "OCCUPIED_TOTAL",
        "INFECTABLE_TOTAL",
    ]

    traj_by_metric = {m: [] for m in metrics}

    for _ in range(n_experiments):
        max_cells = float(max(1.0, np.random.normal(max_cells_setpoint, max_cells_sd)))
        grid_size = max(2, int(round(math.sqrt(max_cells))))

        inoc_cells = float(max(0.0, np.random.normal(inoc_cells_per_mc_mean, inoc_cells_per_mc_sd)))
        inoc_density = min(1.0, inoc_cells / float(grid_size * grid_size))  # cap at surface capacity

        grid = np.zeros((grid_size, grid_size), dtype=int)
        multilayer_cycle_grid = np.zeros((grid_size, grid_size), dtype=int)

        initial_occupied_cells = int(round(grid_size * grid_size * inoc_density))
        initial_occupied_cells = min(max(initial_occupied_cells, 0), grid_size * grid_size)

        if initial_occupied_cells > 0:
            initial_positions = random.sample(range(grid_size * grid_size), initial_occupied_cells)
            for pos in initial_positions:
                x, y = divmod(pos, grid_size)
                grid[x, y] = OCCUPIED

        # ---- Day 0 counts
        counts0 = count_states(grid)
        traj = {m: [float(counts0[m])] for m in metrics}

        prev_total = counts0["OCCUPIED_TOTAL"]

        # ---- Days 1..N
        for _day in range(days):
            grid, multilayer_cycle_grid = update_state(grid, multilayer_cycle_grid)
            counts = count_states(grid)

            # model has no death; should be monotone within stage
            if counts["OCCUPIED_TOTAL"] < prev_total:
                raise RuntimeError("BUG: OCCUPIED_TOTAL decreased within a stage.")
            prev_total = counts["OCCUPIED_TOTAL"]

            for m in metrics:
                traj[m].append(float(counts[m]))

            # commit NEWLY_OCCUPIED to OCCUPIED for next step
            grid[grid == NEWLY_OCCUPIED] = OCCUPIED

        # store per-metric trajectory arrays (each must be length days+1)
        for m in metrics:
            traj_by_metric[m].append(np.array(traj[m], dtype=float))

    mean_dict, std_dict = {}, {}
    for m in metrics:
        arr = np.vstack(traj_by_metric[m])  # (n, days+1)
        mean_dict[m] = arr.mean(axis=0)
        std_dict[m] = arr.std(axis=0, ddof=1) if n_experiments > 1 else np.zeros(arr.shape[1], dtype=float)

    return pd.DataFrame(mean_dict), pd.DataFrame(std_dict)


# ----------------------------
# Unit conversions
# ----------------------------
def mc_to_cells_ml(mean_cells_per_mc: np.ndarray, std_cells_per_mc: np.ndarray, mc_g_per_l: float, particles_per_g: float):
    """cells/mL = (mc_g/L * particles/g * cells/particle) / 1000"""
    cells_ml = (mc_g_per_l * particles_per_g * mean_cells_per_mc) / 1000.0
    cells_ml_std = (mc_g_per_l * particles_per_g * std_cells_per_mc) / 1000.0
    return cells_ml, cells_ml_std


def cells_ml_to_cells_per_mc(inoc_cells_ml: float, mc_g_per_l: float, particles_per_g: float) -> float:
    """
    cells/mL -> cells/MC.
    g/mL = mc_g_per_l / 1000
    cells/g = (cells/mL) / (g/mL) = cells/mL * 1000 / (mc_g/L)
    cells/MC = (cells/g) / (particles/g)
    """
    if mc_g_per_l <= 0 or particles_per_g <= 0:
        raise ValueError("MC g/L and particles/g must be > 0.")
    cells_per_g = (inoc_cells_ml * 1000.0) / mc_g_per_l
    return cells_per_g / particles_per_g


# ----------------------------
# Stage runner
# ----------------------------
@dataclass
class StageResult:
    name: str
    t_h: np.ndarray
    total_cells_ml: np.ndarray
    total_cells_ml_std: np.ndarray
    infectable_cells_ml: np.ndarray
    infectable_cells_ml_std: np.ndarray
    multilayer_cells_ml: np.ndarray
    multilayer_cells_ml_std: np.ndarray
    inoc_cells_ml_requested: float
    inoc_cells_ml_effective: float
    end_total_cells_ml: float
    end_infectable_cells_ml: float
    days: int
    mc_g_per_l: float


def run_stage_surface_growth(
    name: str,
    n: int,
    days: int,
    mc_g_per_l: float,
    particles_per_g: float,
    max_cells_setpoint: float,
    max_cells_sd: float,
    inoc_cells_ml: float,
    inoc_sd_cells_per_mc: float,
    rng_seed: int,
) -> StageResult:
    inoc_cells_per_mc_mean = cells_ml_to_cells_per_mc(inoc_cells_ml, mc_g_per_l, particles_per_g)

    mean_df, std_df = simulate_surface_mc(
        n_experiments=n,
        days=days,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_mean,
        inoc_cells_per_mc_sd=inoc_sd_cells_per_mc,
        rng_seed=rng_seed,
    )

    t_h = np.arange(0, days + 1, dtype=float) * 24.0

    total_ml, total_ml_std = mc_to_cells_ml(
        mean_df["OCCUPIED_TOTAL"].to_numpy(),
        std_df["OCCUPIED_TOTAL"].to_numpy(),
        mc_g_per_l,
        particles_per_g,
    )
    infect_ml, infect_ml_std = mc_to_cells_ml(
        mean_df["INFECTABLE_TOTAL"].to_numpy(),
        std_df["INFECTABLE_TOTAL"].to_numpy(),
        mc_g_per_l,
        particles_per_g,
    )
    multi_ml, multi_ml_std = mc_to_cells_ml(
        mean_df["MULTILAYER"].to_numpy(),
        std_df["MULTILAYER"].to_numpy(),
        mc_g_per_l,
        particles_per_g,
    )

    inoc_effective_ml = float(total_ml[0])

    return StageResult(
        name=name,
        t_h=t_h,
        total_cells_ml=total_ml,
        total_cells_ml_std=total_ml_std,
        infectable_cells_ml=infect_ml,
        infectable_cells_ml_std=infect_ml_std,
        multilayer_cells_ml=multi_ml,
        multilayer_cells_ml_std=multi_ml_std,
        inoc_cells_ml_requested=float(inoc_cells_ml),
        inoc_cells_ml_effective=inoc_effective_ml,
        end_total_cells_ml=float(total_ml[-1]),
        end_infectable_cells_ml=float(infect_ml[-1]),
        days=int(days),
        mc_g_per_l=float(mc_g_per_l),
    )


# ----------------------------
# Seed train
# ----------------------------
def run_seed_train(params: dict):
    particles_per_g = float(params["particles_per_g"])

    V_BIO40 = 40000.0
    V_BIO200A = 200000.0
    V_BIO750 = 750000.0
    V_BIO1500 = 1500000.0

    MC_BIO40 = float(params["mc_bio40_g_per_l"])
    MC_BIO200A = float(params["mc_bio200a_g_per_l"])
    MC_BIO750 = float(params["mc_bio750_g_per_l"])

    d40 = int(params["days_bio40"])
    d200 = int(params["days_bio200a"])
    d750 = int(params["days_bio750"])
    n = int(params["n_experiments"])

    max_cells_setpoint = float(params["max_cells_setpoint"])
    max_cells_sd = float(params["max_cells_sd"])
    rng_seed = int(params["rng_seed"])

    # BIO40
    r40 = run_stage_surface_growth(
        name="BIO40",
        n=n,
        days=d40,
        mc_g_per_l=MC_BIO40,
        particles_per_g=particles_per_g,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_ml=float(params["inoc_bio40_cells_ml"]),
        inoc_sd_cells_per_mc=float(params["inoc_sd_cells_per_mc_bio40"]),
        rng_seed=rng_seed,
    )

    # Transfer to BIO200A (use END of BIO40 TOTAL, volume-based)
    tryps_yield_200 = float(params["trypsin_yield_bio200a"])
    bio200_inoc_ml = tryps_yield_200 * V_BIO40 * r40.end_total_cells_ml / V_BIO200A

    r200 = run_stage_surface_growth(
        name="BIO200A",
        n=n,
        days=d200,
        mc_g_per_l=MC_BIO200A,
        particles_per_g=particles_per_g,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_ml=bio200_inoc_ml,
        inoc_sd_cells_per_mc=float(params["inoc_sd_cells_per_mc_bio200a"]),
        rng_seed=rng_seed + 1,
    )

    # Split to BIO750 (use END of BIO200A TOTAL)
    bio200_end_total = r200.end_total_cells_ml
    tryps_yield_750 = float(params["trypsin_yield_bio750"])
    dist_factor_EF = float(params["distribution_factor_EF"])
    dist_factor_GH = float(params["distribution_factor_GH"])

    def bio750_inoc(dist_factor: float) -> float:
        return dist_factor * tryps_yield_750 * V_BIO200A * bio200_end_total / V_BIO750

    r750E = run_stage_surface_growth("BIO750E", n, d750, MC_BIO750, particles_per_g, max_cells_setpoint, max_cells_sd,
                                    bio750_inoc(dist_factor_EF), float(params["inoc_sd_cells_per_mc_bio750"]), rng_seed + 2)
    r750F = run_stage_surface_growth("BIO750F", n, d750, MC_BIO750, particles_per_g, max_cells_setpoint, max_cells_sd,
                                    bio750_inoc(dist_factor_EF), float(params["inoc_sd_cells_per_mc_bio750"]), rng_seed + 3)
    r750G = run_stage_surface_growth("BIO750G", n, d750, MC_BIO750, particles_per_g, max_cells_setpoint, max_cells_sd,
                                    bio750_inoc(dist_factor_GH), float(params["inoc_sd_cells_per_mc_bio750"]), rng_seed + 4)
    r750H = run_stage_surface_growth("BIO750H", n, d750, MC_BIO750, particles_per_g, max_cells_setpoint, max_cells_sd,
                                    bio750_inoc(dist_factor_GH), float(params["inoc_sd_cells_per_mc_bio750"]), rng_seed + 5)

    # Combine to BIO1500 (inoc only)
    transfer_1500A = float(params["transfer_bio1500a"])
    transfer_1500B = float(params["transfer_bio1500b"])

    inoc_1500A = transfer_1500A * (r750E.end_total_cells_ml * V_BIO750 + r750F.end_total_cells_ml * V_BIO750) / V_BIO1500
    inoc_1500B = transfer_1500B * (r750G.end_total_cells_ml * V_BIO750 + r750H.end_total_cells_ml * V_BIO750) / V_BIO1500

    def to_df(stage: StageResult) -> pd.DataFrame:
        return pd.DataFrame({
            "time_h": stage.t_h,
            "total_cells_ml": stage.total_cells_ml,
            "infectable_cells_ml": stage.infectable_cells_ml,
            "multilayer_cells_ml": stage.multilayer_cells_ml,
        })

    series = {
        "BIO40": to_df(r40),
        "BIO200A": to_df(r200),
        "BIO750E": to_df(r750E),
        "BIO750F": to_df(r750F),
        "BIO750G": to_df(r750G),
        "BIO750H": to_df(r750H),
        "BIO1500A": pd.DataFrame({"time_h": [0.0, 24.0], "total_cells_ml": [inoc_1500A, inoc_1500A],
                                  "infectable_cells_ml": [inoc_1500A, inoc_1500A], "multilayer_cells_ml": [0.0, 0.0]}),
        "BIO1500B": pd.DataFrame({"time_h": [0.0, 24.0], "total_cells_ml": [inoc_1500B, inoc_1500B],
                                  "infectable_cells_ml": [inoc_1500B, inoc_1500B], "multilayer_cells_ml": [0.0, 0.0]}),
    }

    # cumulative time alignment
    t0_40 = 0.0
    t0_200 = float(d40) * 24.0
    t0_750 = float(d40 + d200) * 24.0
    t0_1500 = t0_750 + float(d750) * 24.0

    series["BIO40"]["time_h_cum"] = series["BIO40"]["time_h"] + t0_40
    series["BIO200A"]["time_h_cum"] = series["BIO200A"]["time_h"] + t0_200
    for k in ["BIO750E", "BIO750F", "BIO750G", "BIO750H"]:
        series[k]["time_h_cum"] = series[k]["time_h"] + t0_750
    series["BIO1500A"]["time_h_cum"] = series["BIO1500A"]["time_h"] + t0_1500
    series["BIO1500B"]["time_h_cum"] = series["BIO1500B"]["time_h"] + t0_1500

    summary = pd.DataFrame([
        {"Stage": "BIO40", "Inoc_req_cells/mL": r40.inoc_cells_ml_requested, "Inoc_eff_cells/mL": r40.inoc_cells_ml_effective,
         "End_TOTAL_cells/mL": r40.end_total_cells_ml, "End_INFECTABLE_cells/mL": r40.end_infectable_cells_ml, "Days": d40, "MC_g/L": MC_BIO40},
        {"Stage": "BIO200A", "Inoc_req_cells/mL": bio200_inoc_ml, "Inoc_eff_cells/mL": r200.inoc_cells_ml_effective,
         "End_TOTAL_cells/mL": r200.end_total_cells_ml, "End_INFECTABLE_cells/mL": r200.end_infectable_cells_ml, "Days": d200, "MC_g/L": MC_BIO200A},
        {"Stage": "BIO750E", "Inoc_req_cells/mL": r750E.inoc_cells_ml_requested, "Inoc_eff_cells/mL": r750E.inoc_cells_ml_effective,
         "End_TOTAL_cells/mL": r750E.end_total_cells_ml, "End_INFECTABLE_cells/mL": r750E.end_infectable_cells_ml, "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO750F", "Inoc_req_cells/mL": r750F.inoc_cells_ml_requested, "Inoc_eff_cells/mL": r750F.inoc_cells_ml_effective,
         "End_TOTAL_cells/mL": r750F.end_total_cells_ml, "End_INFECTABLE_cells/mL": r750F.end_infectable_cells_ml, "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO750G", "Inoc_req_cells/mL": r750G.inoc_cells_ml_requested, "Inoc_eff_cells/mL": r750G.inoc_cells_ml_effective,
         "End_TOTAL_cells/mL": r750G.end_total_cells_ml, "End_INFECTABLE_cells/mL": r750G.end_infectable_cells_ml, "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO750H", "Inoc_req_cells/mL": r750H.inoc_cells_ml_requested, "Inoc_eff_cells/mL": r750H.inoc_cells_ml_effective,
         "End_TOTAL_cells/mL": r750H.end_total_cells_ml, "End_INFECTABLE_cells/mL": r750H.end_infectable_cells_ml, "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO1500A", "Inoc_req_cells/mL": inoc_1500A, "Inoc_eff_cells/mL": inoc_1500A,
         "End_TOTAL_cells/mL": inoc_1500A, "End_INFECTABLE_cells/mL": inoc_1500A, "Days": 0, "MC_g/L": np.nan},
        {"Stage": "BIO1500B", "Inoc_req_cells/mL": inoc_1500B, "Inoc_eff_cells/mL": inoc_1500B,
         "End_TOTAL_cells/mL": inoc_1500B, "End_INFECTABLE_cells/mL": inoc_1500B, "Days": 0, "MC_g/L": np.nan},
    ])

    return series, summary


# ----------------------------
# Plot + export
# ----------------------------
def plot_all_bioreactors_one_plot(series: Dict[str, pd.DataFrame], y_col: str, title: str):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    for name, df in series.items():
        ax.plot(df["time_h_cum"], df[y_col], marker="o", label=name)
    ax.set_xlabel("Time (h) — cumulative")
    ax.set_ylabel("Cells / mL")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    ax.set_title(title)
    return fig


def export_zip_csv(series: Dict[str, pd.DataFrame], summary: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", summary.to_csv(index=False))
        for k, df in series.items():
            zf.writestr(f"{k}.csv", df.to_csv(index=False))
    return buffer.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Seed Train Monte Carlo (Cytodex)", layout="wide")
st.title("Seed Train Monte Carlo (Cytodex surface model)")
st.caption("Fix applied: Day-0 comes from the model (no forced inoc point). Infectable = OCCUPIED + NEWLY_OCCUPIED + INHIBITED.")

with st.sidebar:
    st.header("Run controls")
    n_experiments = st.slider("Monte Carlo experiments (n)", 10, 5000, 100, step=10)

    st.subheader("Days per bioreactor")
    days_bio40 = st.slider("BIO40 days", 1, 200, 6)
    days_bio200a = st.slider("BIO200A days", 1, 200, 6)
    days_bio750 = st.slider("BIO750 (E/F/G/H) days", 1, 200, 6)

    st.subheader("Surface capacity")
    max_cells_setpoint = st.slider("Max cells/MC setpoint", 50, 60000, 140)
    max_cells_sd = st.slider("Max cells/MC SD", 0.0, 8000.0, 23.0)

    st.subheader("Microcarriers / particles")
    particles_per_g = st.number_input("Particles per g Cytodex", value=7087.0, step=1.0)

    st.subheader("MC concentration (g/L)")
    mc_bio40 = st.number_input("BIO40 MC (g/L)", value=1.2, step=0.1)
    mc_bio200a = st.number_input("BIO200A MC (g/L)", value=4.0, step=0.1)
    mc_bio750 = st.number_input("BIO750 MC (g/L)", value=2.6, step=0.1)

    st.subheader("Inoculation")
    inoc_bio40_cells_ml = st.number_input("BIO40 inoculation (cells/mL)", value=48000.0, step=1000.0)

    st.subheader("Transfer / trypsin yields")
    trypsin_yield_bio200a = st.slider("Trypsin yield BIO200A", 0.6, 1.0, 0.90, step=0.01)
    trypsin_yield_bio750 = st.slider("Trypsin yield BIO750", 0.6, 1.0, 0.87, step=0.01)
    transfer_bio1500a = st.slider("Transfer factor BIO1500A", 0.6, 1.0, 0.85, step=0.01)
    transfer_bio1500b = st.slider("Transfer factor BIO1500B", 0.6, 1.0, 0.85, step=0.01)

    st.subheader("Distribution factors")
    distribution_factor_EF = st.number_input("Distribution factor (E/F)", value=1.05 / 4, step=0.01)
    distribution_factor_GH = st.number_input("Distribution factor (G/H)", value=0.95 / 4, step=0.01)

    st.subheader("Inoculation SD (cells/MC)")
    inoc_sd_bio40 = st.number_input("BIO40 inoc SD (cells/MC)", value=2.94, step=0.1)
    inoc_sd_bio200a = st.number_input("BIO200A inoc SD (cells/MC)", value=1.0, step=0.1)
    inoc_sd_bio750 = st.number_input("BIO750 inoc SD (cells/MC)", value=2.2, step=0.1)

    rng_seed = st.number_input("RNG seed", value=1, step=1)

    y_choice = st.radio(
        "Plot metric",
        options=["TOTAL cells/mL", "INFECTABLE cells/mL", "MULTILAYER cells/mL"],
        index=1,
    )

    run_btn = st.button("Run simulation")

if run_btn:
    params = dict(
        n_experiments=int(n_experiments),
        days_bio40=int(days_bio40),
        days_bio200a=int(days_bio200a),
        days_bio750=int(days_bio750),
        max_cells_setpoint=float(max_cells_setpoint),
        max_cells_sd=float(max_cells_sd),
        particles_per_g=float(particles_per_g),
        mc_bio40_g_per_l=float(mc_bio40),
        mc_bio200a_g_per_l=float(mc_bio200a),
        mc_bio750_g_per_l=float(mc_bio750),
        inoc_bio40_cells_ml=float(inoc_bio40_cells_ml),
        inoc_sd_cells_per_mc_bio40=float(inoc_sd_bio40),
        inoc_sd_cells_per_mc_bio200a=float(inoc_sd_bio200a),
        inoc_sd_cells_per_mc_bio750=float(inoc_sd_bio750),
        trypsin_yield_bio200a=float(trypsin_yield_bio200a),
        trypsin_yield_bio750=float(trypsin_yield_bio750),
        transfer_bio1500a=float(transfer_bio1500a),
        transfer_bio1500b=float(transfer_bio1500b),
        distribution_factor_EF=float(distribution_factor_EF),
        distribution_factor_GH=float(distribution_factor_GH),
        rng_seed=int(rng_seed),
    )

    with st.spinner("Running Monte Carlo…"):
        series, summary = run_seed_train(params)

    if y_choice == "TOTAL cells/mL":
        y_col = "total_cells_ml"
        title = "All bioreactors — TOTAL cells/mL vs cumulative time"
    elif y_choice == "INFECTABLE cells/mL":
        y_col = "infectable_cells_ml"
        title = "All bioreactors — INFECTABLE cells/mL vs cumulative time"
    else:
        y_col = "multilayer_cells_ml"
        title = "All bioreactors — MULTILAYER cells/mL vs cumulative time"

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader(title)
        fig = plot_all_bioreactors_one_plot(series, y_col=y_col, title=title)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)

        zip_bytes = export_zip_csv(series, summary)
        st.download_button(
            "Download results (ZIP of CSVs)",
            data=zip_bytes,
            file_name=f"seed_train_sim_n{int(n_experiments)}.zip",
            mime="application/zip",
        )

    st.subheader("Raw time series (per bioreactor)")
    tabs = st.tabs(list(series.keys()))
    for tab, (name, df) in zip(tabs, series.items()):
        with tab:
            st.dataframe(df, use_container_width=True)

else:
    st.info("Set parameters in the sidebar and click **Run simulation**.")