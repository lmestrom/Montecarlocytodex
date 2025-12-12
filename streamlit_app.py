# streamlit_app.py
# Seed Train Monte Carlo (Cytodex surface model) — Oxygen removed
# - Run button (no auto-run)
# - Sliders for days per bioreactor
# - Plots all bioreactors in ONE plot on a common time axis (cumulative hours)
# - Excel export WITHOUT openpyxl dependency (uses CSV-in-ZIP, works on Streamlit Cloud)

import io
import math
import random
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# Core grid model (surface only)
# ----------------------------
UNOCCUPIED = 0
OCCUPIED = 1
NEWLY_OCCUPIED = 2
INHIBITED = 3
MULTILAYER = 4


def update_state(grid: np.ndarray, multilayer_cycle_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One growth step on a periodic grid."""
    grid_size = grid.shape[0]
    new_grid = np.copy(grid)

    for i in range(grid_size):
        for j in range(grid_size):

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

            # multilayer rule
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


def count_states(grid: np.ndarray) -> dict:
    """Counts + an 'occupied_total' that includes all non-empty states."""
    uno = np.count_nonzero(grid == UNOCCUPIED)
    occ = np.count_nonzero(grid == OCCUPIED)
    new = np.count_nonzero(grid == NEWLY_OCCUPIED)
    inh = np.count_nonzero(grid == INHIBITED)
    mul = np.count_nonzero(grid == MULTILAYER)

    occupied_total = occ + new + inh + mul
    total = uno + occ + new + inh + mul
    return {
        "UNOCCUPIED": int(uno),
        "OCCUPIED": int(occ),
        "NEWLY_OCCUPIED": int(new),
        "INHIBITED": int(inh),
        "MULTILAYER": int(mul),
        "OCCUPIED_TOTAL": int(occupied_total),
        "TOTAL": int(total),
    }


def simulate_surface_mc(
    n_experiments: int,
    days: int,
    max_cells_setpoint: float,
    max_cells_sd: float,
    inoc_cells_per_mc_mean: float,
    inoc_cells_per_mc_sd: float,
    rng_seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation of cells/MC over time (surface model).
    Returns mean & std of OCCUPIED_TOTAL per day (days steps).
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    per_day_counts = []

    for _ in range(int(n_experiments)):
        max_cells = float(max(1.0, np.random.normal(max_cells_setpoint, max_cells_sd)))
        grid_size = max(2, int(round(math.sqrt(max_cells))))  # keep at least 2x2

        inoc_cells = float(max(0.0, np.random.normal(inoc_cells_per_mc_mean, inoc_cells_per_mc_sd)))

        # NOTE: grid capacity is grid_size^2 (not max_cells), because we discretize to a square grid.
        capacity_sites = float(grid_size * grid_size)
        inoc_density = min(1.0, inoc_cells / capacity_sites)

        grid = np.zeros((grid_size, grid_size), dtype=int)
        multilayer_cycle_grid = np.zeros((grid_size, grid_size), dtype=int)

        initial_occupied_cells = int(capacity_sites * inoc_density)
        initial_positions = random.sample(range(int(capacity_sites)), initial_occupied_cells)
        for pos in initial_positions:
            x, y = divmod(pos, grid_size)
            grid[x, y] = OCCUPIED

        traj = []
        for _day in range(int(days)):
            grid, multilayer_cycle_grid = update_state(grid, multilayer_cycle_grid)
            counts = count_states(grid)
            traj.append(float(counts["OCCUPIED_TOTAL"]))
            grid[grid == NEWLY_OCCUPIED] = OCCUPIED

        per_day_counts.append(np.array(traj, dtype=float))

    arr = np.vstack(per_day_counts) if per_day_counts else np.zeros((0, days), dtype=float)
    if arr.shape[0] == 0:
        return np.zeros(days, dtype=float), np.zeros(days, dtype=float)

    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


def cells_ml_from_cells_per_mc(
    mean_cells_per_mc: np.ndarray,
    std_cells_per_mc: np.ndarray,
    mc_g_per_l: float,
    particles_per_g: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts cells/MC to cells/mL using:
      cells/mL = (mc_g/L) * (particles/g) * (cells/particle) / 1000 (mL/L)
    """
    factor = (mc_g_per_l * particles_per_g) / 1000.0
    return factor * mean_cells_per_mc, factor * std_cells_per_mc


def make_stage_timeseries(
    stage_name: str,
    start_time_h: float,
    days: int,
    mean_cells_per_mc: np.ndarray,
    std_cells_per_mc: np.ndarray,
    mc_g_per_l: float,
    particles_per_g: float,
    inoc_cells_ml: float,
) -> pd.DataFrame:
    """Builds a dataframe with cumulative time (hours), including t0 row for the stage."""
    # day steps at 24h increments
    t = np.arange(1, days + 1, dtype=float) * 24.0
    cells_ml, cells_ml_std = cells_ml_from_cells_per_mc(mean_cells_per_mc, std_cells_per_mc, mc_g_per_l, particles_per_g)

    # include t0 row
    t_all = np.concatenate(([0.0], t))
    cells_all = np.concatenate(([float(inoc_cells_ml)], cells_ml))
    std_all = np.concatenate(([float(0.0)], cells_ml_std))

    df = pd.DataFrame(
        {
            "stage": stage_name,
            "time_h_stage": t_all,
            "time_h_total": start_time_h + t_all,
            "cells_ml": cells_all,
            "cells_ml_std": std_all,
        }
    )
    return df


def flat_stage_df(stage_name: str, start_time_h: float, days: int, inoc_cells_ml: float) -> pd.DataFrame:
    """Stage displayed as a flat line (no growth model applied)."""
    t_all = np.array([0.0, float(days) * 24.0], dtype=float)
    df = pd.DataFrame(
        {
            "stage": stage_name,
            "time_h_stage": t_all,
            "time_h_total": start_time_h + t_all,
            "cells_ml": np.array([float(inoc_cells_ml), float(inoc_cells_ml)]),
            "cells_ml_std": np.array([0.0, 0.0]),
        }
    )
    return df


def plot_all_bioreactors_one_plot(series: dict[str, pd.DataFrame]):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    for name, df in series.items():
        ax.plot(df["time_h_total"], df["cells_ml"], marker="o", label=name)

    ax.set_xlabel("Cumulative time (h)")
    ax.set_ylabel("Cells / mL")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    return fig


def export_zip_csv(series: dict[str, pd.DataFrame], summary: pd.DataFrame) -> bytes:
    """Creates a ZIP with Summary.csv + one CSV per stage (no openpyxl needed)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Summary.csv", summary.to_csv(index=False))
        for k, df in series.items():
            safe = k.replace("/", "_").replace("\\", "_")
            zf.writestr(f"{safe}.csv", df.to_csv(index=False))
    return buffer.getvalue()


# ----------------------------
# Seed train logic
# ----------------------------
@dataclass
class Params:
    # MC & particles
    particles_per_g: float = 7087.0
    mc_bio40_g_per_l: float = 1.2
    mc_bio200a_g_per_l: float = 4.0
    mc_bio750_g_per_l: float = 2.6

    # volumes (mL)
    v_bio40_ml: float = 40000.0
    v_bio200a_ml: float = 200000.0
    v_bio750_ml: float = 750000.0
    v_bio1500_ml: float = 1500000.0

    # days
    days_bio40: int = 6
    days_bio200a: int = 6
    days_bio750: int = 6
    days_bio1500: int = 0

    # Monte Carlo
    n_experiments: int = 100
    rng_seed: int = 1

    # surface capacity
    max_cells_setpoint: float = 140.0
    max_cells_sd: float = 23.0

    # inoculations
    inoc_bio40_cells_ml: float = 48000.0

    inoc_sd_cells_per_mc_bio40: float = 2.94
    inoc_sd_cells_per_mc_bio200a: float = 1.0
    inoc_sd_cells_per_mc_bio750: float = 2.2

    # yields / transfers
    trypsin_yield_bio200a: float = 0.90
    trypsin_yield_bio750: float = 0.87
    transfer_bio1500a: float = 0.85
    transfer_bio1500b: float = 0.85

    # split factors
    distribution_factor_EF: float = 1.05 / 4
    distribution_factor_GH: float = 0.95 / 4


def run_seed_train(p: Params) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    PART = float(p.particles_per_g)

    # --- BIO40 inoc (cells/mL -> cells/MC) ---
    inoc_cells_ml_40 = float(p.inoc_bio40_cells_ml)
    cells_per_g_40 = inoc_cells_ml_40 / float(p.mc_bio40_g_per_l)
    inoc_cells_per_mc_40 = cells_per_g_40 / PART

    mean40, std40 = simulate_surface_mc(
        n_experiments=p.n_experiments,
        days=p.days_bio40,
        max_cells_setpoint=p.max_cells_setpoint,
        max_cells_sd=p.max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_40,
        inoc_cells_per_mc_sd=p.inoc_sd_cells_per_mc_bio40,
        rng_seed=p.rng_seed,
    )

    t0 = 0.0
    df40 = make_stage_timeseries(
        "BIO40",
        start_time_h=t0,
        days=p.days_bio40,
        mean_cells_per_mc=mean40,
        std_cells_per_mc=std40,
        mc_g_per_l=p.mc_bio40_g_per_l,
        particles_per_g=PART,
        inoc_cells_ml=inoc_cells_ml_40,
    )
    end40 = float(df40["cells_ml"].iloc[-1])
    t_end40 = float(df40["time_h_total"].iloc[-1])

    # --- Transfer to BIO200A (trypsin yield loss) ---
    inoc_cells_ml_200 = float(p.trypsin_yield_bio200a) * float(p.v_bio40_ml) * end40 / float(p.v_bio200a_ml)
    cells_per_g_200 = inoc_cells_ml_200 / float(p.mc_bio200a_g_per_l)
    inoc_cells_per_mc_200 = cells_per_g_200 / PART

    mean200, std200 = simulate_surface_mc(
        n_experiments=p.n_experiments,
        days=p.days_bio200a,
        max_cells_setpoint=p.max_cells_setpoint,
        max_cells_sd=p.max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_200,
        inoc_cells_per_mc_sd=p.inoc_sd_cells_per_mc_bio200a,
        rng_seed=p.rng_seed + 1,
    )

    df200 = make_stage_timeseries(
        "BIO200A",
        start_time_h=t_end40,
        days=p.days_bio200a,
        mean_cells_per_mc=mean200,
        std_cells_per_mc=std200,
        mc_g_per_l=p.mc_bio200a_g_per_l,
        particles_per_g=PART,
        inoc_cells_ml=inoc_cells_ml_200,
    )
    end200 = float(df200["cells_ml"].iloc[-1])
    t_end200 = float(df200["time_h_total"].iloc[-1])

    # --- 4x BIO750 (E/F/G/H) ---
    def run_750(stage: str, dist_factor: float, seed_offset: int) -> tuple[float, pd.DataFrame]:
        inoc_cells_ml_750 = float(dist_factor) * float(p.trypsin_yield_bio750) * float(p.v_bio200a_ml) * end200 / float(p.v_bio750_ml)
        cells_per_g_750 = inoc_cells_ml_750 / float(p.mc_bio750_g_per_l)
        inoc_cells_per_mc_750 = cells_per_g_750 / PART

        mean750, std750 = simulate_surface_mc(
            n_experiments=p.n_experiments,
            days=p.days_bio750,
            max_cells_setpoint=p.max_cells_setpoint,
            max_cells_sd=p.max_cells_sd,
            inoc_cells_per_mc_mean=inoc_cells_per_mc_750,
            inoc_cells_per_mc_sd=p.inoc_sd_cells_per_mc_bio750,
            rng_seed=p.rng_seed + seed_offset,
        )

        df750 = make_stage_timeseries(
            stage,
            start_time_h=t_end200,
            days=p.days_bio750,
            mean_cells_per_mc=mean750,
            std_cells_per_mc=std750,
            mc_g_per_l=p.mc_bio750_g_per_l,
            particles_per_g=PART,
            inoc_cells_ml=inoc_cells_ml_750,
        )
        return inoc_cells_ml_750, df750

    inocE, df750E = run_750("BIO750E", p.distribution_factor_EF, 2)
    inocF, df750F = run_750("BIO750F", p.distribution_factor_EF, 3)
    inocG, df750G = run_750("BIO750G", p.distribution_factor_GH, 4)
    inocH, df750H = run_750("BIO750H", p.distribution_factor_GH, 5)

    endE = float(df750E["cells_ml"].iloc[-1])
    endF = float(df750F["cells_ml"].iloc[-1])
    endG = float(df750G["cells_ml"].iloc[-1])
    endH = float(df750H["cells_ml"].iloc[-1])

    t_end750 = float(df750E["time_h_total"].iloc[-1])  # same timing for all 750s

    # --- Combine to BIO1500 A/B ---
    inoc_1500A = float(p.transfer_bio1500a) * (endE * float(p.v_bio750_ml) + endF * float(p.v_bio750_ml)) / float(p.v_bio1500_ml)
    inoc_1500B = float(p.transfer_bio1500b) * (endG * float(p.v_bio750_ml) + endH * float(p.v_bio750_ml)) / float(p.v_bio1500_ml)

    df1500A = flat_stage_df("BIO1500A", start_time_h=t_end750, days=p.days_bio1500, inoc_cells_ml=inoc_1500A)
    df1500B = flat_stage_df("BIO1500B", start_time_h=t_end750, days=p.days_bio1500, inoc_cells_ml=inoc_1500B)

    series = {
        "BIO40": df40,
        "BIO200A": df200,
        "BIO750E": df750E,
        "BIO750F": df750F,
        "BIO750G": df750G,
        "BIO750H": df750H,
        "BIO1500A": df1500A,
        "BIO1500B": df1500B,
    }

    summary = pd.DataFrame(
        [
            {"Stage": "BIO40", "Inoc_cells/mL": inoc_cells_ml_40, "End_cells/mL": end40, "Days": p.days_bio40, "MC_g/L": p.mc_bio40_g_per_l},
            {"Stage": "BIO200A", "Inoc_cells/mL": inoc_cells_ml_200, "End_cells/mL": end200, "Days": p.days_bio200a, "MC_g/L": p.mc_bio200a_g_per_l},
            {"Stage": "BIO750E", "Inoc_cells/mL": inocE, "End_cells/mL": endE, "Days": p.days_bio750, "MC_g/L": p.mc_bio750_g_per_l},
            {"Stage": "BIO750F", "Inoc_cells/mL": inocF, "End_cells/mL": endF, "Days": p.days_bio750, "MC_g/L": p.mc_bio750_g_per_l},
            {"Stage": "BIO750G", "Inoc_cells/mL": inocG, "End_cells/mL": endG, "Days": p.days_bio750, "MC_g/L": p.mc_bio750_g_per_l},
            {"Stage": "BIO750H", "Inoc_cells/mL": inocH, "End_cells/mL": endH, "Days": p.days_bio750, "MC_g/L": p.mc_bio750_g_per_l},
            {"Stage": "BIO1500A", "Inoc_cells/mL": inoc_1500A, "End_cells/mL": inoc_1500A, "Days": p.days_bio1500, "MC_g/L": np.nan},
            {"Stage": "BIO1500B", "Inoc_cells/mL": inoc_1500B, "End_cells/mL": inoc_1500B, "Days": p.days_bio1500, "MC_g/L": np.nan},
        ]
    )

    return series, summary


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Seed Train Monte Carlo (Cytodex)", layout="wide")
st.title("Seed Train Monte Carlo (Cytodex surface model) — Oxygen removed")

with st.sidebar:
    st.header("Run controls")
    n_experiments = st.slider("Monte Carlo experiments (n)", 10, 2000, 100, step=10)

    st.subheader("Days per bioreactor")
    days_bio40 = st.slider("BIO40 days", 1, 14, 6)
    days_bio200a = st.slider("BIO200A days", 1, 14, 6)
    days_bio750 = st.slider("BIO750 (E/F/G/H) days", 1, 14, 6)
    days_bio1500 = st.slider("BIO1500 (A/B) days (display only)", 0, 14, 0)

    st.subheader("Surface capacity")
    max_cells_setpoint = st.slider("Max cells/MC setpoint", 50, 300, 140)
    max_cells_sd = st.slider("Max cells/MC SD", 0.0, 80.0, 23.0)

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

    run_btn = st.button("Run simulation", type="primary")


if run_btn:
    params = Params(
        n_experiments=int(n_experiments),
        days_bio40=int(days_bio40),
        days_bio200a=int(days_bio200a),
        days_bio750=int(days_bio750),
        days_bio1500=int(days_bio1500),
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

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("All bioreactors — one plot (cells/mL vs cumulative time)")
        fig = plot_all_bioreactors_one_plot(series)
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
```