import io
import math
import random
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

def update_state(grid, multilayer_cycle_grid):
    """One growth step on a periodic grid."""
    grid_size = grid.shape[0]
    new_grid = np.copy(grid)

    for i in range(grid_size):
        for j in range(grid_size):

            if grid[i, j] == OCCUPIED:
                neighbors = [(x % grid_size, y % grid_size)
                             for x in range(i - 1, i + 2)
                             for y in range(j - 1, j + 2)
                             if (x, y) != (i, j)]
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
                neighbors = [(x % grid_size, y % grid_size)
                             for x in range(i - 1, i + 2)
                             for y in range(j - 1, j + 2)
                             if (x, y) != (i, j)]

                if any(new_grid[x, y] in [INHIBITED, MULTILAYER] for x, y in neighbors):
                    multilayer_cycle_grid[i, j] += 1
                    if multilayer_cycle_grid[i, j] >= 1:
                        new_grid[i, j] = MULTILAYER
                        multilayer_cycle_grid[i, j] = 0
                else:
                    multilayer_cycle_grid[i, j] = 0

    return new_grid, multilayer_cycle_grid


def count_states(grid):
    """Counts + an 'occupied_total' that includes all non-empty states."""
    uno = np.count_nonzero(grid == UNOCCUPIED)
    occ = np.count_nonzero(grid == OCCUPIED)
    new = np.count_nonzero(grid == NEWLY_OCCUPIED)
    inh = np.count_nonzero(grid == INHIBITED)
    mul = np.count_nonzero(grid == MULTILAYER)

    occupied_total = occ + new + inh + mul
    total = uno + occ + new + inh + mul
    return {
        "UNOCCUPIED": uno,
        "OCCUPIED": occ,
        "NEWLY_OCCUPIED": new,
        "INHIBITED": inh,
        "MULTILAYER": mul,
        "OCCUPIED_TOTAL": occupied_total,
        "TOTAL": total
    }


def simulate_surface_mc(
    n_experiments: int,
    days: int,
    max_cells_setpoint: float,
    max_cells_sd: float,
    inoc_cells_per_mc_mean: float,
    inoc_cells_per_mc_sd: float,
    rng_seed: int = 1
):
    """
    Monte Carlo simulation of cells/MC over time (surface model).
    Returns mean & std of OCCUPIED_TOTAL per day (days steps).
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    per_day_counts = []  # list of arrays shape (days,) for each experiment

    for _ in range(n_experiments):
        max_cells = max(1.0, np.random.normal(max_cells_setpoint, max_cells_sd))
        grid_size = max(2, int(round(math.sqrt(max_cells))))  # keep at least 2x2

        inoc_cells = max(0.0, np.random.normal(inoc_cells_per_mc_mean, inoc_cells_per_mc_sd))
        inoc_density = min(1.0, inoc_cells / (grid_size * grid_size))  # safe cap

        grid = np.zeros((grid_size, grid_size), dtype=int)
        multilayer_cycle_grid = np.zeros((grid_size, grid_size), dtype=int)

        initial_occupied_cells = int(grid_size * grid_size * inoc_density)
        initial_positions = random.sample(range(grid_size * grid_size), initial_occupied_cells)
        for pos in initial_positions:
            x, y = divmod(pos, grid_size)
            grid[x, y] = OCCUPIED

        traj = []
        for _day in range(days):
            grid, multilayer_cycle_grid = update_state(grid, multilayer_cycle_grid)
            counts = count_states(grid)
            traj.append(counts["OCCUPIED_TOTAL"])
            grid[grid == NEWLY_OCCUPIED] = OCCUPIED

        per_day_counts.append(np.array(traj, dtype=float))

    arr = np.vstack(per_day_counts)  # (n, days)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(mean)
    return mean, std


def growth_curve_cells_ml_from_surface(
    mean_cells_per_mc: np.ndarray,
    std_cells_per_mc: np.ndarray,
    days: int,
    mc_g_per_l: float,
    particles_per_g: float,
    include_t0: bool = True
):
    """
    Converts cells/MC to cells/mL using:
      cells/mL = mc_g/L * (particles/g) * (cells/particle) / 1000 (mL/L)
    """
    # cells/mL per day step
    cells_per_ml = (mc_g_per_l * particles_per_g * mean_cells_per_mc) / 1000.0
    cells_per_ml_std = (mc_g_per_l * particles_per_g * std_cells_per_mc) / 1000.0

    # time axis (hours)
    t_hours = np.arange(1, days + 1) * 24  # 24, 48, ...
    if include_t0:
        t_hours = np.concatenate(([0], t_hours))
        cells_per_ml = np.concatenate(([cells_per_ml[0] * 0 + cells_per_ml[0]], cells_per_ml))
        cells_per_ml_std = np.concatenate(([cells_per_ml_std[0] * 0 + cells_per_ml_std[0]], cells_per_ml_std))

    return t_hours, cells_per_ml, cells_per_ml_std


# ----------------------------
# Seed train logic
# ----------------------------
def run_seed_train(params):
    """
    Runs BIO40 -> BIO200A -> BIO750E/F/G/H -> BIO1500A/B
    Returns:
      - dict of time series per bioreactor: {name: df(time_h, cells_ml, cells_ml_std)}
      - summary table of inoculation & end-point
    """
    PARTICLES_PER_G = params["particles_per_g"]  # 7087 by your assumption

    # volumes (mL)
    V_BIO40 = 40000
    V_BIO200A = 200000
    V_BIO750 = 750000
    V_BIO1500 = 1500000

    # MC (g/L)
    MC_BIO40 = params["mc_bio40_g_per_l"]
    MC_BIO200A = params["mc_bio200a_g_per_l"]
    MC_BIO750 = params["mc_bio750_g_per_l"]

    # days
    d40 = params["days_bio40"]
    d200 = params["days_bio200a"]
    d750 = params["days_bio750"]
    d1500 = params["days_bio1500"]

    n = params["n_experiments"]

    # Surface model parameters
    max_cells_setpoint = params["max_cells_setpoint"]
    max_cells_sd = params["max_cells_sd"]

    # --- BIO40 inoc (cells/mL -> cells/MC) ---
    inoc_cells_ml_40 = params["inoc_bio40_cells_ml"]
    cells_per_g_40 = inoc_cells_ml_40 / MC_BIO40
    inoc_cells_per_mc_40 = cells_per_g_40 / PARTICLES_PER_G

    mean40, std40 = simulate_surface_mc(
        n_experiments=n,
        days=d40,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_40,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio40"],
        rng_seed=params["rng_seed"]
    )

    t40, bio40_cells_ml, bio40_cells_ml_std = growth_curve_cells_ml_from_surface(
        mean40, std40, d40, MC_BIO40, PARTICLES_PER_G
    )

    # --- Transfer to BIO200A (trypsin yield loss) ---
    tryps_yield_200 = params["trypsin_yield_bio200a"]
    inoc_cells_ml_200 = tryps_yield_200 * V_BIO40 * bio40_cells_ml[-1] / V_BIO200A
    cells_per_g_200 = inoc_cells_ml_200 / MC_BIO200A
    inoc_cells_per_mc_200 = cells_per_g_200 / PARTICLES_PER_G

    mean200, std200 = simulate_surface_mc(
        n_experiments=n,
        days=d200,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_200,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio200a"],
        rng_seed=params["rng_seed"] + 1
    )
    t200, bio200_cells_ml, bio200_cells_ml_std = growth_curve_cells_ml_from_surface(
        mean200, std200, d200, MC_BIO200A, PARTICLES_PER_G
    )

    # --- Split to 4x BIO750 (E/F/G/H) ---
    bio200_end = bio200_cells_ml[-1]
    tryps_yield_750 = params["trypsin_yield_bio750"]
    dist_factor_EF = params["distribution_factor_EF"]
    dist_factor_GH = params["distribution_factor_GH"]

    def run_750(dist_factor, seed_offset):
        inoc_cells_ml_750 = dist_factor * tryps_yield_750 * V_BIO200A * bio200_end / V_BIO750
        cells_per_g_750 = inoc_cells_ml_750 / MC_BIO750
        inoc_cells_per_mc_750 = cells_per_g_750 / PARTICLES_PER_G

        mean750, std750 = simulate_surface_mc(
            n_experiments=n,
            days=d750,
            max_cells_setpoint=max_cells_setpoint,
            max_cells_sd=max_cells_sd,
            inoc_cells_per_mc_mean=inoc_cells_per_mc_750,
            inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio750"],
            rng_seed=params["rng_seed"] + seed_offset
        )
        t750, cells_ml, cells_ml_std = growth_curve_cells_ml_from_surface(
            mean750, std750, d750, MC_BIO750, PARTICLES_PER_G
        )
        return inoc_cells_ml_750, t750, cells_ml, cells_ml_std

    inocE, t750, bio750E_cells_ml, bio750E_cells_ml_std = run_750(dist_factor_EF, 2)
    inocF, _,    bio750F_cells_ml, bio750F_cells_ml_std = run_750(dist_factor_EF, 3)
    inocG, _,    bio750G_cells_ml, bio750G_cells_ml_std = run_750(dist_factor_GH, 4)
    inocH, _,    bio750H_cells_ml, bio750H_cells_ml_std = run_750(dist_factor_GH, 5)

    # --- Combine to BIO1500 A/B (two trains) ---
    transfer_1500A = params["transfer_bio1500a"]
    transfer_1500B = params["transfer_bio1500b"]

    inoc_1500A = transfer_1500A * (bio750E_cells_ml[-1] * V_BIO750 + bio750F_cells_ml[-1] * V_BIO750) / V_BIO1500
    inoc_1500B = transfer_1500B * (bio750G_cells_ml[-1] * V_BIO750 + bio750H_cells_ml[-1] * V_BIO750) / V_BIO1500

    # For BIO1500, you can either:
    # (A) stop here (seed concentration result), or
    # (B) run another surface growth stage if you also use microcarriers there.
    # For now: we just show inoc as a flat "end-point" (no growth), unless you add MC_1500 & days model.
    t1500 = np.array([0, d1500 * 24], dtype=float)
    bio1500A_cells_ml = np.array([inoc_1500A, inoc_1500A], dtype=float)
    bio1500B_cells_ml = np.array([inoc_1500B, inoc_1500B], dtype=float)
    bio1500A_cells_ml_std = np.array([0.0, 0.0])
    bio1500B_cells_ml_std = np.array([0.0, 0.0])

    series = {
        "BIO40":   pd.DataFrame({"time_h": t40,  "cells_ml": bio40_cells_ml,  "cells_ml_std": bio40_cells_ml_std}),
        "BIO200A": pd.DataFrame({"time_h": t200, "cells_ml": bio200_cells_ml, "cells_ml_std": bio200_cells_ml_std}),
        "BIO750E": pd.DataFrame({"time_h": t750, "cells_ml": bio750E_cells_ml, "cells_ml_std": bio750E_cells_ml_std}),
        "BIO750F": pd.DataFrame({"time_h": t750, "cells_ml": bio750F_cells_ml, "cells_ml_std": bio750F_cells_ml_std}),
        "BIO750G": pd.DataFrame({"time_h": t750, "cells_ml": bio750G_cells_ml, "cells_ml_std": bio750G_cells_ml_std}),
        "BIO750H": pd.DataFrame({"time_h": t750, "cells_ml": bio750H_cells_ml, "cells_ml_std": bio750H_cells_ml_std}),
        "BIO1500A": pd.DataFrame({"time_h": t1500, "cells_ml": bio1500A_cells_ml, "cells_ml_std": bio1500A_cells_ml_std}),
        "BIO1500B": pd.DataFrame({"time_h": t1500, "cells_ml": bio1500B_cells_ml, "cells_ml_std": bio1500B_cells_ml_std}),
    }

    summary = pd.DataFrame([
        {"Stage": "BIO40",   "Inoc_cells/mL": inoc_cells_ml_40,  "End_cells/mL": bio40_cells_ml[-1],  "Days": d40,  "MC_g/L": MC_BIO40},
        {"Stage": "BIO200A", "Inoc_cells/mL": inoc_cells_ml_200, "End_cells/mL": bio200_cells_ml[-1], "Days": d200, "MC_g/L": MC_BIO200A},
        {"Stage": "BIO750E", "Inoc_cells/mL": inocE,            "End_cells/mL": bio750E_cells_ml[-1], "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO750F", "Inoc_cells/mL": inocF,            "End_cells/mL": bio750F_cells_ml[-1], "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO750G", "Inoc_cells/mL": inocG,            "End_cells/mL": bio750G_cells_ml[-1], "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO750H", "Inoc_cells/mL": inocH,            "End_cells/mL": bio750H_cells_ml[-1], "Days": d750, "MC_g/L": MC_BIO750},
        {"Stage": "BIO1500A","Inoc_cells/mL": inoc_1500A,       "End_cells/mL": inoc_1500A,          "Days": d1500,"MC_g/L": np.nan},
        {"Stage": "BIO1500B","Inoc_cells/mL": inoc_1500B,       "End_cells/mL": inoc_1500B,          "Days": d1500,"MC_g/L": np.nan},
    ])

    return series, summary


def plot_all_bioreactors_one_plot(series: dict):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    for name, df in series.items():
        ax.plot(df["time_h"], df["cells_ml"], marker="o", label=name)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cells / mL")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Seed Train Monte Carlo (Cytodex)", layout="wide")
st.title("Seed Train Monte Carlo (Cytodex surface model) — Oxygen removed")

with st.sidebar:
    st.header("Run controls")
    n_experiments = st.slider("Monte Carlo experiments (n)", 10, 2000, 100, step=10)

    st.subheader("Days per bioreactor (sliders)")
    days_bio40 = st.slider("BIO40 days", 1, 14, 6)
    days_bio200a = st.slider("BIO200A days", 1, 14, 6)
    days_bio750 = st.slider("BIO750 (E/F/G/H) days", 1, 14, 6)
    days_bio1500 = st.slider("BIO1500 (A/B) days (display only for now)", 0, 14, 0)

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
    distribution_factor_EF = st.number_input("Distribution factor (E/F)", value=1.05/4, step=0.01)
    distribution_factor_GH = st.number_input("Distribution factor (G/H)", value=0.95/4, step=0.01)

    st.subheader("Inoculation SD (cells/MC)")
    inoc_sd_bio40 = st.number_input("BIO40 inoc SD (cells/MC)", value=2.94, step=0.1)
    inoc_sd_bio200a = st.number_input("BIO200A inoc SD (cells/MC)", value=1.0, step=0.1)
    inoc_sd_bio750 = st.number_input("BIO750 inoc SD (cells/MC)", value=2.2, step=0.1)

    rng_seed = st.number_input("RNG seed", value=1, step=1)

    run_btn = st.button("Run simulation")


if run_btn:
    params = dict(
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
        st.subheader("All bioreactors — one plot (cells/mL vs time)")
        fig = plot_all_bioreactors_one_plot(series)
        st.pyplot(fig)

    with col2:
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True)

        # Export bundle
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            for k, df in series.items():
                df.to_excel(writer, sheet_name=k[:31], index=False)

        st.download_button(
            "Download Excel",
            data=buffer.getvalue(),
            file_name=f"seed_train_sim_n{n_experiments}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Raw time series (per bioreactor)")
    tabs = st.tabs(list(series.keys()))
    for tab, (name, df) in zip(tabs, series.items()):
        with tab:
            st.dataframe(df, use_container_width=True)