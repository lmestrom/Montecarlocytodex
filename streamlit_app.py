import io
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import zipfile

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


def count_states(grid):
    """
    Counts +:
      - OCCUPIED_TOTAL: all non-empty (occupied+new+inhibited+multilayer)
      - INFECTABLE: monolayer-accessible (occupied+new)
    """
    uno = np.count_nonzero(grid == UNOCCUPIED)
    occ = np.count_nonzero(grid == OCCUPIED)
    new = np.count_nonzero(grid == NEWLY_OCCUPIED)
    inh = np.count_nonzero(grid == INHIBITED)
    mul = np.count_nonzero(grid == MULTILAYER)

    infectable = occ + new
    occupied_total = occ + new + inh + mul
    total = uno + occupied_total

    return {
        "UNOCCUPIED": uno,
        "OCCUPIED": occ,
        "NEWLY_OCCUPIED": new,
        "INHIBITED": inh,
        "MULTILAYER": mul,
        "INFECTABLE": infectable,
        "OCCUPIED_TOTAL": occupied_total,
        "TOTAL": total,
    }


def simulate_surface_mc(
    n_experiments: int,
    days: int,
    max_cells_setpoint: float,
    max_cells_sd: float,
    inoc_cells_per_mc_mean: float,
    inoc_cells_per_mc_sd: float,
    rng_seed: int = 1,
):
    """
    Monte Carlo simulation of cells/MC over time (surface model).
    Returns:
      mean_total, std_total, mean_infectable, std_infectable
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    per_day_total = []
    per_day_infect = []

    for _ in range(n_experiments):
        max_cells = max(1.0, float(np.random.normal(max_cells_setpoint, max_cells_sd)))
        grid_size = max(2, int(round(math.sqrt(max_cells))))
        capacity_sites = grid_size * grid_size

        inoc_cells = max(0.0, float(np.random.normal(inoc_cells_per_mc_mean, inoc_cells_per_mc_sd)))
        inoc_density = min(1.0, inoc_cells / capacity_sites)

        grid = np.zeros((grid_size, grid_size), dtype=int)
        multilayer_cycle_grid = np.zeros((grid_size, grid_size), dtype=int)

        initial_occupied_cells = int(capacity_sites * inoc_density)
        if initial_occupied_cells > 0:
            initial_positions = random.sample(range(capacity_sites), initial_occupied_cells)
            for pos in initial_positions:
                x, y = divmod(pos, grid_size)
                grid[x, y] = OCCUPIED

        traj_total = []
        traj_infect = []

        for _day in range(days):
            grid, multilayer_cycle_grid = update_state(grid, multilayer_cycle_grid)
            counts = count_states(grid)

            traj_total.append(counts["OCCUPIED_TOTAL"])
            traj_infect.append(counts["INFECTABLE"])

            grid[grid == NEWLY_OCCUPIED] = OCCUPIED

        per_day_total.append(np.array(traj_total, dtype=float))
        per_day_infect.append(np.array(traj_infect, dtype=float))

    arr_total = np.vstack(per_day_total)
    arr_infect = np.vstack(per_day_infect)

    mean_total = arr_total.mean(axis=0)
    std_total = arr_total.std(axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(mean_total)

    mean_infect = arr_infect.mean(axis=0)
    std_infect = arr_infect.std(axis=0, ddof=1) if n_experiments > 1 else np.zeros_like(mean_infect)

    return mean_total, std_total, mean_infect, std_infect


# ----------------------------
# Unit conversions
# ----------------------------
def cells_per_mc_to_cells_ml(cells_per_mc: np.ndarray, mc_g_per_l: float, particles_per_g: float) -> np.ndarray:
    """
    cells/mL = (cells/MC) * (MC per mL) * (MC per g)
            = (cells/MC) * (mc_g_per_l/1000) * (particles_per_g)
    """
    return cells_per_mc * (mc_g_per_l / 1000.0) * particles_per_g


def surface_traj_to_cells_ml_timeseries(
    mean_total_cells_per_mc: np.ndarray,
    std_total_cells_per_mc: np.ndarray,
    mean_infect_cells_per_mc: np.ndarray,
    std_infect_cells_per_mc: np.ndarray,
    days: int,
    mc_g_per_l: float,
    particles_per_g: float,
    t0_cells_ml: float,
):
    """
    Returns:
      time_h,
      total_cells_ml, total_cells_ml_std,
      infectable_cells_ml, infectable_cells_ml_std
    """
    t_hours = np.concatenate(([0.0], np.arange(1, days + 1, dtype=float) * 24.0))

    total_ml = cells_per_mc_to_cells_ml(mean_total_cells_per_mc, mc_g_per_l, particles_per_g)
    total_ml_std = cells_per_mc_to_cells_ml(std_total_cells_per_mc, mc_g_per_l, particles_per_g)

    infect_ml = cells_per_mc_to_cells_ml(mean_infect_cells_per_mc, mc_g_per_l, particles_per_g)
    infect_ml_std = cells_per_mc_to_cells_ml(std_infect_cells_per_mc, mc_g_per_l, particles_per_g)

    # include inoculation at t=0
    total_ml = np.concatenate(([t0_cells_ml], total_ml))
    total_ml_std = np.concatenate(([0.0], total_ml_std))

    infect_ml = np.concatenate(([t0_cells_ml], infect_ml))
    infect_ml_std = np.concatenate(([0.0], infect_ml_std))

    return t_hours, total_ml, total_ml_std, infect_ml, infect_ml_std


# ----------------------------
# Seed train logic
# ----------------------------
def run_seed_train(params):
    """
    Runs BIO40 -> BIO200A -> BIO750E/F/G/H -> BIO1500A/B

    Returns:
      - series dict {stage: df(time_h, cells_ml, cells_ml_std, infectable_cells_ml, infectable_cells_ml_std)}
      - summary table
    """
    PARTICLES_PER_G = params["particles_per_g"]

    # volumes (mL)
    V_BIO40 = 40000.0
    V_BIO200A = 200000.0
    V_BIO750 = 750000.0
    V_BIO1500 = 1500000.0

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

    max_cells_setpoint = params["max_cells_setpoint"]
    max_cells_sd = params["max_cells_sd"]

    # --- BIO40 inoculation (cells/mL -> cells/MC mean) ---
    inoc_cells_ml_40 = params["inoc_bio40_cells_ml"]
    cells_per_g_40 = inoc_cells_ml_40 / MC_BIO40
    inoc_cells_per_mc_40 = cells_per_g_40 / PARTICLES_PER_G

    mean40_tot, std40_tot, mean40_inf, std40_inf = simulate_surface_mc(
        n_experiments=n,
        days=d40,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_40,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio40"],
        rng_seed=params["rng_seed"],
    )

    t40, bio40_cells_ml, bio40_cells_ml_std, bio40_inf_ml, bio40_inf_ml_std = surface_traj_to_cells_ml_timeseries(
        mean40_tot, std40_tot, mean40_inf, std40_inf,
        d40, MC_BIO40, PARTICLES_PER_G, t0_cells_ml=inoc_cells_ml_40
    )

    # --- Transfer to BIO200A ---
    tryps_yield_200 = params["trypsin_yield_bio200a"]
    inoc_cells_ml_200 = tryps_yield_200 * V_BIO40 * bio40_cells_ml[-1] / V_BIO200A

    cells_per_g_200 = inoc_cells_ml_200 / MC_BIO200A
    inoc_cells_per_mc_200 = cells_per_g_200 / PARTICLES_PER_G

    mean200_tot, std200_tot, mean200_inf, std200_inf = simulate_surface_mc(
        n_experiments=n,
        days=d200,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc_cells_per_mc_200,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio200a"],
        rng_seed=params["rng_seed"] + 1,
    )

    t200, bio200_cells_ml, bio200_cells_ml_std, bio200_inf_ml, bio200_inf_ml_std = surface_traj_to_cells_ml_timeseries(
        mean200_tot, std200_tot, mean200_inf, std200_inf,
        d200, MC_BIO200A, PARTICLES_PER_G, t0_cells_ml=inoc_cells_ml_200
    )

    # --- Split to 4x BIO750 (E/F/G/H) ---
    bio200_end_total = bio200_cells_ml[-1]
    tryps_yield_750 = params["trypsin_yield_bio750"]

    dist_factor_EF = params["distribution_factor_EF"]
    dist_factor_GH = params["distribution_factor_GH"]

    def run_750(dist_factor: float, seed_offset: int, stage_name: str):
        inoc_cells_ml_750 = dist_factor * tryps_yield_750 * V_BIO200A * bio200_end_total / V_BIO750

        cells_per_g_750 = inoc_cells_ml_750 / MC_BIO750
        inoc_cells_per_mc_750 = cells_per_g_750 / PARTICLES_PER_G

        mean750_tot, std750_tot, mean750_inf, std750_inf = simulate_surface_mc(
            n_experiments=n,
            days=d750,
            max_cells_setpoint=max_cells_setpoint,
            max_cells_sd=max_cells_sd,
            inoc_cells_per_mc_mean=inoc_cells_per_mc_750,
            inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio750"],
            rng_seed=params["rng_seed"] + seed_offset,
        )

        t750, total_ml, total_ml_std, inf_ml, inf_ml_std = surface_traj_to_cells_ml_timeseries(
            mean750_tot, std750_tot, mean750_inf, std750_inf,
            d750, MC_BIO750, PARTICLES_PER_G, t0_cells_ml=inoc_cells_ml_750
        )

        df = pd.DataFrame(
            {
                "time_h": t750,
                "cells_ml": total_ml,
                "cells_ml_std": total_ml_std,
                "infectable_cells_ml": inf_ml,
                "infectable_cells_ml_std": inf_ml_std,
            }
        )

        return inoc_cells_ml_750, df

    inocE, dfE = run_750(dist_factor_EF, 2, "BIO750E")
    inocF, dfF = run_750(dist_factor_EF, 3, "BIO750F")
    inocG, dfG = run_750(dist_factor_GH, 4, "BIO750G")
    inocH, dfH = run_750(dist_factor_GH, 5, "BIO750H")

    # --- Combine to BIO1500 A/B (inoc only; display flat line) ---
    transfer_1500A = params["transfer_bio1500a"]
    transfer_1500B = params["transfer_bio1500b"]

    bio750E_end = dfE["cells_ml"].iloc[-1]
    bio750F_end = dfF["cells_ml"].iloc[-1]
    bio750G_end = dfG["cells_ml"].iloc[-1]
    bio750H_end = dfH["cells_ml"].iloc[-1]

    inoc_1500A = transfer_1500A * ((bio750E_end * V_BIO750) + (bio750F_end * V_BIO750)) / V_BIO1500
    inoc_1500B = transfer_1500B * ((bio750G_end * V_BIO750) + (bio750H_end * V_BIO750)) / V_BIO1500

    t1500 = np.array([0.0, float(d1500) * 24.0], dtype=float) if d1500 > 0 else np.array([0.0], dtype=float)

    def flat_df(inoc_val: float):
        return pd.DataFrame(
            {
                "time_h": t1500,
                "cells_ml": np.full_like(t1500, inoc_val, dtype=float),
                "cells_ml_std": np.zeros_like(t1500, dtype=float),
                "infectable_cells_ml": np.full_like(t1500, inoc_val, dtype=float),
                "infectable_cells_ml_std": np.zeros_like(t1500, dtype=float),
            }
        )

    df1500A = flat_df(inoc_1500A)
    df1500B = flat_df(inoc_1500B)

    series = {
        "BIO40": pd.DataFrame(
            {
                "time_h": t40,
                "cells_ml": bio40_cells_ml,
                "cells_ml_std": bio40_cells_ml_std,
                "infectable_cells_ml": bio40_inf_ml,
                "infectable_cells_ml_std": bio40_inf_ml_std,
            }
        ),
        "BIO200A": pd.DataFrame(
            {
                "time_h": t200,
                "cells_ml": bio200_cells_ml,
                "cells_ml_std": bio200_cells_ml_std,
                "infectable_cells_ml": bio200_inf_ml,
                "infectable_cells_ml_std": bio200_inf_ml_std,
            }
        ),
        "BIO750E": dfE,
        "BIO750F": dfF,
        "BIO750G": dfG,
        "BIO750H": dfH,
        "BIO1500A": df1500A,
        "BIO1500B": df1500B,
    }

    summary = pd.DataFrame(
        [
            {
                "Stage": "BIO40",
                "Inoc_cells/mL": inoc_cells_ml_40,
                "End_cells/mL": float(series["BIO40"]["cells_ml"].iloc[-1]),
                "End_infectable_cells/mL": float(series["BIO40"]["infectable_cells_ml"].iloc[-1]),
                "Days": d40,
                "MC_g/L": MC_BIO40,
            },
            {
                "Stage": "BIO200A",
                "Inoc_cells/mL": inoc_cells_ml_200,
                "End_cells/mL": float(series["BIO200A"]["cells_ml"].iloc[-1]),
                "End_infectable_cells/mL": float(series["BIO200A"]["infectable_cells_ml"].iloc[-1]),
                "Days": d200,
                "MC_g/L": MC_BIO200A,
            },
            {
                "Stage": "BIO750E",
                "Inoc_cells/mL": inocE,
                "End_cells/mL": float(dfE["cells_ml"].iloc[-1]),
                "End_infectable_cells/mL": float(dfE["infectable_cells_ml"].iloc[-1]),
                "Days": d750,
                "MC_g/L": MC_BIO750,
            },
            {
                "Stage": "BIO750F",
                "Inoc_cells/mL": inocF,
                "End_cells/mL": float(dfF["cells_ml"].iloc[-1]),
                "End_infectable_cells/mL": float(dfF["infectable_cells_ml"].iloc[-1]),
                "Days": d750,
                "MC_g/L": MC_BIO750,
            },
            {
                "Stage": "BIO750G",
                "Inoc_cells/mL": inocG,
                "End_cells/mL": float(dfG["cells_ml"].iloc[-1]),
                "End_infectable_cells/mL": float(dfG["infectable_cells_ml"].iloc[-1]),
                "Days": d750,
                "MC_g/L": MC_BIO750,
            },
            {
                "Stage": "BIO750H",
                "Inoc_cells/mL": inocH,
                "End_cells/mL": float(dfH["cells_ml"].iloc[-1]),
                "End_infectable_cells/mL": float(dfH["infectable_cells_ml"].iloc[-1]),
                "Days": d750,
                "MC_g/L": MC_BIO750,
            },
            {
                "Stage": "BIO1500A",
                "Inoc_cells/mL": inoc_1500A,
                "End_cells/mL": inoc_1500A,
                "End_infectable_cells/mL": inoc_1500A,
                "Days": d1500,
                "MC_g/L": np.nan,
            },
            {
                "Stage": "BIO1500B",
                "Inoc_cells/mL": inoc_1500B,
                "End_cells/mL": inoc_1500B,
                "End_infectable_cells/mL": inoc_1500B,
                "Days": d1500,
                "MC_g/L": np.nan,
            },
        ]
    )

    return series, summary


# ----------------------------
# Plotting
# ----------------------------
def plot_all_bioreactors_one_plot(series: dict, ycol: str):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    for name, df in series.items():
        ax.plot(df["time_h"], df[ycol], marker="o", label=name)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cells / mL")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    return fig


# ----------------------------
# Export ZIP of CSVs
# ----------------------------
def export_zip_csv(series: dict, summary: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Summary.csv", summary.to_csv(index=False))
        for name, df in series.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    return buf.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Seed Train Monte Carlo (Cytodex)", layout="wide")
st.title("Seed Train Monte Carlo (Cytodex surface model) — Infectable cells included")

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

    st.subheader("Plot selection")
    plot_infectable = st.checkbox("Plot infectable cells (monolayer accessible)", value=False)

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
        ycol = "infectable_cells_ml" if plot_infectable else "cells_ml"
        fig = plot_all_bioreactors_one_plot(series, ycol=ycol)
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
    st.info("Set parameters in the sidebar and click Run simulation.")