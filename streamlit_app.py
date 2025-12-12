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
# Core grid model (surface only)
# ----------------------------
UNOCCUPIED = 0
OCCUPIED = 1
NEWLY_OCCUPIED = 2
INHIBITED = 3
MULTILAYER = 4


def update_state(grid: np.ndarray, multilayer_cycle_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One growth step on a periodic grid."""
    grid_size = grid.shape[0]
    new_grid = np.copy(grid)

    for i in range(grid_size):
        for j in range(grid_size):

            # spreading from OCCUPIED
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

            # multilayer rule triggered from INHIBITED
            if new_grid[i, j] == INHIBITED:
                neighbors = [(x % grid_size, y % grid_size)
                             for x in range(i - 1, i + 2)
                             for y in range(j - 1, j + 2)
                             if (x, y) != (i, j)]

                if any(new_grid[x, y] in (INHIBITED, MULTILAYER) for x, y in neighbors):
                    multilayer_cycle_grid[i, j] += 1
                    if multilayer_cycle_grid[i, j] >= 1:
                        new_grid[i, j] = MULTILAYER
                        multilayer_cycle_grid[i, j] = 0
                else:
                    multilayer_cycle_grid[i, j] = 0

    return new_grid, multilayer_cycle_grid


def count_states(grid: np.ndarray) -> Dict[str, int]:
    """Counts states + derived totals."""
    uno = int(np.count_nonzero(grid == UNOCCUPIED))
    occ = int(np.count_nonzero(grid == OCCUPIED))
    new = int(np.count_nonzero(grid == NEWLY_OCCUPIED))
    inh = int(np.count_nonzero(grid == INHIBITED))
    mul = int(np.count_nonzero(grid == MULTILAYER))

    total = occ + new + inh + mul
    infectable = occ + new + inh  # excludes MULTILAYER

    return {
        "UNOCCUPIED": uno,
        "OCCUPIED": occ,
        "NEWLY_OCCUPIED": new,
        "INHIBITED": inh,
        "MULTILAYER": mul,
        "TOTAL": total,
        "INFECTABLE": infectable,
    }


def simulate_surface_mc(
    n_experiments: int,
    days: int,
    max_cells_setpoint: float,
    max_cells_sd: float,
    inoc_cells_per_mc_mean: float,
    inoc_cells_per_mc_sd: float,
    rng_seed: int = 1
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Monte Carlo simulation on surface (cells/bead-like grid).
    Returns mean & std trajectories for:
      TOTAL, INFECTABLE, MULTILAYER
    Each trajectory is length (days+1) including day0 (initial state).
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    traj_total = []
    traj_infectable = []
    traj_multilayer = []

    for _ in range(n_experiments):
        max_cells = float(max(1.0, np.random.normal(max_cells_setpoint, max_cells_sd)))
        grid_size = max(2, int(round(math.sqrt(max_cells))))

        inoc_cells = float(max(0.0, np.random.normal(inoc_cells_per_mc_mean, inoc_cells_per_mc_sd)))

        # IMPORTANT: inoculation is in cells/MC; grid has grid_size^2 sites (capacity)
        # Cap inoc_density so initial occupied never exceeds capacity
        capacity = grid_size * grid_size
        inoc_density = min(1.0, inoc_cells / capacity)

        grid = np.zeros((grid_size, grid_size), dtype=int)
        multilayer_cycle_grid = np.zeros((grid_size, grid_size), dtype=int)

        initial_occupied = int(round(capacity * inoc_density))
        if initial_occupied > 0:
            positions = random.sample(range(capacity), min(initial_occupied, capacity))
            for pos in positions:
                x, y = divmod(pos, grid_size)
                grid[x, y] = OCCUPIED

        # day 0 counts (model-based)
        c0 = count_states(grid)
        series_total = [c0["TOTAL"]]
        series_inf = [c0["INFECTABLE"]]
        series_mul = [c0["MULTILAYER"]]

        for _day in range(days):
            grid, multilayer_cycle_grid = update_state(grid, multilayer_cycle_grid)
            counts = count_states(grid)

            series_total.append(counts["TOTAL"])
            series_inf.append(counts["INFECTABLE"])
            series_mul.append(counts["MULTILAYER"])

            # commit newly occupied -> occupied for next step
            grid[grid == NEWLY_OCCUPIED] = OCCUPIED

        traj_total.append(np.array(series_total, dtype=float))
        traj_infectable.append(np.array(series_inf, dtype=float))
        traj_multilayer.append(np.array(series_mul, dtype=float))

    A_total = np.vstack(traj_total)
    A_inf = np.vstack(traj_infectable)
    A_mul = np.vstack(traj_multilayer)

    def mean_std(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = A.mean(axis=0)
        std = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
        return mean, std

    return {
        "TOTAL": mean_std(A_total),
        "INFECTABLE": mean_std(A_inf),
        "MULTILAYER": mean_std(A_mul),
    }


def beads_per_ml_from_mc(mc_g_per_l: float, beads_per_g: float) -> float:
    """
    beads/mL = (g/L)*(beads/g)/1000
    """
    return (mc_g_per_l * beads_per_g) / 1000.0


def cells_ml_from_cells_per_mc(
    mean_cells_per_mc: np.ndarray,
    std_cells_per_mc: np.ndarray,
    mc_g_per_l: float,
    beads_per_g: float,
    days: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    cells/mL = (beads/mL) * (cells/bead)
    with beads/mL from mc_g_per_l & beads_per_g.

    mean_cells_per_mc/std_cells_per_mc must be length (days+1).
    time_h will be length (days+1) with 0,24,48,...
    """
    bml = beads_per_ml_from_mc(mc_g_per_l, beads_per_g)
    cells_ml = bml * mean_cells_per_mc
    cells_ml_std = bml * std_cells_per_mc

    time_h = np.arange(0, days + 1, dtype=float) * 24.0
    return time_h, cells_ml, cells_ml_std


def export_zip_csv(series: Dict[str, pd.DataFrame], summary: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", summary.to_csv(index=False))
        for name, df in series.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False))
    return buf.getvalue()


@dataclass
class StageResult:
    name: str
    days: int
    mc_g_per_l: float | None
    inoc_cells_ml: float
    end_cells_ml_total: float
    end_cells_ml_infectable: float
    end_cells_ml_multilayer: float


def run_seed_train(params: dict) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    BIO40 -> BIO200A -> BIO750E/F/G/H -> BIO1500A/B (mix only)
    Returns:
      series dict: each df has time_h, time_h_cum, TOTAL, INFECTABLE, MULTILAYER (+ std)
      summary table
    """
    # Volumes (mL)
    V_BIO40 = 40000.0
    V_BIO200A = 200000.0
    V_BIO750 = 750000.0
    V_BIO1500 = 1500000.0

    # Beads per g (FIXED OPTION 1: user gives beads/mg)
    beads_per_g = params["beads_per_mg"] * 1000.0

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

    # Helper: convert cells/mL inoc to cells/MC for grid seeding
    def inoc_cells_per_mc_from_cells_ml(cells_ml: float, mc_g_per_l: float) -> float:
        # beads/mL = mc_g/L * beads/g / 1000
        bml = beads_per_ml_from_mc(mc_g_per_l, beads_per_g)
        # cells/bead = cells/mL / beads/mL
        if bml <= 0:
            return 0.0
        return float(cells_ml / bml)

    # Helper: build df from simulation output (per metric)
    def build_df(name: str, days: int, mc_g_per_l: float, sim_out: dict, t_offset_h: float) -> pd.DataFrame:
        # sim_out: metric -> (mean_cells_per_mc, std_cells_per_mc)
        records = {"time_h": None}
        time_h = None

        for metric in ("TOTAL", "INFECTABLE", "MULTILAYER"):
            mean_mc, std_mc = sim_out[metric]
            th, mean_ml, std_ml = cells_ml_from_cells_per_mc(mean_mc, std_mc, mc_g_per_l, beads_per_g, days)
            if time_h is None:
                time_h = th
            records[f"{metric}_cells_ml"] = mean_ml
            records[f"{metric}_cells_ml_std"] = std_ml

        df = pd.DataFrame({"time_h": time_h})
        df["time_h_cum"] = df["time_h"] + float(t_offset_h)
        for k, v in records.items():
            if k == "time_h":
                continue
            df[k] = v
        df["stage"] = name
        return df

    results: Dict[str, pd.DataFrame] = {}
    summary_rows: list[StageResult] = []

    # ---------------- BIO40 ----------------
    inoc40_cells_ml = params["inoc_bio40_cells_ml"]
    inoc40_cells_per_mc = inoc_cells_per_mc_from_cells_ml(inoc40_cells_ml, MC_BIO40)

    sim40 = simulate_surface_mc(
        n_experiments=n,
        days=d40,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc40_cells_per_mc,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio40"],
        rng_seed=params["rng_seed"],
    )

    df40 = build_df("BIO40", d40, MC_BIO40, sim40, t_offset_h=0.0)
    results["BIO40"] = df40

    end40_total = float(df40["TOTAL_cells_ml"].iloc[-1])
    end40_inf = float(df40["INFECTABLE_cells_ml"].iloc[-1])
    end40_mul = float(df40["MULTILAYER_cells_ml"].iloc[-1])

    summary_rows.append(StageResult(
        name="BIO40", days=d40, mc_g_per_l=MC_BIO40,
        inoc_cells_ml=inoc40_cells_ml,
        end_cells_ml_total=end40_total,
        end_cells_ml_infectable=end40_inf,
        end_cells_ml_multilayer=end40_mul,
    ))

    # ---------------- BIO200A (transfer) ----------------
    tryps200 = params["trypsin_yield_bio200a"]
    inoc200_cells_ml = tryps200 * V_BIO40 * end40_total / V_BIO200A
    inoc200_cells_per_mc = inoc_cells_per_mc_from_cells_ml(inoc200_cells_ml, MC_BIO200A)

    sim200 = simulate_surface_mc(
        n_experiments=n,
        days=d200,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc200_cells_per_mc,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio200a"],
        rng_seed=params["rng_seed"] + 1,
    )

    offset200 = d40 * 24.0
    df200 = build_df("BIO200A", d200, MC_BIO200A, sim200, t_offset_h=offset200)
    results["BIO200A"] = df200

    end200_total = float(df200["TOTAL_cells_ml"].iloc[-1])
    end200_inf = float(df200["INFECTABLE_cells_ml"].iloc[-1])
    end200_mul = float(df200["MULTILAYER_cells_ml"].iloc[-1])

    summary_rows.append(StageResult(
        name="BIO200A", days=d200, mc_g_per_l=MC_BIO200A,
        inoc_cells_ml=float(inoc200_cells_ml),
        end_cells_ml_total=end200_total,
        end_cells_ml_infectable=end200_inf,
        end_cells_ml_multilayer=end200_mul,
    ))

    # ---------------- 4x BIO750 (split) ----------------
    tryps750 = params["trypsin_yield_bio750"]
    dist_EF = params["distribution_factor_EF"]
    dist_GH = params["distribution_factor_GH"]

    offset750 = (d40 + d200) * 24.0

    def run_750(tag: str, dist_factor: float, seed_add: int) -> pd.DataFrame:
        inoc750_cells_ml = dist_factor * tryps750 * V_BIO200A * end200_total / V_BIO750
        inoc750_cells_per_mc = inoc_cells_per_mc_from_cells_ml(inoc750_cells_ml, MC_BIO750)

        sim750 = simulate_surface_mc(
            n_experiments=n,
            days=d750,
            max_cells_setpoint=max_cells_setpoint,
            max_cells_sd=max_cells_sd,
            inoc_cells_per_mc_mean=inoc750_cells_per_mc,
            inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio750"],
            rng_seed=params["rng_seed"] + seed_add,
        )

        df750 = build_df(f"BIO750{tag}", d750, MC_BIO750, sim750, t_offset_h=offset750)

        end_total = float(df750["TOTAL_cells_ml"].iloc[-1])
        end_inf = float(df750["INFECTABLE_cells_ml"].iloc[-1])
        end_mul = float(df750["MULTILAYER_cells_ml"].iloc[-1])

        summary_rows.append(StageResult(
            name=f"BIO750{tag}", days=d750, mc_g_per_l=MC_BIO750,
            inoc_cells_ml=float(inoc750_cells_ml),
            end_cells_ml_total=end_total,
            end_cells_ml_infectable=end_inf,
            end_cells_ml_multilayer=end_mul,
        ))
        return df750

    results["BIO750E"] = run_750("E", dist_EF, 2)
    results["BIO750F"] = run_750("F", dist_EF, 3)
    results["BIO750G"] = run_750("G", dist_GH, 4)
    results["BIO750H"] = run_750("H", dist_GH, 5)

    # ---------------- BIO1500 (mix only, no growth modeled here) ----------------
    transferA = params["transfer_bio1500a"]
    transferB = params["transfer_bio1500b"]

    endE = float(results["BIO750E"]["TOTAL_cells_ml"].iloc[-1])
    endF = float(results["BIO750F"]["TOTAL_cells_ml"].iloc[-1])
    endG = float(results["BIO750G"]["TOTAL_cells_ml"].iloc[-1])
    endH = float(results["BIO750H"]["TOTAL_cells_ml"].iloc[-1])

    inoc1500A = transferA * ((endE * V_BIO750) + (endF * V_BIO750)) / V_BIO1500
    inoc1500B = transferB * ((endG * V_BIO750) + (endH * V_BIO750)) / V_BIO1500

    # keep as flat lines for display (your "Option 1" was conversion fix, not adding 1500 growth)
    t1500 = np.array([0.0, max(0, d1500) * 24.0], dtype=float)
    offset1500 = (d40 + d200 + d750) * 24.0

    def mix_df(name: str, inoc_ml: float) -> pd.DataFrame:
        df = pd.DataFrame({"time_h": t1500})
        df["time_h_cum"] = df["time_h"] + offset1500
        df["TOTAL_cells_ml"] = [inoc_ml, inoc_ml]
        df["INFECTABLE_cells_ml"] = [inoc_ml, inoc_ml]
        df["MULTILAYER_cells_ml"] = [0.0, 0.0]
        df["TOTAL_cells_ml_std"] = [0.0, 0.0]
        df["INFECTABLE_cells_ml_std"] = [0.0, 0.0]
        df["MULTILAYER_cells_ml_std"] = [0.0, 0.0]
        df["stage"] = name
        return df

    results["BIO1500A"] = mix_df("BIO1500A", float(inoc1500A))
    results["BIO1500B"] = mix_df("BIO1500B", float(inoc1500B))

    summary_rows.append(StageResult(
        name="BIO1500A", days=d1500, mc_g_per_l=None,
        inoc_cells_ml=float(inoc1500A),
        end_cells_ml_total=float(inoc1500A),
        end_cells_ml_infectable=float(inoc1500A),
        end_cells_ml_multilayer=0.0,
    ))
    summary_rows.append(StageResult(
        name="BIO1500B", days=d1500, mc_g_per_l=None,
        inoc_cells_ml=float(inoc1500B),
        end_cells_ml_total=float(inoc1500B),
        end_cells_ml_infectable=float(inoc1500B),
        end_cells_ml_multilayer=0.0,
    ))

    summary = pd.DataFrame([{
        "Stage": r.name,
        "Days": r.days,
        "MC_g/L": ("" if r.mc_g_per_l is None else r.mc_g_per_l),
        "Inoc_cells/mL": r.inoc_cells_ml,
        "End_TOTAL_cells/mL": r.end_cells_ml_total,
        "End_INFECTABLE_cells/mL": r.end_cells_ml_infectable,
        "End_MULTILAYER_cells/mL": r.end_cells_ml_multilayer,
    } for r in summary_rows])

    return results, summary


def plot_all_bioreactors_one_plot(series: Dict[str, pd.DataFrame], metric: str):
    """
    metric in {"TOTAL", "INFECTABLE", "MULTILAYER"}
    plots <metric>_cells_ml vs time_h_cum
    """
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    col = f"{metric}_cells_ml"

    for name, df in series.items():
        ax.plot(df["time_h_cum"], df[col], marker="o", label=name)

    ax.set_title(f"All bioreactors — {metric} cells/mL vs cumulative time")
    ax.set_xlabel("Time (h) — cumulative")
    ax.set_ylabel("Cells / mL")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    return fig


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Seed Train Monte Carlo (Cytodex)", layout="wide")
st.title("Seed Train Monte Carlo (Cytodex surface model) — FIXED beads unit (7087 per mg)")

with st.sidebar:
    st.header("Run controls")
    n_experiments = st.slider("Monte Carlo experiments (n)", 10, 5000, 100, step=10)

    st.subheader("Days per bioreactor")
    days_bio40 = st.slider("BIO40 days", 1, 20, 6)
    days_bio200a = st.slider("BIO200A days", 1, 20, 6)
    days_bio750 = st.slider("BIO750 (E/F/G/H) days", 1, 20, 6)
    days_bio1500 = st.slider("BIO1500 (A/B) days (display only)", 0, 20, 0)

    st.subheader("Surface capacity")
    max_cells_setpoint = st.slider("Max cells/MC setpoint", 50, 300, 140)
    max_cells_sd = st.slider("Max cells/MC SD", 0.0, 80.0, 23.0)

    st.subheader("Microcarriers / particles (FIX)")
    beads_per_mg = st.number_input("Beads per mg Cytodex (NOT per g)", value=7087.0, step=1.0)
    beads_per_g = beads_per_mg * 1000.0

    st.subheader("MC concentration (g/L)")
    mc_bio40 = st.number_input("BIO40 MC (g/L)", value=1.2, step=0.1)
    mc_bio200a = st.number_input("BIO200A MC (g/L)", value=4.0, step=0.1)
    mc_bio750 = st.number_input("BIO750 MC (g/L)", value=2.6, step=0.1)

    # Sanity check on beads/mL and theoretical max cells/mL
    bml40 = beads_per_ml_from_mc(float(mc_bio40), float(beads_per_g))
    st.caption(f"BIO40 beads/mL ≈ {bml40:,.0f}")
    st.caption(f"BIO40 max cells/mL @ {max_cells_setpoint} cells/bead ≈ {(bml40*max_cells_setpoint):,.0f}")

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

    st.subheader("Plot metric")
    plot_metric = st.radio("Plot metric", ["TOTAL", "INFECTABLE", "MULTILAYER"], horizontal=False)

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
        beads_per_mg=float(beads_per_mg),
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
        st.subheader(f"All bioreactors — one plot ({plot_metric} cells/mL vs cumulative time)")
        fig = plot_all_bioreactors_one_plot(series, metric=plot_metric)
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