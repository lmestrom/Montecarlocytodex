# *Disclaimer: Experimental validation is still awaiting; this application is hypothetical modelling for exploratory use only.*

import io
import math
import random
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Plotly for prettier, interactive plots
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Core grid model (surface only)
# ----------------------------
UNOCCUPIED = 0
OCCUPIED = 1
NEWLY_OCCUPIED = 2
INHIBITED = 3
MULTILAYER = 4


def update_state(
    grid: np.ndarray,
    multilayer_cycle_grid: np.ndarray,
    spread_fraction: float = 1.0,  # throttle spreading attempts (0..1)
) -> Tuple[np.ndarray, np.ndarray]:
    """One growth step on a periodic grid."""
    grid_size = grid.shape[0]
    new_grid = np.copy(grid)

    spread_fraction = float(np.clip(spread_fraction, 0.0, 1.0))

    for i in range(grid_size):
        for j in range(grid_size):

            # spreading from OCCUPIED
            if grid[i, j] == OCCUPIED:

                # oxygen-limitation throttle
                if random.random() > spread_fraction:
                    continue

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
    rng_seed: int = 1,
    spread_fraction_by_day: Optional[Callable[[int], float]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Monte Carlo simulation on surface (cells/bead-like grid).
    Returns mean & std trajectories for:
      TOTAL, INFECTABLE, MULTILAYER
    Each trajectory is length (days+1) including day0 (initial state).

    spread_fraction_by_day(day_index) -> float in [0,1]
      day_index is 0-based (0 = first simulated day step)
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

        c0 = count_states(grid)
        series_total = [c0["TOTAL"]]
        series_inf = [c0["INFECTABLE"]]
        series_mul = [c0["MULTILAYER"]]

        for _day in range(days):
            sf = 1.0
            if spread_fraction_by_day is not None:
                sf = float(spread_fraction_by_day(_day))

            grid, multilayer_cycle_grid = update_state(
                grid,
                multilayer_cycle_grid,
                spread_fraction=sf
            )
            counts = count_states(grid)

            series_total.append(counts["TOTAL"])
            series_inf.append(counts["INFECTABLE"])
            series_mul.append(counts["MULTILAYER"])

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
    """beads/mL = (g/L)*(beads/g)/1000"""
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
    mean_cells_per_mc/std_cells_per_mc must be length (days+1).
    """
    bml = beads_per_ml_from_mc(mc_g_per_l, beads_per_g)
    cells_ml = bml * mean_cells_per_mc
    cells_ml_std = bml * std_cells_per_mc

    time_days = np.arange(0, days + 1, dtype=float)
    return time_days, cells_ml, cells_ml_std


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
      series dict: each df has time_days, time_days_cum, TOTAL, INFECTABLE, MULTILAYER (+ std)
      summary table
    """
    # Volumes (mL)
    V_BIO40 = 40000.0
    V_BIO200A = 200000.0
    V_BIO750 = 750000.0
    V_BIO1500 = 1500000.0

    beads_per_g = params["beads_per_mg"] * 1000.0

    MC_BIO40 = params["mc_bio40_g_per_l"]
    MC_BIO200A = params["mc_bio200a_g_per_l"]
    MC_BIO750 = params["mc_bio750_g_per_l"]

    d40 = params["days_bio40"]
    d200 = params["days_bio200a"]
    d750 = params["days_bio750"]
    d1500 = params["days_bio1500"]

    n = params["n_experiments"]

    max_cells_setpoint = params["max_cells_setpoint"]
    max_cells_sd = params["max_cells_sd"]

    def inoc_cells_per_mc_from_cells_ml(cells_ml: float, mc_g_per_l: float) -> float:
        bml = beads_per_ml_from_mc(mc_g_per_l, beads_per_g)
        if bml <= 0:
            return 0.0
        return float(cells_ml / bml)

    def build_df(name: str, days: int, mc_g_per_l: float, sim_out: dict, t_offset_days: float) -> pd.DataFrame:
        time_days = None
        cols = {}
        for metric in ("TOTAL", "INFECTABLE", "MULTILAYER"):
            mean_mc, std_mc = sim_out[metric]
            td, mean_ml, std_ml = cells_ml_from_cells_per_mc(mean_mc, std_mc, mc_g_per_l, beads_per_g, days)
            if time_days is None:
                time_days = td
            cols[f"{metric}_cells_ml"] = mean_ml
            cols[f"{metric}_cells_ml_std"] = std_ml

        df = pd.DataFrame({"time_days": time_days})
        df["time_days_cum"] = df["time_days"] + float(t_offset_days)
        for k, v in cols.items():
            df[k] = v
        df["stage"] = name
        return df

    results: Dict[str, pd.DataFrame] = {}
    summary_rows: list[StageResult] = []

    # BIO40
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

    df40 = build_df("BIO40", d40, MC_BIO40, sim40, t_offset_days=0.0)
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

    # BIO200A
    tryps200 = params["trypsin_yield_bio200a"]
    inoc200_cells_ml = tryps200 * V_BIO40 * end40_total / V_BIO200A
    inoc200_cells_per_mc = inoc_cells_per_mc_from_cells_ml(inoc200_cells_ml, MC_BIO200A)

    def bio200_spread_fn(day0_based: int) -> float:
        if (not params["limit_bio200_oxygen"]) or (day0_based < params["bio200_limit_start_day"]):
            return 1.0
        return float(np.clip(params["bio200_spread_fraction"], 0.0, 1.0))

    sim200 = simulate_surface_mc(
        n_experiments=n,
        days=d200,
        max_cells_setpoint=max_cells_setpoint,
        max_cells_sd=max_cells_sd,
        inoc_cells_per_mc_mean=inoc200_cells_per_mc,
        inoc_cells_per_mc_sd=params["inoc_sd_cells_per_mc_bio200a"],
        rng_seed=params["rng_seed"] + 1,
        spread_fraction_by_day=bio200_spread_fn,
    )

    offset200 = float(d40)
    df200 = build_df("BIO200A", d200, MC_BIO200A, sim200, t_offset_days=offset200)
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

    # 4x BIO750
    tryps750 = params["trypsin_yield_bio750"]
    dist_EF = params["distribution_factor_EF"]
    dist_GH = params["distribution_factor_GH"]

    offset750 = float(d40 + d200)

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

        df750 = build_df(f"BIO750{tag}", d750, MC_BIO750, sim750, t_offset_days=offset750)

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

    # BIO1500 (mix only)
    transferA = params["transfer_bio1500a"]
    transferB = params["transfer_bio1500b"]

    endE = float(results["BIO750E"]["TOTAL_cells_ml"].iloc[-1])
    endF = float(results["BIO750F"]["TOTAL_cells_ml"].iloc[-1])
    endG = float(results["BIO750G"]["TOTAL_cells_ml"].iloc[-1])
    endH = float(results["BIO750H"]["TOTAL_cells_ml"].iloc[-1])

    inoc1500A = transferA * ((endE * V_BIO750) + (endF * V_BIO750)) / V_BIO1500
    inoc1500B = transferB * ((endG * V_BIO750) + (endH * V_BIO750)) / V_BIO1500

    t1500 = np.array([0.0, float(max(0, d1500))], dtype=float)
    offset1500 = float(d40 + d200 + d750)

    def mix_df(name: str, inoc_ml: float) -> pd.DataFrame:
        df = pd.DataFrame({"time_days": t1500})
        df["time_days_cum"] = df["time_days"] + offset1500
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


# ----------------------------
# Prettier plot helpers (Plotly)
# ----------------------------
def plot_all_bioreactors_plotly(series: Dict[str, pd.DataFrame], metric: str) -> go.Figure:
    col = f"{metric}_cells_ml"
    fig = go.Figure()
    for name, df in series.items():
        fig.add_trace(go.Scatter(
            x=df["time_days_cum"],
            y=df[col],
            mode="lines+markers",
            name=name,
        ))
    fig.update_layout(
        template="simple_white",
        title=f"All bioreactors — {metric} cells/mL vs cumulative days",
        xaxis_title="Time (days) — cumulative",
        yaxis_title="Cells / mL",
        legend_title="Stage",
        margin=dict(l=20, r=20, t=60, b=20),
        height=520,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def kpi_block(summary: pd.DataFrame) -> dict:
    """Compute headline KPIs from the last stage entries."""
    # Choose the last non-empty stage as "final"
    # (BIO1500A/B are mix-only; still meaningful)
    final_row = summary.iloc[-1].copy()

    final_total = float(final_row["End_TOTAL_cells/mL"])
    final_inf = float(final_row["End_INFECTABLE_cells/mL"])
    final_mul = float(final_row["End_MULTILAYER_cells/mL"])

    inf_frac = (final_inf / final_total) if final_total > 0 else 0.0
    mul_frac = (final_mul / final_total) if final_total > 0 else 0.0

    return {
        "final_stage": str(final_row["Stage"]),
        "final_total": final_total,
        "final_inf": final_inf,
        "final_mul": final_mul,
        "inf_frac": inf_frac,
        "mul_frac": mul_frac,
    }


def format_sci(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if abs(x) >= 1e6 or (abs(x) > 0 and abs(x) < 1e-2):
        return f"{x:.3e}"
    return f"{x:,.0f}"


# ----------------------------
# Streamlit UI (polished)
# ----------------------------
st.set_page_config(
    page_title="Digital twin of Vero cell seed train",
    page_icon="🧫",
    layout="wide"
)

# Minimal CSS polish
st.markdown("""
<style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
    [data-testid="stSidebar"] { padding-top: 1.0rem; }
    .hero {
        background: white;
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.06);
    }
    .subtle {
        color: rgba(0,0,0,0.65);
        font-size: 0.95rem;
        line-height: 1.35rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 14px 14px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 6px 14px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Hero header
st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; gap:10px;">
    <div style="font-size:28px;">🧫</div>
    <div style="font-size:26px; font-weight:700;">Digital twin of Vero cell seed train</div>
  </div>
  <div class="subtle" style="margin-top:6px;">
    Hypothetical modelling for exploratory use only. Experimental validation is pending.
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# Sidebar controls (grouped)
with st.sidebar:
    st.title("Control room")

    mode = st.radio("View mode", ["Executive", "Process Engineer", "Scientist"], horizontal=True)

    st.divider()
    st.header("Run controls")
    n_experiments = st.slider("Monte Carlo experiments (n)", 10, 10000, 1000, step=10)
    rng_seed = st.number_input("RNG seed", value=1, step=1)

    st.divider()
    with st.expander("Seed train schedule (days)", expanded=(mode != "Executive")):
        days_bio40 = st.slider("BIO40 days", 1, 20, 5)
        days_bio200a = st.slider("BIO200A days", 1, 20, 5)
        days_bio750 = st.slider("BIO750 (E/F/G/H) days", 1, 20, 4)
        days_bio1500 = st.slider("BIO1500 (A/B) days (display only)", 0, 20, 0)

    with st.expander("Surface capacity (cells/MC)", expanded=(mode == "Scientist")):
        max_cells_setpoint = st.slider("Max cells/MC setpoint", 50, 300, 140)
        max_cells_sd = st.slider("Max cells/MC SD", 0.0, 80.0, 23.0)

    with st.expander("Microcarriers & concentrations", expanded=True):
        beads_per_mg = st.number_input("Beads per mg Cytodex (NOT per g)", value=7087.0, step=1.0)
        beads_per_g = beads_per_mg * 1000.0

        mc_bio40 = st.number_input("BIO40 MC (g/L)", value=1.2, step=0.1)
        mc_bio200a = st.number_input("BIO200A MC (g/L)", value=4.0, step=0.1)
        mc_bio750 = st.number_input("BIO750 MC (g/L)", value=2.6, step=0.1)

        # quick sanity checks
        bml40 = beads_per_ml_from_mc(float(mc_bio40), float(beads_per_g))
        st.caption(f"BIO40 beads/mL ≈ {bml40:,.0f}")
        st.caption(f"BIO40 max cells/mL @ {max_cells_setpoint} cells/bead ≈ {(bml40*max_cells_setpoint):,.0f}")

    with st.expander("Inoculation & variability", expanded=(mode != "Executive")):
        inoc_bio40_cells_ml = st.number_input("BIO40 inoculation (cells/mL)", value=48000.0, step=1000.0)
        inoc_sd_bio40 = st.number_input("BIO40 inoc SD (cells/MC)", value=2.94, step=0.1)
        inoc_sd_bio200a = st.number_input("BIO200A inoc SD (cells/MC)", value=1.0, step=0.1)
        inoc_sd_bio750 = st.number_input("BIO750 inoc SD (cells/MC)", value=2.2, step=0.1)

    with st.expander("Transfer / trypsin yields", expanded=(mode != "Executive")):
        trypsin_yield_bio200a = st.slider("Trypsin yield BIO200A", 0.6, 1.0, 0.90, step=0.01)
        trypsin_yield_bio750 = st.slider("Trypsin yield BIO750", 0.6, 1.0, 0.87, step=0.01)
        transfer_bio1500a = st.slider("Transfer factor BIO1500A", 0.6, 1.0, 0.85, step=0.01)
        transfer_bio1500b = st.slider("Transfer factor BIO1500B", 0.6, 1.0, 0.85, step=0.01)

    with st.expander("Distribution factors", expanded=(mode == "Scientist")):
        distribution_factor_EF = st.number_input("Distribution factor (E/F)", value=1.05 / 4, step=0.01)
        distribution_factor_GH = st.number_input("Distribution factor (G/H)", value=0.95 / 4, step=0.01)

    with st.expander("BIO200A oxygen limitation", expanded=True):
        limit_bio200_oxygen = st.checkbox("Enable O2-limited growth in BIO200A", value=True)
        default_start_day = min(4, int(days_bio200a))
        bio200_limit_start_day = st.slider("Start day (BIO200A)", 0, int(days_bio200a), default_start_day)
        bio200_spread_fraction = st.slider(
            "Growth throttle (fraction of cells that can spread)", 0.0, 1.0, 0.15, step=0.01
        )
        st.caption("Interpretation: fraction of occupied cells that attempt to spread per day under limitation.")

    st.divider()
    plot_metric = st.radio("Primary plot metric", ["TOTAL", "INFECTABLE", "MULTILAYER"], horizontal=True)
    run_btn = st.button("Run simulation", type="primary", use_container_width=True)

# Cache the expensive simulation
@st.cache_data(show_spinner=False)
def _cached_run_seed_train(params: dict) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    return run_seed_train(params)

# Main tabs
tab_results, tab_sweep, tab_raw, tab_assump = st.tabs(["📈 Results", "🧪 MC Sweep", "🧾 Raw data", "ℹ️ Assumptions"])

if not run_btn:
    with tab_results:
        st.info("Set parameters in the sidebar and click **Run simulation**.")
else:
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
        limit_bio200_oxygen=bool(limit_bio200_oxygen),
        bio200_limit_start_day=int(bio200_limit_start_day),
        bio200_spread_fraction=float(bio200_spread_fraction),
    )

    with st.status("Running Monte Carlo simulation…", expanded=False) as status:
        series, summary = _cached_run_seed_train(params)
        status.update(label="Simulation complete", state="complete")

    # RESULTS TAB
    with tab_results:
        # KPI strip
        kpis = kpi_block(summary)
        c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1], gap="large")
        with c1:
            st.metric("Final stage", kpis["final_stage"])
        with c2:
            st.metric("End TOTAL (cells/mL)", format_sci(kpis["final_total"]))
        with c3:
            st.metric("End INFECTABLE (cells/mL)", format_sci(kpis["final_inf"]))
        with c4:
            st.metric("End MULTILAYER (cells/mL)", format_sci(kpis["final_mul"]))
        with c5:
            st.metric("Infectable fraction", f"{100*kpis['inf_frac']:.1f}%")

        # Narrative (mode-dependent)
        if mode == "Executive":
            st.write(
                "This model estimates seed-train cell expansion across stages. "
                "Key decision signals are the **infectable fraction** and the **multilayer burden**, "
                "which may indicate diminishing returns at higher effective surface loading."
            )
        elif mode == "Process Engineer":
            st.write(
                "Use this view to stress-test assumptions across MC loading, inoculation variability, "
                "and BIO200 oxygen limitation. Focus on stage-by-stage trends and the effect on **infectable cells/mL**."
            )
        else:
            st.write(
                "Scientist mode exposes more variability controls and assumptions. "
                "Remember: grid capacity is a proxy for cells/bead-like surface capacity."
            )

        # Main plot + summary + download
        left, right = st.columns([2.2, 1.0], gap="large")
        with left:
            fig = plot_all_bioreactors_plotly(series, metric=plot_metric)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Summary table")
            st.dataframe(summary, use_container_width=True, height=380)

            zip_bytes = export_zip_csv(series, summary)
            st.download_button(
                "Download results (ZIP of CSVs)",
                data=zip_bytes,
                file_name=f"seed_train_sim_n{int(n_experiments)}.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # MC SWEEP TAB
    with tab_sweep:
        st.subheader("BIO750E — MC sweep (0–6 g/L)")
        st.caption("Re-runs BIO750E growth at different MC loadings; plots TOTAL and INFECTABLE (cells/mL).")

        # Pull BIO750E inoculum (cells/mL) from summary table
        try:
            inoc750E_cells_ml = float(summary.loc[summary["Stage"] == "BIO750E", "Inoc_cells/mL"].iloc[0])
        except Exception:
            inoc750E_cells_ml = None

        if inoc750E_cells_ml is None or (isinstance(inoc750E_cells_ml, float) and np.isnan(inoc750E_cells_ml)):
            st.warning("Could not determine BIO750E inoculation from summary; MC sweep plot skipped.")
        else:
            mc_sweep = list(range(0, 7))  # 0..6 g/L
            d750_local = int(days_bio750)
            beads_per_g_local = float(beads_per_mg) * 1000.0

            sweep_rows = []
            for mc_val_int in mc_sweep:
                mc_val = float(mc_val_int)

                if mc_val <= 0.0:
                    for day in range(d750_local + 1):
                        sweep_rows.append({
                            "day": float(day),
                            "MC_g/L": mc_val,
                            "TOTAL_cells/mL": 0.0,
                            "INFECTABLE_cells/mL": 0.0
                        })
                    continue

                bml = beads_per_ml_from_mc(mc_val, beads_per_g_local)
                inoc_cells_per_mc = 0.0 if bml <= 0 else float(inoc750E_cells_ml / bml)

                sim750_mc = simulate_surface_mc(
                    n_experiments=int(n_experiments),
                    days=d750_local,
                    max_cells_setpoint=float(max_cells_setpoint),
                    max_cells_sd=float(max_cells_sd),
                    inoc_cells_per_mc_mean=float(inoc_cells_per_mc),
                    inoc_cells_per_mc_sd=float(inoc_sd_bio750),
                    rng_seed=int(rng_seed) + 100 + mc_val_int,
                )

                time_days, total_ml, _ = cells_ml_from_cells_per_mc(
                    sim750_mc["TOTAL"][0], sim750_mc["TOTAL"][1],
                    mc_val, beads_per_g_local, d750_local
                )
                _, inf_ml, _ = cells_ml_from_cells_per_mc(
                    sim750_mc["INFECTABLE"][0], sim750_mc["INFECTABLE"][1],
                    mc_val, beads_per_g_local, d750_local
                )

                for k in range(len(time_days)):
                    sweep_rows.append({
                        "day": float(time_days[k]),
                        "MC_g/L": mc_val,
                        "TOTAL_cells/mL": float(total_ml[k]),
                        "INFECTABLE_cells/mL": float(inf_ml[k]),
                    })

            df_sweep = pd.DataFrame(sweep_rows)

            # Plotly lines
            fig_total = px.line(
                df_sweep, x="day", y="TOTAL_cells/mL", color="MC_g/L",
                markers=True, template="simple_white",
                title="BIO750E — TOTAL cells/mL vs days (MC sweep)"
            )
            fig_total.update_layout(height=420, legend_title="MC (g/L)")
            st.plotly_chart(fig_total, use_container_width=True)

            fig_inf = px.line(
                df_sweep, x="day", y="INFECTABLE_cells/mL", color="MC_g/L",
                markers=True, template="simple_white",
                title="BIO750E — INFECTABLE cells/mL vs days (MC sweep)"
            )
            fig_inf.update_layout(height=420, legend_title="MC (g/L)")
            st.plotly_chart(fig_inf, use_container_width=True)

            with st.expander("Show BIO750E MC sweep data table"):
                st.dataframe(df_sweep, use_container_width=True)

    # RAW DATA TAB
    with tab_raw:
        st.subheader("Raw time series (per bioreactor)")
        st.caption("Each table includes cumulative time and mean ± std (cells/mL) per metric.")
        tabs = st.tabs(list(series.keys()))
        for tab, (name, df) in zip(tabs, series.items()):
            with tab:
                st.dataframe(df, use_container_width=True, height=520)

    # ASSUMPTIONS TAB
    with tab_assump:
        st.subheader("Model assumptions & interpretation")
        st.markdown("""
- **Purpose:** exploratory, not a release / validation tool.
- **Grid model:** each microcarrier is approximated as a 2D grid of capacity ~ `max_cells_setpoint` (with variability).
- **States:** OCCUPIED/NEWLY_OCCUPIED/INHIBITED/MULTILAYER represent simplified surface crowding dynamics.
- **Infectable definition:** `OCCUPIED + NEWLY_OCCUPIED + INHIBITED` (excludes MULTILAYER).
- **BIO200 oxygen limitation:** implemented as a throttle on daily spread attempts from a user-defined start day.
- **BIO1500 stages:** mixing only (no growth modelled); displayed as flat lines.
        """.strip())

        st.warning("If you want, we can add confidence bands (mean ± std shading) and per-stage delta metrics (e.g., growth factor).")