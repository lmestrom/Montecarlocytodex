import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ------------------------
# Core grid model
# ------------------------
UNOCCUPIED, OCCUPIED, NEWLY_OCCUPIED, INHIBITED, MULTILAYER = 0, 1, 2, 3, 4


def simulate_surface_growth(
    n_experiments: int,
    time_steps: int,
    max_cells_setpoint: float,
    std_dev_maxcells: float,
    inoc_cells_intended: float,
    std_dev_inoc: float,
    seed: int,
):
    """
    Replicates your rules:
      - OCCUPIED tries to occupy a random UNOCCUPIED neighbor; else becomes INHIBITED.
      - INHIBITED becomes MULTILAYER if it has any neighbor INHIBITED or MULTILAYER (cycle>=1).
      - NEWLY_OCCUPIED converts to OCCUPIED each step.
    Returns per-time-step means and std devs for each state + OCCUPIED_averaged.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    counts_list = []

    for _exp in range(n_experiments):
        MAX_VEROcells = max(1.0, rng.normal(max_cells_setpoint, std_dev_maxcells))
        VERO_cells_NORMAL = max(0.0, rng.normal(inoc_cells_intended, std_dev_inoc))
        inoc_density = 0.0 if MAX_VEROcells == 0 else VERO_cells_NORMAL / MAX_VEROcells

        grid_size = max(1, int(round(math.sqrt(MAX_VEROcells))))
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        multilayer_cycle_grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        initial_occupied = int(grid_size * grid_size * inoc_density)
        if initial_occupied > grid_size * grid_size:
            # boundary condition: skip this experiment
            continue

        positions = py_rng.sample(range(grid_size * grid_size), initial_occupied) if initial_occupied > 0 else []
        for pos in positions:
            x, y = divmod(pos, grid_size)
            grid[x, y] = OCCUPIED

        def update_state(g):
            nonlocal multilayer_cycle_grid
            new_g = np.array(g, copy=True)

            for i in range(grid_size):
                for j in range(grid_size):
                    if g[i, j] == OCCUPIED:
                        neighbors = [
                            (x % grid_size, y % grid_size)
                            for x in range(i - 1, i + 2)
                            for y in range(j - 1, j + 2)
                            if (x, y) != (i, j)
                        ]
                        py_rng.shuffle(neighbors)

                        placed = False
                        for x, y in neighbors:
                            if new_g[x, y] == UNOCCUPIED:
                                new_g[x, y] = NEWLY_OCCUPIED
                                placed = True
                                break
                        if not placed:
                            new_g[i, j] = INHIBITED

                    if new_g[i, j] == INHIBITED:
                        neighbors = [
                            (x % grid_size, y % grid_size)
                            for x in range(i - 1, i + 2)
                            for y in range(j - 1, j + 2)
                            if (x, y) != (i, j)
                        ]
                        if any(new_g[x, y] in (INHIBITED, MULTILAYER) for x, y in neighbors):
                            multilayer_cycle_grid[i, j] += 1
                            if multilayer_cycle_grid[i, j] >= 1:
                                new_g[i, j] = MULTILAYER
                                multilayer_cycle_grid[i, j] = 0
                        else:
                            multilayer_cycle_grid[i, j] = 0

            return new_g

        def count_values(g):
            return {
                "UNOCCUPIED": int(np.count_nonzero(g == UNOCCUPIED)),
                "OCCUPIED": int(np.count_nonzero(g == OCCUPIED)),
                "NEWLY_OCCUPIED": int(np.count_nonzero(g == NEWLY_OCCUPIED)),
                "INHIBITED": int(np.count_nonzero(g == INHIBITED) + np.count_nonzero(g == MULTILAYER)),
                "MULTILAYER": int(np.count_nonzero(g == MULTILAYER)),
                "OCCUPIED_averaged": int(
                    np.count_nonzero(g == OCCUPIED)
                    + np.count_nonzero(g == NEWLY_OCCUPIED)
                    + np.count_nonzero(g == INHIBITED)
                    + np.count_nonzero(g == MULTILAYER)
                ),
            }

        per_exp = {k: [] for k in ["UNOCCUPIED", "OCCUPIED", "NEWLY_OCCUPIED", "INHIBITED", "MULTILAYER", "OCCUPIED_averaged"]}

        for _t in range(time_steps):
            grid = update_state(grid)
            c = count_values(grid)
            for k in per_exp:
                per_exp[k].append(c[k])
            grid[grid == NEWLY_OCCUPIED] = OCCUPIED

        counts_list.append(per_exp)

    if not counts_list:
        raise RuntimeError("All experiments skipped due to boundary condition; check parameters.")

    states = ["UNOCCUPIED", "OCCUPIED", "NEWLY_OCCUPIED", "INHIBITED", "MULTILAYER", "OCCUPIED_averaged"]
    means = {s: np.mean([exp[s] for exp in counts_list], axis=0) for s in states}
    stds = {s: np.std([exp[s] for exp in counts_list], axis=0, ddof=1) for s in states}

    means_df = pd.DataFrame(means)
    stds_df = pd.DataFrame(stds)
    return means_df, stds_df, len(counts_list)


def mu_from_surface(means_df, mu_max, K_s, oxygen):
    occ = means_df["OCCUPIED_averaged"].to_numpy(dtype=float)
    mu_surface = []
    for i in range(len(occ) - 1):
        N0, Nt = occ[i], occ[i + 1]
        if N0 <= 0 or Nt <= 0:
            mu_surface.append(np.nan)
        else:
            mu_surface.append((np.log(Nt) - np.log(N0)) / 24.0)  # per hour
    mu_surface = np.array(mu_surface, dtype=float)

    mu_oxygen = mu_max * oxygen / (K_s + oxygen)
    mu_oxygen_vec = np.full_like(mu_surface, mu_oxygen)
    mu_actual = np.fmin(mu_surface, mu_oxygen_vec)
    return mu_surface, mu_oxygen_vec, mu_actual


def cells_per_ml_trajectory(N0, mu_actual, time_steps):
    t_hours = np.array([0] + [24 * (i + 1) for i in range(time_steps)], dtype=float)
    N = [float(N0)]
    for mu in mu_actual:
        N.append(float(N[-1] * np.exp(mu * 24.0)))
    return t_hours[: len(N)], np.array(N)


@dataclass
class StageConfig:
    name: str
    MC_gL: float
    std_inoc: float
    K_s: float


def run_seed_train(n, time_steps, max_setpoint, std_max, mu_max, oxygen, seed_base=100):
    # BIO40 constants (from your script)
    V_BIO40 = 40000
    Inoculation_BIO40 = 48000
    MC_BIO40 = 1.2
    Cell_per_MC_BIO40 = (Inoculation_BIO40 / MC_BIO40) / 7087

    stages = [
        StageConfig("BIO40", MC_BIO40, 2.94, 0.96),
        StageConfig("BIO200A", 4.0, 1.0, 2.0),
        StageConfig("BIO750E", 2.6, 2.2, 1.5),
        StageConfig("BIO750F", 2.6, 2.2, 1.5),
        StageConfig("BIO750G", 2.6, 2.2, 1.5),
        StageConfig("BIO750H", 2.6, 2.2, 1.5),
    ]

    means_dfs, stds_dfs, trajs = {}, {}, {}
    rows = []

    # --- BIO40 ---
    means40, stds40, n_eff40 = simulate_surface_growth(
        n, time_steps, max_setpoint, std_max, Cell_per_MC_BIO40, stages[0].std_inoc, seed_base + 1
    )
    mu_s40, mu_o40, mu_a40 = mu_from_surface(means40, mu_max, stages[0].K_s, oxygen)
    N0_40 = stages[0].MC_gL * Cell_per_MC_BIO40 * 7087
    t40, N40 = cells_per_ml_trajectory(N0_40, mu_a40, time_steps)
    final40 = float(N40[-1])

    means_dfs["BIO40"], stds_dfs["BIO40"] = means40, stds40
    trajs["BIO40"] = pd.DataFrame({"t_hours": t40, "cells_per_ml": N40})
    rows.append({"Stage": "BIO40", "n_effective": n_eff40, "Inoc_cells_per_MC": Cell_per_MC_BIO40, "N0_cells_per_ml": N0_40, "Final_cells_per_ml": final40})

    # --- BIO200A inoculation from BIO40 ---
    V_BIO200A = 200000
    Trypsine_yield_BIO200A = 0.9
    Inoculation_BIO200A = Trypsine_yield_BIO200A * V_BIO40 * final40 / V_BIO200A
    Cell_per_MC_BIO200A = (Inoculation_BIO200A / stages[1].MC_gL) / 7087

    means200, stds200, n_eff200 = simulate_surface_growth(
        n, time_steps, max_setpoint, std_max, Cell_per_MC_BIO200A, stages[1].std_inoc, seed_base + 2
    )
    mu_s200, mu_o200, mu_a200 = mu_from_surface(means200, mu_max, stages[1].K_s, oxygen)
    N0_200 = stages[1].MC_gL * Cell_per_MC_BIO200A * 7087
    t200, N200 = cells_per_ml_trajectory(N0_200, mu_a200, time_steps)
    final200 = float(N200[-1])

    means_dfs["BIO200A"], stds_dfs["BIO200A"] = means200, stds200
    trajs["BIO200A"] = pd.DataFrame({"t_hours": t200, "cells_per_ml": N200})
    rows.append({"Stage": "BIO200A", "n_effective": n_eff200, "Inoc_cells_per_MC": Cell_per_MC_BIO200A, "N0_cells_per_ml": N0_200, "Final_cells_per_ml": final200})

    # --- BIO750s from BIO200A ---
    BIO200A_cell = final200
    V_BIO750 = 750000
    V_BIO200A_vol = 200000
    Trypsine_yield_BIO750 = 0.87
    dist_EF = 1.05 / 4
    dist_GH = 0.95 / 4

    def run_bio750(label, dist_factor, seed_add):
        Inoc = dist_factor * Trypsine_yield_BIO750 * V_BIO200A_vol * BIO200A_cell / V_BIO750
        cell_per_mc = (Inoc / 2.6) / 7087

        means, stds, n_eff = simulate_surface_growth(
            n, time_steps, max_setpoint, std_max, cell_per_mc, 2.2, seed_base + seed_add
        )
        mu_s, mu_o, mu_a = mu_from_surface(means, mu_max, 1.5, oxygen)
        N0 = 2.6 * cell_per_mc * 7087
        t, Nt = cells_per_ml_trajectory(N0, mu_a, time_steps)
        final = float(Nt[-1])

        means_dfs[label], stds_dfs[label] = means, stds
        trajs[label] = pd.DataFrame({"t_hours": t, "cells_per_ml": Nt})
        rows.append({"Stage": label, "n_effective": n_eff, "Inoc_cells_per_MC": cell_per_mc, "N0_cells_per_ml": N0, "Final_cells_per_ml": final})
        return final

    finalE = run_bio750("BIO750E", dist_EF, 10)
    finalF = run_bio750("BIO750F", dist_EF, 11)
    finalG = run_bio750("BIO750G", dist_GH, 12)
    finalH = run_bio750("BIO750H", dist_GH, 13)

    # --- BIO1500 inoculations ---
    V_BIO1500 = 1500000
    transfer_A = 0.85
    transfer_B = 0.85
    Inoc1500A = transfer_A * (finalE * V_BIO750 + finalF * V_BIO750) / V_BIO1500
    Inoc1500B = transfer_B * (finalG * V_BIO750 + finalH * V_BIO750) / V_BIO1500

    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {"Stage": "BIO1500A_inoc", "n_effective": np.nan, "Inoc_cells_per_MC": np.nan, "N0_cells_per_ml": np.nan, "Final_cells_per_ml": Inoc1500A}
    summary.loc[len(summary)] = {"Stage": "BIO1500B_inoc", "n_effective": np.nan, "Inoc_cells_per_MC": np.nan, "N0_cells_per_ml": np.nan, "Final_cells_per_ml": Inoc1500B}

    return summary, trajs, means_dfs, stds_dfs


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Vero seed-train Monte Carlo", layout="wide")
st.title("Vero/Cytodex seed-train Monte Carlo (grid model)")

with st.sidebar:
    st.header("Simulation settings")
    n = st.slider("n experiments per stage", 10, 5000, 100, step=10)
    time_steps = st.slider("time steps (days)", 2, 12, 6, step=1)
    max_setpoint = st.slider("MAX cells/MC (setpoint)", 50, 300, 140, step=5)
    std_max = st.slider("Std dev MAX cells/MC", 1, 80, 23, step=1)

    st.header("Oxygen cap (Monod)")
    mu_max = st.number_input("mu_max (1/h)", value=0.2887298, format="%.7f")
    oxygen = st.number_input("oxygen concentration (arbitrary units)", value=3.5, format="%.3f")

    seed_base = st.number_input("random seed base", value=100, step=1)

run = st.button("Run simulation")

if run:
    with st.spinner("Running…"):
        summary, trajs, means_dfs, stds_dfs = run_seed_train(
            n=n,
            time_steps=time_steps,
            max_setpoint=max_setpoint,
            std_max=std_max,
            mu_max=mu_max,
            oxygen=oxygen,
            seed_base=int(seed_base),
        )

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Trajectories (cells/mL)")
    stage = st.selectbox("Select stage", list(trajs.keys()))
    df_traj = trajs[stage]
    st.dataframe(df_traj, use_container_width=True)

    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(df_traj["t_hours"], df_traj["cells_per_ml"], marker="o")
    plt.xlabel("Time (h)")
    plt.ylabel("Cells/mL")
    plt.grid(True, which="both")
    st.pyplot(fig)

    st.subheader("Surface-state means (per grid)")
    st.dataframe(means_dfs[stage], use_container_width=True)
    st.subheader("Surface-state std devs (per grid)")
    st.dataframe(stds_dfs[stage], use_container_width=True)

    # Export to Excel in-memory
    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for k, df in trajs.items():
            df.to_excel(writer, sheet_name=f"{k}_traj", index=False)
        for k, df in means_dfs.items():
            df.to_excel(writer, sheet_name=f"{k}_means", index=False)
        for k, df in stds_dfs.items():
            df.to_excel(writer, sheet_name=f"{k}_std", index=False)

    st.download_button(
        "Download Excel",
        data=buffer.getvalue(),
        file_name=f"seed_train_sim_n{n}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Adjust parameters in the sidebar and click **Run simulation**.")