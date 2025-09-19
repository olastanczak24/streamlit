# app_minikube_ports.py — Agent Simulation (Kube & Krateo) with Auto-Run & Cost-Reduction Viz
"""
Run:
  pip install streamlit numpy pandas matplotlib
  streamlit run app_minikube_ports.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Polygon

plt.switch_backend("Agg")  # safe for Streamlit servers

# =========================
# Data model
# =========================
@dataclass
class Task:
    task_id: int
    service_name: str
    type: str              # "stateless" | "stateful" | "cron"
    complexity: int        # 1..5
    criticality: str       # "low" | "med" | "high"

    discovered_at: Optional[int] = None
    planned_at: Optional[int] = None
    migrated_at: Optional[int] = None
    observed_at: Optional[int] = None  # final stage


@dataclass
class Context:
    rng: np.random.Generator
    tick: int = 0
    logs: List[Tuple[int, str, str]] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

    def log(self, agent: str, msg: str) -> None:
        self.logs.append((self.tick, agent, msg))


# =========================
# Helpers
# =========================
SERVICES = [
    "checkout", "catalog", "search", "payments", "notifications",
    "analytics", "user-profile", "inventory", "reco-engine", "billing"
]

def generate_tasks(n: int, rng: np.random.Generator) -> List[Task]:
    types = ["stateless", "stateful", "cron"]
    crits = ["low", "med", "high"]
    tasks: List[Task] = []
    for i in range(n):
        svc = SERVICES[i % len(SERVICES)] + f"-{i}"
        t = rng.choice(types)
        c = int(rng.integers(1, 6))
        crit = rng.choice(crits, p=[0.5, 0.35, 0.15])
        tasks.append(Task(task_id=i, service_name=svc, type=t, complexity=c, criticality=crit))
    return tasks


# =========================
# Agents
# =========================
class BaseAgent:
    name = "Agent"

    def step(self, task: Task, ctx: Context) -> None:
        raise NotImplementedError

    def effort_delay(self, task: Task, ctx: Context, base: int) -> int:
        """
        Synthetic tick delay based on complexity/type.
        Higher for stateful, scaled by complexity, occasional blocker.
        """
        mult = 1
        if task.type == "stateful":
            mult += 1
        elif task.type == "cron":
            mult += 0
        mult += max(0, task.complexity - 1) // 2
        if ctx.rng.random() < 0.08:
            mult += 1
            ctx.log(self.name, f"Task {task.task_id}: blocker encountered")
        return max(1, base * mult)


class DiscoveryAgent(BaseAgent):
    name = "Discovery"
    def step(self, task: Task, ctx: Context) -> None:
        if task.discovered_at is None:
            task.discovered_at = ctx.tick + self.effort_delay(task, ctx, base=1)
            ctx.log(self.name, f"Discovered dependencies for {task.service_name}")


class PlanningAgent(BaseAgent):
    name = "Planning"
    def step(self, task: Task, ctx: Context) -> None:
        if task.discovered_at is not None and task.planned_at is None and ctx.tick >= task.discovered_at:
            task.planned_at = ctx.tick + self.effort_delay(task, ctx, base=1)
            ctx.log(self.name, f"Planned K8s objects & ingress for {task.service_name}")


class MigrationAgent(BaseAgent):
    name = "Migration"
    def step(self, task: Task, ctx: Context) -> None:
        if task.planned_at is not None and task.migrated_at is None and ctx.tick >= task.planned_at:
            task.migrated_at = ctx.tick + self.effort_delay(task, ctx, base=2)
            ctx.log(self.name, f"Applied Helm/YAML + Krateo templates for {task.service_name}")


class ObservabilityAgent(BaseAgent):
    name = "Observability"
    def step(self, task: Task, ctx: Context) -> None:
        if task.migrated_at is not None and task.observed_at is None and ctx.tick >= task.migrated_at:
            task.observed_at = ctx.tick + self.effort_delay(task, ctx, base=1)
            ctx.log(self.name, f"Enabled OTel & Splunk Connect; SLOs for {task.service_name}")


# =========================
# Orchestrator (now supports per-tick stepping)
# =========================
class Orchestrator:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.agents = [DiscoveryAgent(), PlanningAgent(), MigrationAgent(), ObservabilityAgent()]

    def priority(self, task: Task) -> int:
        crit = {"low": 0, "med": 1, "high": 2}[task.criticality]
        return -(crit * 10 + task.complexity)

    def tick_once(self, tasks: List[Task]) -> None:
        for task in sorted(tasks, key=self.priority):
            for agent in self.agents:
                before = (task.discovered_at, task.planned_at, task.migrated_at, task.observed_at)
                agent.step(task, self.ctx)
                after = (task.discovered_at, task.planned_at, task.migrated_at, task.observed_at)
                if before != after:
                    break  # one transition max per tick per task

    def step(self, tasks: List[Task]) -> bool:
        """
        Advance the simulation by one tick.
        Returns True if all tasks are finished after this tick; else False.
        """
        self.tick_once(tasks)
        t = self.ctx.tick
        all_done = all(x.observed_at is not None and t >= x.observed_at for x in tasks)
        if not all_done:
            self.ctx.tick = t + 1
        return all_done


# =========================
# Ports & Ships visualization
# =========================
PORTS = {
    "backlog": (0.05, 0.5),
    "discovery": (0.25, 0.78),
    "planning": (0.45, 0.25),
    "migration": (0.65, 0.72),
    "observability": (0.90, 0.5),
}
LANES = [
    ("backlog", "discovery"),
    ("discovery", "planning"),
    ("planning", "migration"),
    ("migration", "observability"),
]

CRIT_COLOR = {"low": "#4caf50", "med": "#ffb300", "high": "#e53935"}
TYPE_EDGE = {"stateless": "#2e7d32", "stateful": "#1e88e5", "cron": "#6a1b9a"}

def _interp(p1, p2, t: float) -> Tuple[float, float]:
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

def task_position_at_tick(task: Task, tick: int) -> Tuple[float, float]:
    t0 = task.discovered_at
    t1 = task.planned_at
    t2 = task.migrated_at
    t3 = task.observed_at

    if t0 is None or tick <= t0:
        total = max(t0 or 1, 1)
        frac = np.clip((tick) / total, 0.0, 1.0)
        return _interp(PORTS["backlog"], PORTS["discovery"], frac)

    if t1 is None or tick <= t1:
        total = max((t1 or (t0 + 1)) - t0, 1)
        frac = np.clip((tick - t0) / total, 0.0, 1.0)
        return _interp(PORTS["discovery"], PORTS["planning"], frac)

    if t2 is None or tick <= t2:
        total = max((t2 or (t1 + 1)) - t1, 1)
        frac = np.clip((tick - t1) / total, 0.0, 1.0)
        return _interp(PORTS["planning"], PORTS["migration"], frac)

    if t3 is None or tick <= t3:
        total = max((t3 or (t2 + 1)) - t2, 1)
        frac = np.clip((tick - t2) / total, 0.0, 1.0)
        return _interp(PORTS["migration"], PORTS["observability"], frac)

    return PORTS["observability"]

def draw_ports_background(ax):
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#e6f3ff", edgecolor="none", zorder=0))
    for a, b in LANES:
        p1, p2 = PORTS[a], PORTS[b]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-", linewidth=1.5,
                                     linestyle="--", alpha=0.35, zorder=1))
    for name, (x, y) in PORTS.items():
        ax.add_patch(Circle((x, y), 0.035, facecolor="#cfd8dc", edgecolor="#455a64", linewidth=1.5, zorder=3))
        ax.add_patch(Rectangle((x - 0.05, y - 0.01), 0.1, 0.02, facecolor="#90a4ae", edgecolor="#37474f", zorder=2))
        ax.text(x, y + 0.07, name.capitalize(), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#263238", zorder=4)

def draw_ship(ax, x: float, y: float, task: Task, scale: float = 0.03):
    """
    More ship-like drawing:
    - tapered bow & stern polygon hull
    - colored by criticality
    - edge color by type (stateless/stateful/cron)
    - stack & portholes, subtle wake
    """
    # hull
    L, H = 6 * scale, 2 * scale
    bow = (x + L * 0.5, y)
    stern = (x - L * 0.5, y)
    hull_pts = [
        (stern[0] + L * 0.10, y - H * 0.5),              # stern lower
        (bow[0] - L * 0.12, y - H * 0.5),               # near bow lower
        (bow[0], y),                                     # bow tip
        (bow[0] - L * 0.12, y + H * 0.5),               # near bow upper
        (stern[0] + L * 0.10, y + H * 0.5),             # stern upper
        (stern[0], y),                                   # stern tip
    ]
    hull = Polygon(hull_pts, closed=True,
                   facecolor=CRIT_COLOR[task.criticality],
                   edgecolor=TYPE_EDGE[task.type],
                   linewidth=1.0, alpha=0.95, zorder=10)
    ax.add_patch(hull)

    # deckhouse / stack
    stack_w, stack_h = L * 0.10, H * 0.9
    stack = Rectangle((x - L * 0.05, y + H * 0.1), stack_w, stack_h,
                      facecolor="#eeeeee", edgecolor="#455a64", linewidth=0.8, zorder=12)
    ax.add_patch(stack)

    # portholes (number reflects complexity)
    holes = min(4, max(1, task.complexity // 2 + 1))
    for i in range(holes):
        px = x - L * 0.25 + i * (L * 0.15)
        py = y
        ax.add_patch(Circle((px, py), radius=0.12 * scale,
                            facecolor="#bbdefb", edgecolor="#37474f", linewidth=0.5, zorder=13))

    # wake (slightly curved lines behind stern; strength varies with complexity)
    wake_len = L * (0.25 + 0.1 * (task.complexity - 1))
    ax.plot([stern[0], stern[0] - wake_len], [y, y - 0.01], linewidth=0.6, alpha=0.5, zorder=5)
    ax.plot([stern[0], stern[0] - wake_len * 0.9], [y, y + 0.012], linewidth=0.5, alpha=0.4, zorder=5)

    # service initials
    initials = "".join([tok[0] for tok in task.service_name.split("-")[:2]]).upper()
    ax.text(x, y - H * 0.9, initials, ha="center", va="top", fontsize=6, color="#0d1b2a", zorder=13)

def ships_frame(tasks: List[Task], tick: int):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_ports_background(ax)

    rng = np.random.default_rng(12345)  # stable jitter to avoid overlaps
    jit = {t.task_id: (rng.random() - 0.5) * 0.028 for t in tasks}

    for t in tasks:
        x, y = task_position_at_tick(t, tick)
        y += jit[t.task_id]
        draw_ship(ax, x, y, t)

    ax.set_title("Shipping lanes between departments — tasks as container ships", fontsize=12, pad=8)
    plt.tight_layout()
    return fig


# =========================
# Costs model & charts
# =========================
def per_task_cost_legacy(t: Task) -> float:
    """
    Synthetic legacy cost (e.g., VM/server, licenses, ops overhead).
    Tunable but stable: stateful & high-criticality cost more.
    """
    crit_mult = {"low": 1.0, "med": 1.35, "high": 1.8}[t.criticality]
    type_mult = {"stateless": 1.0, "stateful": 1.6, "cron": 0.6}[t.type]
    base = 100  # arbitrary unit per time step
    return base * crit_mult * type_mult * (0.6 + 0.1 * t.complexity)

def per_task_cost_k8s(t: Task) -> float:
    """
    Synthetic k8s cost (shared nodes, autoscaling, infra as code).
    Generally lower, slight redundancy overhead baked in.
    """
    crit_mult = {"low": 0.7, "med": 0.95, "high": 1.2}[t.criticality]
    type_mult = {"stateless": 0.7, "stateful": 1.1, "cron": 0.4}[t.type]
    base = 100
    return base * crit_mult * type_mult * (0.5 + 0.08 * t.complexity)

def cost_at_tick(tasks: List[Task], tick: int) -> Tuple[float, float, float]:
    """
    Returns (legacy_cost, k8s_cost, total_cost) at given tick.
    A task counts as 'k8s' once it's fully Observed (observed_at <= tick).
    """
    legacy, k8s = 0.0, 0.0
    for t in tasks:
        if t.observed_at is not None and tick >= t.observed_at:
            k8s += per_task_cost_k8s(t)
        else:
            legacy += per_task_cost_legacy(t)
    return legacy, k8s, legacy + k8s

def build_cost_series(tasks: List[Task], max_tick: int) -> pd.DataFrame:
    data = []
    for x in range(0, max_tick + 1):
        lg, k8, tot = cost_at_tick(tasks, x)
        data.append({"tick": x, "legacy": lg, "k8s": k8, "total": tot})
    return pd.DataFrame(data)

def plot_cost_reduction(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(df["tick"], df["total"], marker="o", label="Total cost")
    ax.plot(df["tick"], df["legacy"], linestyle="--", label="Legacy portion")
    ax.plot(df["tick"], df["k8s"], linestyle=":", label="K8s portion")
    ax.set_xlabel("Tick (simulation time)")
    ax.set_ylabel("Cost (relative units)")
    ax.set_title("Company cost reduction as services migrate to Kubernetes")
    ax.legend()
    st.pyplot(fig, use_container_width=True)


# =========================
# Streamlit UI (auto-play loop)
# =========================
def center_title(text: str):
    st.markdown(
        f"""
        <div style="text-align:center; font-size: 28px; font-weight: 700; padding: 6px 0;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def sidebar_params() -> tuple[int, int, bool]:
    st.sidebar.header("Simulation")
    seed = st.sidebar.number_input("Seed RNG", min_value=0, max_value=10_000_000, value=42, step=1)
    num_tasks = st.sidebar.slider("Number of services/tasks", 5, 100, 30,
                                  help="Symuluj nawet 100 usług/serwerów.")
    run_btn = st.sidebar.button("Run / Reset", type="primary")
    return num_tasks, seed, run_btn

def timeline_table(tasks: List[Task]):
    rows = []
    for t in tasks:
        rows.append({
            "task_id": t.task_id,
            "service": t.service_name,
            "type": t.type,
            "complexity": t.complexity,
            "criticality": t.criticality,
            "discovered": t.discovered_at,
            "planned": t.planned_at,
            "migrated": t.migrated_at,
            "observed": t.observed_at,
        })
    st.dataframe(pd.DataFrame(rows).sort_values("task_id"), use_container_width=True)

def render_logs(logs: List[Tuple[int, str, str]]):
    st.write("### Simulation logs")
    st.code("\n".join(f"[{tick:04d}] {agent:12s} | {msg}" for tick, agent, msg in logs), language="text")

def init_sim(num_tasks: int, seed: int):
    rng = np.random.default_rng(seed)
    tasks = generate_tasks(num_tasks, rng)
    ctx = Context(rng=rng, tick=0)
    orch = Orchestrator(ctx)
    return tasks, ctx, orch

def step_and_record_costs(tasks: List[Task], orch: Orchestrator, cost_hist: list):
    """
    Advance one tick and append cost snapshot BEFORE increment (so t aligns with screen).
    """
    t = orch.ctx.tick
    lg, k8, tot = cost_at_tick(tasks, t)
    cost_hist.append((t, lg, k8, tot))
    done = orch.step(tasks)
    return done

def main():
    st.set_page_config(page_title="Agent Simulation — Ships & Costs", layout="wide")
    center_title("Agent-based migration with Kube & Krateo — Ships & Cost Reduction")

    num_tasks, seed, run_btn = sidebar_params()

    # --- Session state wiring ---
    if run_btn or "sim_started" not in st.session_state:
        # (Re)start sim
        tasks, ctx, orch = init_sim(num_tasks, seed)
        st.session_state.sim_started = True
        st.session_state.tasks = tasks
        st.session_state.ctx = ctx
        st.session_state.orch = orch
        st.session_state.cost_hist = []  # list of tuples (tick, legacy, k8s, total)
        st.session_state.done = False

    tasks: List[Task] = st.session_state.tasks
    ctx: Context = st.session_state.ctx
    orch: Orchestrator = st.session_state.orch
    cost_hist: list = st.session_state.cost_hist

    # --- Auto-play: advance one tick per refresh until done ---
    if not st.session_state.done:
        st.session_state.done = step_and_record_costs(tasks, orch, cost_hist)

    # Trigger auto-refresh while not done
    if not st.session_state.done:
        # ~6 FPS is overkill; 500-700ms works well for readability
        st.experimental_set_query_params(tick=str(ctx.tick))
        st.autorefresh = st.experimental_rerun  # minimal alias to avoid lint warnings
        # NOTE: Streamlit's st_autorefresh is convenient but not always available;
        # experimental_rerun suffices as we change the URL query to bust cache.
        # We only set params; rerun occurs via user agent on each run.

    # --- Layout ---
    c_top = st.container()
    with c_top:
        st.subheader(f"NetLogo-like center — tick: {ctx.tick} {'(done)' if st.session_state.done else ''}")
        fig = ships_frame(tasks, ctx.tick)
        st.pyplot(fig, use_container_width=True)

    st.divider()

    # Cost reduction panel
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.subheader("Cost reduction over time (as services move to Kubernetes)")
        if len(cost_hist) > 0:
            df = pd.DataFrame(cost_hist, columns=["tick", "legacy", "k8s", "total"])
            plot_cost_reduction(df)
        else:
            st.info("Costs will appear once the simulation starts ticking.")

        st.subheader("Timeline")
        timeline_table(tasks)

    with c2:
        st.subheader("Summary")
        done_cnt = sum(t.observed_at is not None and ctx.tick >= t.observed_at for t in tasks)
        st.metric("Services migrated (observed)", f"{done_cnt}/{len(tasks)}")
        st.metric("Current tick", ctx.tick)
        if len(cost_hist) > 0:
            cur_total = cost_hist[-1][3]
            st.metric("Current total cost (rel.)", f"{cur_total:,.0f}")
            if len(cost_hist) > 1:
                delta = cost_hist[-2][3] - cur_total
                st.caption(f"Δ since last tick: {delta:+.0f}")

    st.divider()
    render_logs(ctx.logs)

if __name__ == "__main__":
    main()

