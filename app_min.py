# app_minimal.py — Streamlit Agent-Only Simulation (Kube & Krateo) + Port/Ships Viz
"""
Run:
  pip install streamlit numpy pandas matplotlib
  streamlit run app_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, RegularPolygon

plt.switch_backend("Agg")  # safer for Streamlit servers


# ---------- Data model ----------
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
    observed_at: Optional[int] = None  # final stage in this minimal app


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


# ---------- Helpers ----------
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


# ---------- Agents ----------
class BaseAgent:
    name = "Agent"

    def step(self, task: Task, ctx: Context) -> None:
        raise NotImplementedError

    def effort_delay(self, task: Task, ctx: Context, base: int) -> int:
        """Return a synthetic tick delay based on complexity/type (no sleeping; just modeling)."""
        mult = 1
        if task.type == "stateful":
            mult += 1
        elif task.type == "cron":
            mult += 0  # small tasks
        mult += max(0, task.complexity - 1) // 2
        # occasional extra delay (blocker)
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


# ---------- Orchestrator ----------
class Orchestrator:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.agents = [DiscoveryAgent(), PlanningAgent(), MigrationAgent(), ObservabilityAgent()]

    def priority(self, task: Task) -> int:
        crit = {"low": 0, "med": 1, "high": 2}[task.criticality]
        return -(crit * 10 + task.complexity)

    def tick_once(self, tasks: List[Task]) -> None:
        # Process tasks in priority order; at most one transition per tick per task.
        for task in sorted(tasks, key=self.priority):
            for agent in self.agents:
                before = (task.discovered_at, task.planned_at, task.migrated_at, task.observed_at)
                agent.step(task, self.ctx)
                after = (task.discovered_at, task.planned_at, task.migrated_at, task.observed_at)
                if before != after:
                    break  # moved one stage this tick; stop here

    def run(self, tasks: List[Task], max_ticks: int = 200) -> None:
        for t in range(max_ticks):
            self.ctx.tick = t
            if all(x.observed_at is not None and t >= x.observed_at for x in tasks):
                break
            self.tick_once(tasks)


# ---------- Simulation ----------
def simulate(num_tasks: int, seed: int) -> tuple[list[tuple[int, str, str]], list[Task], int]:
    rng = np.random.default_rng(seed)
    tasks = generate_tasks(num_tasks, rng)
    ctx = Context(rng=rng)
    Orchestrator(ctx).run(tasks, max_ticks=300)
    return ctx.logs, tasks, ctx.tick


# ---------- Visualization helpers (Ports & Ships) ----------
PORTS = {
    "backlog": (0.05, 0.5),
    "discovery": (0.25, 0.75),
    "planning": (0.45, 0.25),
    "migration": (0.65, 0.7),
    "observability": (0.88, 0.5),
}
LANES = [
    ("backlog", "discovery"),
    ("discovery", "planning"),
    ("planning", "migration"),
    ("migration", "observability"),
]

CRIT_COLOR = {"low": "#4caf50", "med": "#ffb300", "high": "#e53935"}
TYPE_HATCH = {"stateless": "", "stateful": "////", "cron": "..."}

def _interp(p1, p2, t: float) -> Tuple[float, float]:
    x = p1[0] + (p2[0] - p1[0]) * t
    y = p1[1] + (p2[1] - p1[1]) * t
    return x, y

def task_position_at_tick(task: Task, tick: int) -> Tuple[float, float]:
    """Linear interpolation along the pipeline between ports based on stage end times."""
    # Define time checkpoints (end of each stage)
    t0 = task.discovered_at
    t1 = task.planned_at
    t2 = task.migrated_at
    t3 = task.observed_at

    # segment 0: backlog -> discovery (0..t0)
    if t0 is None or tick <= t0:
        total = max(t0 or 1, 1)
        frac = np.clip((tick) / total, 0.0, 1.0)
        return _interp(PORTS["backlog"], PORTS["discovery"], frac)

    # segment 1: discovery -> planning (t0..t1)
    if t1 is None or tick <= t1:
        total = max((t1 or (t0+1)) - t0, 1)
        frac = np.clip((tick - t0) / total, 0.0, 1.0)
        return _interp(PORTS["discovery"], PORTS["planning"], frac)

    # segment 2: planning -> migration (t1..t2)
    if t2 is None or tick <= t2:
        total = max((t2 or (t1+1)) - t1, 1)
        frac = np.clip((tick - t1) / total, 0.0, 1.0)
        return _interp(PORTS["planning"], PORTS["migration"], frac)

    # segment 3: migration -> observability (t2..t3)
    if t3 is None or tick <= t3:
        total = max((t3 or (t2+1)) - t2, 1)
        frac = np.clip((tick - t2) / total, 0.0, 1.0)
        return _interp(PORTS["migration"], PORTS["observability"], frac)

    # done: parked in observability
    return PORTS["observability"]

def draw_ports_background(ax):
    # water background
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#e6f3ff", edgecolor="none", zorder=0))
    # lanes
    for a, b in LanesForRedundancy():
        p1, p2 = PORTS[a], PORTS[b]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-", linewidth=1.5, linestyle="--", alpha=0.35, zorder=1))
    # ports (departments)
    for name, (x, y) in PORTS.items():
        ax.add_patch(Circle((x, y), 0.035, facecolor="#cfd8dc", edgecolor="#455a64", linewidth=1.5, zorder=3))
        ax.add_patch(Rectangle((x-0.05, y-0.01), 0.1, 0.02, facecolor="#90a4ae", edgecolor="#37474f", zorder=2))
        label = name.capitalize()
        ax.text(x, y+0.07, label, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#263238", zorder=4)

def LanesForRedundancy():
    """Return duplicated lanes to hint redundancy/HA (active/active)."""
    lanes = []
    for a, b in LANES:
        lanes.append((a, b))
        # add slight offset dual path
        # (we'll draw both to visually hint redundancy)
    return lanes

def draw_ship(ax, x: float, y: float, task: Task, scale: float = 0.02):
    """Draw a tiny container ship representing a Task."""
    # hull
    w, h = 4*scale, 1.6*scale
    hull = Rectangle((x - w/2, y - h/2), w, h,
                     facecolor=CRIT_COLOR[task.criticality],
                     edgecolor="#263238", linewidth=0.6,
                     hatch=TYPE_HATCH[task.type], zorder=10, alpha=0.9)
    ax.add_patch(hull)
    # bow triangle
    bow = RegularPolygon((x + w/2, y), numVertices=3, radius=0.9*scale, orientation=0,
                         facecolor="#90a4ae", edgecolor="#263238", linewidth=0.5, zorder=11)
    ax.add_patch(bow)
    # containers: draw up to 3 small boxes based on complexity
    boxes = min(3, max(1, task.complexity // 2 + (1 if task.complexity >= 5 else 0)))
    for i in range(boxes):
        bx = x - w/2 + 0.2*scale + i*(0.9*scale)
        by = y + 0.2*scale
        ax.add_patch(Rectangle((bx, by), 0.8*scale, 0.5*scale,
                               facecolor="#fff", edgecolor="#37474f", linewidth=0.5, zorder=12))
    # service initials
    initials = "".join([tok[0] for tok in task.service_name.split("-")[:2]]).upper()
    ax.text(x, y - 0.015, initials, ha="center", va="top", fontsize=6, color="#0d1b2a", zorder=13)

def ships_frame(tasks: List[Task], tick: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_ports_background(ax)

    # spread ships slightly to avoid overlaps (lane jitter by task_id)
    rng = np.random.default_rng(12345)
    jitter_map = {}
    for t in tasks:
        jitter_map[t.task_id] = (rng.random() - 0.5) * 0.03

    for t in tasks:
        x, y = task_position_at_tick(t, tick)
        y += jitter_map[t.task_id]
        draw_ship(ax, x, y, t)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=10, markerfacecolor=CRIT_COLOR["low"], label="low"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=10, markerfacecolor=CRIT_COLOR["med"], label="med"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=10, markerfacecolor=CRIT_COLOR["high"], label="high"),
        Line2D([0], [0], color="#000", linestyle="--", label="redundant lane"),
    ]
    ax.legend(handles=handles, title="Criticality & Lanes", loc="lower left", frameon=True, fontsize=8)
    ax.set_title("Shipping lanes between departments (ports) — tasks as container ships", fontsize=12, pad=8)
    plt.tight_layout()
    return fig


# ---------- Streamlit UI ----------
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
    num_tasks = st.sidebar.slider("Number of tasks", 5, 100, 24,
                                  help="Możesz zasymulować nawet 100 'serwerów/usług'.")
    run_btn = st.sidebar.button("Run / Reset", type="primary")
    return num_tasks, seed, run_btn

def plot_progress(tasks: List[Task]):
    max_tick = 0
    if tasks:
        max_tick = max([(t.observed_at or 0) for t in tasks])
    xs = list(range(0, max_tick + 1))
    done_per_tick = []
    for x in xs:
        done = sum(1 for t in tasks if t.observed_at is not None and t.observed_at <= x)
        done_per_tick.append(done)
    fig, ax = plt.subplots()
    ax.plot(xs, done_per_tick, marker="o")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Tasks completed")
    ax.set_title("Task completion over time")
    st.pyplot(fig, use_container_width=True)

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

def netlogo_like_center(tasks: List[Task], max_tick: int):
    st.subheader("NetLogo-like viz (Ships & Ports) — centrum")
    tick = st.slider("Tick (czas symulacji)", 0, max_tick, min(10, max_tick), key="viz_tick")
    fig = ships_frame(tasks, tick)
    st.pyplot(fig, use_container_width=True)

def summary_cards(tasks: List[Task], ticks: int):
    st.subheader("Summary")
    st.metric("Tasks completed", f"{sum(t.observed_at is not None for t in tasks)}/{len(tasks)}")
    st.metric("Ticks (sim time)", ticks)
    high = sum(1 for t in tasks if t.criticality == "high")
    st.metric("High criticality", high)

def main():
    st.set_page_config(page_title="Agent Simulation (Kube & Krateo)", layout="wide")
    center_title("Agent-based migration with Kube & Krateo — Ships & Ports Viz")

    num_tasks, seed, run_btn = sidebar_params()

    if "_ran" not in st.session_state:
        st.session_state["_ran"] = False
    if run_btn:
        st.session_state["_ran"] = True

    if not st.session_state["_ran"]:
        st.info("Ustaw liczbę zadań po lewej i kliknij **Run / Reset**. Następnie użyj suwaka 'Tick' w centrum, by obejrzeć animację.")
        return

    logs, tasks, ticks = simulate(num_tasks=num_tasks, seed=seed)

    # --- CENTRAL: NetLogo-like visualization ---
    c_top = st.container()
    with c_top:
        netlogo_like_center(tasks, max_tick=ticks)

    st.divider()

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.subheader("Progress")
        plot_progress(tasks)
        st.subheader("Timeline")
        timeline_table(tasks)
    with c2:
        summary_cards(tasks, ticks)

    st.divider()
    render_logs(logs)

if __name__ == "__main__":
    main()

