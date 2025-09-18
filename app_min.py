# app_minimal.py — Streamlit Agent-Only Simulation (Kube & Krateo)
"""
Run:
  pip install streamlit numpy pandas matplotlib
  streamlit run app_minimal.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ---------- Data model ----------
@dataclass
class Task:
    task_id: int
    service_name: str
    type: str              # "stateless" | "stateful" | "cron"
    complexity: int        # 1..5
    criticality: str       # "low" | "med" | "high"

    discovered_at: int | None = None
    planned_at: int | None = None
    migrated_at: int | None = None
    observed_at: int | None = None  # final stage in this minimal app


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
    num_tasks = st.sidebar.slider("Number of tasks", 5, 30, 12)
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
    st.pyplot(fig)

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

def main():
    st.set_page_config(page_title="Agent Simulation (Kube & Krateo)", layout="wide")
    center_title("Agent based simulation with Kube & Krateo")

    num_tasks, seed, run_btn = sidebar_params()

    if "_ran" not in st.session_state:
        st.session_state["_ran"] = False
    if run_btn:
        st.session_state["_ran"] = True

    if not st.session_state["_ran"]:
        st.info("Set the number of tasks on the left and press **Run / Reset**.")
        return

    logs, tasks, ticks = simulate(num_tasks=num_tasks, seed=seed)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Progress")
        plot_progress(tasks)
        st.subheader("Timeline")
        timeline_table(tasks)
    with c2:
        st.subheader("Summary")
        st.metric("Tasks completed", f"{sum(t.observed_at is not None for t in tasks)}/{len(tasks)}")
        st.metric("Ticks (sim time)", ticks)

    st.divider()
    render_logs(logs)

if __name__ == "__main__":
    main()
