# app_min.py — Tick-by-tick, optionally sequential, agent-based sim (Kube & Krateo)
"""
Run:
  pip install streamlit numpy pandas
  streamlit run app_min.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import time
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Data model ----------
@dataclass
class Task:
    task_id: int
    service_name: str
    type: str              # "stateless" | "stateful" | "cron"
    complexity: int        # 1..5
    criticality: str       # "low" | "med" | "high"

    # Finish times for each stage (when the stage is completed)
    discovered_at: Optional[int] = None
    planned_at: Optional[int] = None
    migrated_at: Optional[int] = None
    observed_at: Optional[int] = None   # final stage (done when tick >= observed_at)


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


def priority(task: Task) -> int:
    # higher priority for higher criticality; then complexity
    crit = {"low": 0, "med": 1, "high": 2}[task.criticality]
    return -(crit * 10 + task.complexity)


# ---------- Agents ----------
class BaseAgent:
    name = "Agent"

    def step(self, task: Task, ctx: Context) -> bool:
        """Try to move task by scheduling the finish time of the current stage.
        Returns True if task advanced (scheduled a stage), else False."""
        raise NotImplementedError

    def duration(self, task: Task, ctx: Context, base: int) -> int:
        """Synthetic duration in ticks; depends on complexity/type + random blocker."""
        mult = 1
        if task.type == "stateful":
            mult += 1
        elif task.type == "cron":
            mult += 0  # usually quick
        mult += max(0, task.complexity - 1) // 2
        if ctx.rng.random() < 0.08:
            mult += 1
            ctx.log(self.name, f"Task {task.task_id}: blocker encountered")
        return max(1, base * mult)


class DiscoveryAgent(BaseAgent):
    name = "Discovery"
    def step(self, task: Task, ctx: Context) -> bool:
        if task.discovered_at is None:
            task.discovered_at = ctx.tick + self.duration(task, ctx, base=1)
            ctx.log(self.name, f"Discovered dependencies scheduled to finish at t={task.discovered_at} for {task.service_name}")
            return True
        return False


class PlanningAgent(BaseAgent):
    name = "Planning"
    def step(self, task: Task, ctx: Context) -> bool:
        if task.discovered_at is not None and ctx.tick >= task.discovered_at and task.planned_at is None:
            task.planned_at = ctx.tick + self.duration(task, ctx, base=1)
            ctx.log(self.name, f"Planning scheduled to finish at t={task.planned_at} for {task.service_name}")
            return True
        return False


class MigrationAgent(BaseAgent):
    name = "Migration"
    def step(self, task: Task, ctx: Context) -> bool:
        if task.planned_at is not None and ctx.tick >= task.planned_at and task.migrated_at is None:
            task.migrated_at = ctx.tick + self.duration(task, ctx, base=2)
            ctx.log(self.name, f"Migration scheduled to finish at t={task.migrated_at} for {task.service_name}")
            return True
        return False


class ObservabilityAgent(BaseAgent):
    name = "Observability"
    def step(self, task: Task, ctx: Context) -> bool:
        if task.migrated_at is not None and ctx.tick >= task.migrated_at and task.observed_at is None:
            task.observed_at = ctx.tick + self.duration(task, ctx, base=1)
            ctx.log(self.name, f"Observability scheduled to finish at t={task.observed_at} for {task.service_name}")
            return True
        return False


AGENTS = [DiscoveryAgent(), PlanningAgent(), MigrationAgent(), ObservabilityAgent()]


# ---------- Status helpers for UI ----------
def status_of(task: Task, now: int) -> str:
    if task.observed_at is not None and now >= task.observed_at:
        return "done"
    if task.migrated_at is not None and now < task.migrated_at:
        return "migrating…"
    if task.planned_at is not None and now < task.planned_at:
        return "planning…"
    if task.discovered_at is not None and now < task.discovered_at:
        return "discovering…"
    if task.migrated_at is not None and now >= task.migrated_at and task.observed_at is None:
        return "ready for observability"
    if task.planned_at is not None and now >= task.planned_at and task.migrated_at is None:
        return "ready for migration"
    if task.discovered_at is not None and now >= task.discovered_at and task.planned_at is None:
        return "ready for planning"
    return "queued"


def next_suggested_task(tasks: List[Task], now: int) -> Optional[Task]:
    """Pick the highest-priority task that can advance at this tick."""
    for t in sorted(tasks, key=priority):
        # Is there a stage we can schedule RIGHT NOW?
        if t.observed_at is None:
            # Try in order: discover -> plan -> migrate -> observe
            if t.discovered_at is None:
                return t
            if t.planned_at is None and now >= t.discovered_at:
                return t
            if t.migrated_at is None and t.planned_at is not None and now >= t.planned_at:
                return t
            if t.observed_at is None and t.migrated_at is not None and now >= t.migrated_at:
                return t
    return None


# ---------- Orchestrator (tick-by-tick) ----------
def tick_once(tasks: List[Task], ctx: Context, sequential: bool) -> None:
    """Advance simulation by exactly one tick."""
    moved_any = False
    if sequential:
        # Only move ONE task per tick (strictly sequential suggestion)
        t = next_suggested_task(tasks, ctx.tick)
        if t is not None:
            for agent in AGENTS:
                if agent.step(t, ctx):
                    moved_any = True
                    break
    else:
        # Parallel-friendly: each task can move by at most one stage per tick
        for t in sorted(tasks, key=priority):
            before = (t.discovered_at, t.planned_at, t.migrated_at, t.observed_at)
            for agent in AGENTS:
                if agent.step(t, ctx):
                    moved_any = True
                    break
            after = (t.discovered_at, t.planned_at, t.migrated_at, t.observed_at)
            if before != after:
                continue

    # Move time forward by one tick
    ctx.tick += 1
    if not moved_any:
        ctx.log("Scheduler", "No stage scheduled this tick (waiting for earlier steps to complete).")


def all_done(tasks: List[Task], now: int) -> bool:
    return all(t.observed_at is not None and now >= t.observed_at for t in tasks)


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

def sidebar_params() -> tuple[int, int, bool, int]:
    st.sidebar.header("Simulation")
    seed = st.sidebar.number_input("Seed RNG", min_value=0, max_value=10_000_000, value=42, step=1)
    num_tasks = st.sidebar.slider("Number of tasks", 3, 30, 10)
    sequential = st.sidebar.toggle("Sequential mode (one service at a time)", value=True)
    delay_ms = st.sidebar.slider("Auto-play delay per tick (ms)", 50, 2000, 400, step=50)
    return num_tasks, seed, sequential, delay_ms

def ensure_state(num_tasks: int, seed: int, sequential: bool):
    if "sim" not in st.session_state or st.session_state.sim.get("needs_reset"):
        rng = np.random.default_rng(seed)
        tasks = generate_tasks(num_tasks, rng)
        ctx = Context(rng=rng, tick=0)
        st.session_state.sim = {
            "tasks": tasks,
            "ctx": ctx,
            "sequential": sequential,
            "autoplay": False,
            "needs_reset": False,
        }
    else:
        # If user toggled sequential or changed counts, offer reset hint
        st.session_state.sim["sequential"] = sequential

def reset_sim(num_tasks: int, seed: int, sequential: bool):
    st.session_state.sim = {"needs_reset": True}
    ensure_state(num_tasks, seed, sequential)

def progress_chart(tasks: List[Task], now: int):
    max_tick = max([t.observed_at or 0 for t in tasks] + [now])
    xs = list(range(0, max_tick + 1))
    done_per_tick = [sum(1 for t in tasks if t.observed_at is not None and t.observed_at <= x) for x in xs]
    df = pd.DataFrame({"tick": xs, "tasks_completed": done_per_tick}).set_index("tick")
    st.line_chart(df)

def timeline_table(tasks: List[Task], now: int):
    rows = []
    for t in tasks:
        rows.append({
            "task_id": t.task_id,
            "service": t.service_name,
            "type": t.type,
            "complexity": t.complexity,
            "criticality": t.criticality,
            "status@now": status_of(t, now),
            "discovered_at": t.discovered_at,
            "planned_at": t.planned_at,
            "migrated_at": t.migrated_at,
            "observed_at": t.observed_at,
        })
    st.dataframe(pd.DataFrame(rows).sort_values(["status@now", "task_id"]), use_container_width=True)

def render_logs(logs: List[Tuple[int, str, str]]):
    st.write("### Simulation logs")
    st.code("\n".join(f"[{tick:04d}] {agent:12s} | {msg}" for tick, agent, msg in logs), language="text")

def main():
    st.set_page_config(page_title="Agent Simulation (Kube & Krateo) — tick-by-tick", layout="wide")
    center_title("Agent based simulation with Kube & Krateo")

    num_tasks, seed, sequential, delay_ms = sidebar_params()

    # Initialize / reset
    ensure_state(num_tasks, seed, sequential)
    sim = st.session_state.sim
    tasks: List[Task] = sim["tasks"]
    ctx: Context = sim["ctx"]

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Reset", type="secondary", use_container_width=True):
        reset_sim(num_tasks, seed, sequential)
        st.experimental_rerun()

    if c2.button("Next tick", type="primary", use_container_width=True):
        tick_once(tasks, ctx, sequential=sim["sequential"])

    autoplay_label = "Stop auto-play" if sim["autoplay"] else "Start auto-play"
    if c3.button(autoplay_label, use_container_width=True):
        sim["autoplay"] = not sim["autoplay"]

    c4.metric("Current tick", ctx.tick)

    # Suggested next service (sequential hint)
    suggested = next_suggested_task(tasks, ctx.tick)
    if suggested:
        st.info(f"Suggested next service: **{suggested.service_name}** (priority first).")
    else:
        st.info("No immediate next step available — waiting for scheduled finishes.")

    # Charts & tables
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Progress")
        progress_chart(tasks, now=ctx.tick)
        st.subheader("Timeline (finish ticks & current status)")
        timeline_table(tasks, now=ctx.tick)
    with right:
        st.subheader("Summary")
        st.metric("Tasks done", f"{sum(t.observed_at is not None and ctx.tick >= t.observed_at for t in tasks)}/{len(tasks)}")
        st.metric("Mode", "Sequential" if sim["sequential"] else "Parallel-friendly")

    st.divider()
    render_logs(ctx.logs)

    # Auto-play loop: advance 1 tick per rerun
    if sim["autoplay"] and not all_done(tasks, ctx.tick):
        tick_once(tasks, ctx, sequential=sim["sequential"])
        time.sleep(max(0.01, delay_ms / 1000.0))
        st.experimental_rerun()

if __name__ == "__main__":
    main()
