"""
QJIC Reflex Cell Network – 5F Department Store Domino Simulation

추가 반영 사항
- 침입자 메인 시나리오: 1층 진입 -> 5층까지 이동 -> 다시 1층으로 도주
- 마우스 입력: 클릭/드래그로 외부 에너지 자극(센서 입력) 주입
- 50개 모든 셀은 독립 유기체: 수신 재전파 금지, 자기 임계 발화시에만 이웃 전달
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ==============================
# Global parameters
# ==============================
FLOORS = 5
ZONES_PER_FLOOR = 10
TOTAL_CELLS = FLOORS * ZONES_PER_FLOOR

DT = 0.10
SENSOR_EPS = 0.2
SENSOR_RANGE = 3.4

# local sensing
LOCAL_A = 0.09
LOCAL_B = 0.55

# manual(mouse) sensing
MANUAL_RADIUS = 1.6
MANUAL_GAIN = 1.35

# cell dynamics
A_DECAY = 0.91
K_FEEDBACK = 0.035
LEAK = 0.028
VTH_ON = 1.20
VTH_OFF = 0.58
REFRACTORY_TICKS = 5
ENERGY_TRANSFER_RATIO = 0.382

# directional blend
DIR_BLEND_LOCAL = 0.55
DIR_BLEND_NET = 0.25
DIR_BLEND_PREV = 0.20

# visualization
CMAP = "turbo"
VMIN, VMAX = 0.0, 2.2


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def blend_dir(local_dir: np.ndarray, net_dir: np.ndarray, prev_dir: np.ndarray) -> np.ndarray:
    vec = (DIR_BLEND_LOCAL * local_dir) + (DIR_BLEND_NET * net_dir) + (DIR_BLEND_PREV * prev_dir)
    return normalize(vec)


@dataclass
class ReflexCell:
    """독립 유기체 셀.

    - 받은 에너지는 상태 계산용 입력이다.
    - 받은 패킷은 그대로 재전파하지 않는다.
    - 오직 자기 fire() 시에만 이웃에게 전달한다.
    """

    cid: int
    floor: int
    zone: int
    pos: Tuple[float, float]

    Vs: float = 0.0
    Vdir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))

    is_firing: bool = False
    refractory_left: int = 0

    received_net_input: float = 0.0
    received_net_dir_vector: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    fired_this_tick: bool = False
    transfer_energy_this_tick: float = 0.0

    def receive_energy(self, e: float, direction: np.ndarray) -> None:
        if e <= 0:
            return
        self.received_net_input += e
        self.received_net_dir_vector += e * direction

    def fire(self) -> float:
        self.fired_this_tick = True
        self.is_firing = True
        self.refractory_left = REFRACTORY_TICKS

        released = max(0.0, self.Vs * ENERGY_TRANSFER_RATIO)
        self.Vs = self.Vs * (1.0 - ENERGY_TRANSFER_RATIO) * 0.40
        self.transfer_energy_this_tick = released
        return released

    def update(self, local_input: float, local_dir: np.ndarray) -> float:
        self.fired_this_tick = False
        self.transfer_energy_this_tick = 0.0

        net_input = self.received_net_input
        net_dir = normalize(self.received_net_dir_vector)

        self.Vdir = blend_dir(local_dir, net_dir, self.Vdir)
        self.Vs = (A_DECAY * self.Vs) + local_input + (K_FEEDBACK * self.Vs) + net_input - LEAK
        self.Vs = max(0.0, self.Vs)

        if self.refractory_left > 0:
            self.refractory_left -= 1

        if self.is_firing:
            if self.Vs <= VTH_OFF:
                self.is_firing = False
        else:
            if self.Vs >= VTH_ON and self.refractory_left == 0:
                self.fire()

        self.received_net_input = 0.0
        self.received_net_dir_vector = np.zeros(2, dtype=float)
        return self.transfer_energy_this_tick


@dataclass
class Intruder:
    iid: int
    path: List[Tuple[int, float, float]]
    speed: float = 1.05
    random_walk: bool = False

    floor: int = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    segment_idx: int = 0
    active: bool = True
    prev_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    def __post_init__(self) -> None:
        f, x, y = self.path[0]
        self.floor = f
        self.position = np.array([x, y], dtype=float)
        self.prev_position = self.position.copy()

    def _random_step(self, dt: float) -> None:
        theta = random.uniform(0, 2 * math.pi)
        step = self.speed * dt * random.uniform(0.4, 1.4)
        self.position += np.array([math.cos(theta), math.sin(theta)]) * step
        self.position[0] = float(np.clip(self.position[0], 0.2, 9.8))
        self.position[1] = float(np.clip(self.position[1], -1.6, 1.6))
        if random.random() < 0.015:
            self.floor = int(np.clip(self.floor + random.choice([-1, 1]), 0, FLOORS - 1))

    def update(self, dt: float) -> None:
        if not self.active:
            return

        self.prev_position = self.position.copy()

        if self.random_walk:
            self._random_step(dt)
            return

        if self.segment_idx >= len(self.path) - 1:
            self.active = False
            return

        sf = self.floor
        sx, sy = self.position
        tf, tx, ty = self.path[self.segment_idx + 1]

        if sf != tf:
            self.floor = sf + (1 if tf > sf else -1)
            if self.floor == tf:
                self.position = np.array([tx, ty], dtype=float)
                self.segment_idx += 1
            return

        dxy = np.array([tx - sx, ty - sy], dtype=float)
        dist = float(np.linalg.norm(dxy))
        if dist < 1e-8:
            self.segment_idx += 1
            return

        step = self.speed * dt
        if step >= dist:
            self.position = np.array([tx, ty], dtype=float)
            self.segment_idx += 1
        else:
            self.position += dxy / dist * step


class HouseSimulation:
    def __init__(self, intruder_count: int = 2, random_walk: bool = False) -> None:
        self.cells = self._build_cells()
        self.neighbors = self._build_topology()
        self.intruders = self._build_intruders(intruder_count=intruder_count, random_walk=random_walk)

        self.manual_energy = np.zeros(TOTAL_CELLS, dtype=float)
        self.manual_dir = [np.zeros(2, dtype=float) for _ in range(TOTAL_CELLS)]

        self.fig, self.axes = plt.subplots(FLOORS, 1, figsize=(12, 11), sharex=True)
        self.fig.suptitle("QJIC Reflex Cells in 5F Department Store – Autonomous Organic Domino Wave", fontsize=13)

        self.cell_scatters = []
        self.dir_quivers = []
        self.intruder_scatters = []
        self.status_text = None
        self.frame_count = 0
        self.mouse_down = False

        self._setup_scene()
        self._connect_mouse_events()

    def _build_cells(self) -> List[ReflexCell]:
        cells: List[ReflexCell] = []
        for f in range(FLOORS):
            for z in range(ZONES_PER_FLOOR):
                x = float(z + 0.5)
                y = float(0.8 if (z % 2 == 0) else -0.8)
                cid = f * ZONES_PER_FLOOR + z
                cells.append(ReflexCell(cid=cid, floor=f, zone=z, pos=(x, y)))
        return cells

    def _build_topology(self) -> Dict[int, List[int]]:
        graph: Dict[int, List[int]] = {c.cid: [] for c in self.cells}

        def idx(floor: int, zone: int) -> int:
            return floor * ZONES_PER_FLOOR + zone

        for f in range(FLOORS):
            for z in range(ZONES_PER_FLOOR):
                me = idx(f, z)
                for nz in [z - 1, z + 1, z - 2, z + 2]:
                    if 0 <= nz < ZONES_PER_FLOOR:
                        graph[me].append(idx(f, nz))

        for f in range(FLOORS - 1):
            for z in [4, 8]:  # escalator/stairs
                a, b = idx(f, z), idx(f + 1, z)
                graph[a].append(b)
                graph[b].append(a)

        for cid in graph:
            graph[cid] = sorted(set(graph[cid]))
        return graph

    def _scenario_path_main(self) -> List[Tuple[int, float, float]]:
        """1층 진입 -> 5층까지 이동 -> 다시 1층으로 도주(요구 반영)."""
        return [
            # ascend
            (0, 0.2, -1.2), (0, 3.0, 0.9), (0, 4.5, 0.0),
            (1, 4.5, 0.0), (1, 7.2, -0.8), (1, 8.5, 0.0),
            (2, 8.5, 0.0), (2, 5.8, 0.9), (2, 4.5, 0.0),
            (3, 4.5, 0.0), (3, 8.2, -0.9), (3, 8.5, 0.0),
            (4, 8.5, 0.0), (4, 4.8, 0.8), (4, 1.3, -0.9),
            # escape down to floor 1
            (4, 4.5, 0.0), (3, 4.5, 0.0), (3, 8.5, 0.0),
            (2, 8.5, 0.0), (2, 4.5, 0.0),
            (1, 4.5, 0.0), (1, 1.0, -1.1),
            (0, 1.0, -1.2), (0, 0.1, -1.3),
        ]

    def _scenario_path_secondary(self) -> List[Tuple[int, float, float]]:
        return [
            (0, 0.3, 1.2), (0, 2.5, -0.8), (0, 4.5, 0.0),
            (1, 4.5, 0.0), (1, 3.0, 1.0), (1, 4.5, 0.0),
            (2, 4.5, 0.0), (2, 6.8, -0.8), (2, 8.5, 0.0),
            (3, 8.5, 0.0), (3, 5.2, 0.8), (4, 4.5, 0.0),
            (4, 6.0, -1.0), (3, 4.5, 0.0), (2, 4.5, 0.0),
            (1, 4.5, 0.0), (0, 2.0, -1.0),
        ]

    def _build_intruders(self, intruder_count: int, random_walk: bool) -> List[Intruder]:
        intruders = [Intruder(iid=0, path=self._scenario_path_main(), speed=1.10, random_walk=random_walk)]
        if intruder_count >= 2:
            intruders.append(Intruder(iid=1, path=self._scenario_path_secondary(), speed=0.96, random_walk=random_walk))
        return intruders

    def _setup_scene(self) -> None:
        for floor_idx, ax in enumerate(self.axes):
            ax.set_xlim(0.0, 10.0)
            ax.set_ylim(-1.8, 1.8)
            ax.grid(alpha=0.20)
            ax.set_ylabel(f"F{floor_idx + 1}")

            floor_cells = [c for c in self.cells if c.floor == floor_idx]
            fx = np.array([c.pos[0] for c in floor_cells])
            fy = np.array([c.pos[1] for c in floor_cells])
            fz = np.array([c.Vs for c in floor_cells])
            fdx = np.array([c.Vdir[0] for c in floor_cells])
            fdy = np.array([c.Vdir[1] for c in floor_cells])

            scatter = ax.scatter(fx, fy, c=fz, cmap=CMAP, vmin=VMIN, vmax=VMAX, s=300, edgecolors="black")
            quiver = ax.quiver(fx, fy, fdx, fdy, angles="xy", scale_units="xy", scale=3.5, width=0.006, color="black", alpha=0.75)

            for c in floor_cells:
                for nb in self.neighbors[c.cid]:
                    ncell = self.cells[nb]
                    if ncell.floor != floor_idx or nb < c.cid:
                        continue
                    ax.plot([c.pos[0], ncell.pos[0]], [c.pos[1], ncell.pos[1]], color="gray", alpha=0.25, linewidth=1.0)

            self.cell_scatters.append(scatter)
            self.dir_quivers.append(quiver)
            self.intruder_scatters.append(ax.scatter([], [], c="black", marker="X", s=120))

        self.axes[-1].set_xlabel("Zone axis (corridor)")
        cbar = self.fig.colorbar(self.cell_scatters[0], ax=self.axes, fraction=0.016, pad=0.01)
        cbar.set_label("Vs energy")
        self.status_text = self.fig.text(0.01, 0.985, "", va="top", fontsize=10)

    def _connect_mouse_events(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _apply_manual_stimulus(self, floor: int, x: float, y: float, strength: float = 1.0) -> None:
        center = np.array([x, y], dtype=float)
        for cell in self.cells:
            if cell.floor != floor:
                continue
            cpos = np.array(cell.pos, dtype=float)
            rel = cpos - center
            d = float(np.linalg.norm(rel))
            if d > MANUAL_RADIUS:
                continue
            falloff = max(0.0, 1.0 - d / MANUAL_RADIUS)
            e = MANUAL_GAIN * strength * falloff
            self.manual_energy[cell.cid] += e
            self.manual_dir[cell.cid] += normalize(rel) * e

    def _axis_to_floor(self, event_ax) -> int | None:
        for idx, ax in enumerate(self.axes):
            if ax == event_ax:
                return idx
        return None

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        floor = self._axis_to_floor(event.inaxes)
        if floor is None:
            return
        self.mouse_down = True
        self._apply_manual_stimulus(floor, float(event.xdata), float(event.ydata), strength=1.2)

    def _on_mouse_release(self, event) -> None:
        self.mouse_down = False

    def _on_mouse_move(self, event) -> None:
        if not self.mouse_down:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        floor = self._axis_to_floor(event.inaxes)
        if floor is None:
            return
        self._apply_manual_stimulus(floor, float(event.xdata), float(event.ydata), strength=0.45)

    def _compute_local_inputs(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        local_inputs = np.zeros(TOTAL_CELLS, dtype=float)
        local_dirs: List[np.ndarray] = [np.zeros(2, dtype=float) for _ in range(TOTAL_CELLS)]

        for cell in self.cells:
            risk_dir_sum = np.zeros(2, dtype=float)

            # intruder sensor input
            for intr in self.intruders:
                if not intr.active or intr.floor != cell.floor:
                    continue

                cpos = np.array(cell.pos, dtype=float)
                rel = intr.position - cpos
                d = float(np.linalg.norm(rel))
                if d > SENSOR_RANGE:
                    continue

                d_safe = max(d, SENSOR_EPS)
                intr_vel = (intr.position - intr.prev_position) / DT
                toward_cell = -rel / d if d > 1e-8 else np.zeros(2, dtype=float)
                approach_speed = max(0.0, float(np.dot(intr_vel, toward_cell)))

                sensor_input = (LOCAL_A * (1.0 / d_safe)) + (LOCAL_B * approach_speed)
                local_inputs[cell.cid] += sensor_input
                risk_dir_sum += sensor_input * normalize(rel)

            # manual(mouse) sensor input
            if self.manual_energy[cell.cid] > 0:
                local_inputs[cell.cid] += self.manual_energy[cell.cid]
                risk_dir_sum += self.manual_dir[cell.cid]

            local_dirs[cell.cid] = normalize(risk_dir_sum)

        # one-tick manual impulse consume
        self.manual_energy[:] = 0.0
        self.manual_dir = [np.zeros(2, dtype=float) for _ in range(TOTAL_CELLS)]

        return local_inputs, local_dirs

    def update(self, frame: int):
        self.frame_count += 1

        for intr in self.intruders:
            intr.update(DT)

        local_inputs, local_dirs = self._compute_local_inputs()

        releases = np.zeros(TOTAL_CELLS, dtype=float)
        fired_cells = 0
        for cell in self.cells:
            released = cell.update(local_inputs[cell.cid], local_dirs[cell.cid])
            releases[cell.cid] = released
            if cell.fired_this_tick:
                fired_cells += 1

        # 핵심: 자기 발화로 생성된 에너지만 이웃 전달
        for cid, released in enumerate(releases):
            if released <= 0:
                continue
            src = self.cells[cid]
            nbrs = self.neighbors[cid]
            if not nbrs:
                continue
            share = released / len(nbrs)
            src_dir = normalize(src.Vdir)
            for nb in nbrs:
                self.cells[nb].receive_energy(share, src_dir)

        for f, ax in enumerate(self.axes):
            floor_cells = [c for c in self.cells if c.floor == f]
            vals = np.array([c.Vs for c in floor_cells])
            arr_xy = np.array([c.pos for c in floor_cells])
            arr_uv = np.array([c.Vdir for c in floor_cells])

            self.cell_scatters[f].set_array(vals)
            self.dir_quivers[f].set_offsets(arr_xy)
            self.dir_quivers[f].set_UVC(arr_uv[:, 0], arr_uv[:, 1])

            intr_points = []
            for intr in self.intruders:
                if intr.active and intr.floor == f:
                    intr_points.append([intr.position[0], intr.position[1]])
            pts = np.array(intr_points) if intr_points else np.empty((0, 2))
            self.intruder_scatters[f].set_offsets(pts)

        active_intruders = sum(1 for intr in self.intruders if intr.active)
        mean_vs = float(np.mean([c.Vs for c in self.cells]))
        self.status_text.set_text(
            f"t={self.frame_count*DT:5.2f}s | active_intruders={active_intruders} | fired={fired_cells} | mean_Vs={mean_vs:.3f} | mouse={'on' if self.mouse_down else 'off'}"
        )

        artists = []
        artists.extend(self.cell_scatters)
        artists.extend(self.dir_quivers)
        artists.extend(self.intruder_scatters)
        artists.append(self.status_text)
        return artists

    def run(self) -> None:
        self.anim = FuncAnimation(self.fig, self.update, interval=60, blit=False)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()


def run_headless_steps(steps: int = 180, intruder_count: int = 2, random_walk: bool = False) -> None:
    sim = HouseSimulation(intruder_count=intruder_count, random_walk=random_walk)
    for i in range(steps):
        sim.update(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QJIC 5F Department Store Domino Simulation")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run N steps without GUI")
    parser.add_argument("--intruders", type=int, default=2, choices=[1, 2], help="Number of intruders")
    parser.add_argument("--random-walk", action="store_true", help="Use random walking instead of scripted paths")
    args = parser.parse_args()

    if args.headless_steps > 0:
        run_headless_steps(args.headless_steps, intruder_count=args.intruders, random_walk=args.random_walk)
    else:
        HouseSimulation(intruder_count=args.intruders, random_walk=args.random_walk).run()
