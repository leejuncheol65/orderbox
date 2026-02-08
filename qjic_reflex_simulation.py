"""
QJIC Reflex Cell Network – 5F Department Store Domino Simulation

요구사항 반영 포인트
- 5층 x 10구역 = 총 50개 QJIC Reflex Cell
- 침입자 1~2명 이동(시나리오 + 랜덤 워킹 옵션)
- 로컬 감지 + 이웃 네트워크 입력 결합
- 히스테리시스(VTH_ON/VTH_OFF), refractory 구간
- "수신 재전파 금지": 받은 패킷은 입력으로만 반영, 재패킷화 금지
- 셀의 자율 판단: 중앙 제어 없이 각 셀이 독립 업데이트
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
# Global parameters (튜닝 포인트)
# ==============================
FLOORS = 5
ZONES_PER_FLOOR = 10
TOTAL_CELLS = FLOORS * ZONES_PER_FLOOR

DT = 0.10
SENSOR_EPS = 0.2
SENSOR_RANGE = 3.4

# local sensing
LOCAL_A = 0.09  # A*(1/d)
LOCAL_B = 0.55  # B*approach_speed

# cell dynamics
A_DECAY = 0.91
K_FEEDBACK = 0.035
LEAK = 0.028
VTH_ON = 1.20
VTH_OFF = 0.58
REFRACTORY_TICKS = 5

ENERGY_TRANSFER_RATIO = 0.382

# directional blending
DIR_BLEND_LOCAL = 0.55
DIR_BLEND_NET = 0.25
DIR_BLEND_PREV = 0.20

# visualization
CMAP = "turbo"
VMIN, VMAX = 0.0, 2.2


# ---------- utility ----------
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
    """QJIC Reflex Cell: 독립 판단 주체.

    핵심 원칙
    - 받은 에너지는 Vs 업데이트에만 기여한다.
    - 받은 패킷(received_net_input)을 그대로 재전파하지 않는다.
    - 오직 자기 상태로 fire()가 발생했을 때만 전달 에너지 패킷 생성.
    """

    cid: int
    floor: int
    zone: int
    pos: Tuple[float, float]

    Vs: float = 0.0
    Vdir: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))

    is_firing: bool = False
    refractory_left: int = 0

    # tick-local buffers
    received_net_input: float = 0.0
    received_net_dir_vector: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))

    fired_this_tick: bool = False
    transfer_energy_this_tick: float = 0.0

    def receive_energy(self, e: float, direction: np.ndarray) -> None:
        """네트워크 입력 수신(재전파 금지 대상).

        이 버퍼는 update()에서 Vs, Vdir 계산에만 반영된다.
        """
        if e <= 0:
            return
        self.received_net_input += e
        self.received_net_dir_vector += e * direction

    def fire(self) -> float:
        """자기 임계 초과로 발화 시에만 전달 패킷 생성."""
        self.fired_this_tick = True
        self.is_firing = True
        self.refractory_left = REFRACTORY_TICKS

        released = max(0.0, self.Vs * ENERGY_TRANSFER_RATIO)
        # 발화 후 잔류 에너지 유지 (완전 소실 X)
        self.Vs = self.Vs * (1.0 - ENERGY_TRANSFER_RATIO) * 0.40
        self.transfer_energy_this_tick = released
        return released

    def update(self, local_input: float, local_dir: np.ndarray) -> float:
        """단일 tick 업데이트.

        Vs = A_DECAY*Vs + local_input + K_FEEDBACK*Vs + net_input - LEAK
        히스테리시스: VTH_ON / VTH_OFF
        refractory_left > 0 동안 전파 억제
        """
        self.fired_this_tick = False
        self.transfer_energy_this_tick = 0.0

        net_input = self.received_net_input
        net_dir = normalize(self.received_net_dir_vector)

        # 방향 벡터 갱신
        self.Vdir = blend_dir(local_dir, net_dir, self.Vdir)

        # 에너지 상태 갱신
        self.Vs = (A_DECAY * self.Vs) + local_input + (K_FEEDBACK * self.Vs) + net_input - LEAK
        self.Vs = max(0.0, self.Vs)

        # refractory 진행
        if self.refractory_left > 0:
            self.refractory_left -= 1

        # 히스테리시스 상태전이
        if self.is_firing:
            if self.Vs <= VTH_OFF:
                self.is_firing = False
        else:
            if self.Vs >= VTH_ON:
                # refractory 기간에는 fire 패킷 억제
                if self.refractory_left == 0:
                    self.fire()

        # tick buffer clear
        self.received_net_input = 0.0
        self.received_net_dir_vector = np.zeros(2, dtype=float)
        return self.transfer_energy_this_tick


@dataclass
class Intruder:
    """침입자 모델: 시나리오 경로 + 랜덤 워크 옵션."""

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
        delta = np.array([math.cos(theta), math.sin(theta)]) * step
        self.position += delta

        # 층 내 이동 범위 제한 (복도/점포 평면)
        self.position[0] = float(np.clip(self.position[0], 0.2, 9.8))
        self.position[1] = float(np.clip(self.position[1], -1.6, 1.6))

        # 낮은 확률로 층 전환(계단/에스컬레이터 이용 가정)
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

        sf, sx, sy = self.floor, self.position[0], self.position[1]
        tf, tx, ty = self.path[self.segment_idx + 1]

        # 층이 다르면 먼저 층을 맞춘다(에스컬레이터/계단 도달 이벤트)
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
    """5층 백화점 QJIC 네트워크 시뮬레이터."""

    def __init__(self, intruder_count: int = 2, random_walk: bool = False) -> None:
        self.cells = self._build_cells()
        self.neighbors = self._build_topology()

        self.intruders = self._build_intruders(intruder_count=intruder_count, random_walk=random_walk)

        self.fig, self.axes = plt.subplots(FLOORS, 1, figsize=(12, 11), sharex=True)
        self.fig.suptitle("QJIC Reflex Cells in 5F Department Store – Domino Energy Wave", fontsize=13)

        self.cell_scatters = []
        self.dir_quivers = []
        self.intruder_scatters = []
        self.status_text = None
        self.frame_count = 0

        self._setup_scene()

    def _build_cells(self) -> List[ReflexCell]:
        cells: List[ReflexCell] = []
        for f in range(FLOORS):
            for z in range(ZONES_PER_FLOOR):
                # x: 복도 방향(0~9), y: 점포 위치 오프셋(지그재그)
                x = float(z + 0.5)
                y = float(0.8 if (z % 2 == 0) else -0.8)
                cid = f * ZONES_PER_FLOOR + z
                cells.append(ReflexCell(cid=cid, floor=f, zone=z, pos=(x, y)))
        return cells

    def _build_topology(self) -> Dict[int, List[int]]:
        """복도-점포 + 층간(에스컬레이터/계단) 링크 구성."""
        graph: Dict[int, List[int]] = {c.cid: [] for c in self.cells}

        def idx(floor: int, zone: int) -> int:
            return floor * ZONES_PER_FLOOR + zone

        for f in range(FLOORS):
            # same-floor corridor chain
            for z in range(ZONES_PER_FLOOR):
                me = idx(f, z)
                for nz in [z - 1, z + 1]:
                    if 0 <= nz < ZONES_PER_FLOOR:
                        graph[me].append(idx(f, nz))

                # nearby shop coupling (skip-edge style)
                for nz in [z - 2, z + 2]:
                    if 0 <= nz < ZONES_PER_FLOOR:
                        graph[me].append(idx(f, nz))

        # inter-floor links: escalator at zone=4, stairs at zone=8
        vertical_zones = [4, 8]
        for f in range(FLOORS - 1):
            for z in vertical_zones:
                a, b = idx(f, z), idx(f + 1, z)
                graph[a].append(b)
                graph[b].append(a)

        # unique + stable order
        for cid in graph:
            graph[cid] = sorted(set(graph[cid]))
        return graph

    def _scenario_path_main(self) -> List[Tuple[int, float, float]]:
        # 입구→점포→에스컬레이터→상층 반복
        return [
            (0, 0.2, -1.2), (0, 2.4, 0.9), (0, 4.4, -0.9),
            (0, 4.5, 0.0), (1, 4.5, 0.0),
            (1, 6.7, 0.9), (1, 8.1, -0.9),
            (1, 8.5, 0.0), (2, 8.5, 0.0),
            (2, 5.0, 0.8), (2, 4.5, 0.0), (3, 4.5, 0.0),
            (3, 7.4, -0.7), (3, 8.5, 0.0), (4, 8.5, 0.0),
            (4, 3.0, 0.8), (4, 1.2, -0.8),
        ]

    def _scenario_path_secondary(self) -> List[Tuple[int, float, float]]:
        return [
            (0, 0.3, 1.2), (0, 1.8, -0.8), (0, 4.4, 0.0),
            (1, 4.5, 0.0), (1, 3.0, 1.0), (1, 4.5, 0.0),
            (2, 4.5, 0.0), (2, 6.3, -0.8), (2, 8.5, 0.0),
            (3, 8.5, 0.0), (3, 5.2, 0.8), (4, 4.5, 0.0),
            (4, 6.0, -1.0),
        ]

    def _build_intruders(self, intruder_count: int, random_walk: bool) -> List[Intruder]:
        intruders = [
            Intruder(iid=0, path=self._scenario_path_main(), speed=1.08, random_walk=random_walk),
        ]
        if intruder_count >= 2:
            intruders.append(Intruder(iid=1, path=self._scenario_path_secondary(), speed=0.96, random_walk=random_walk))
        return intruders

    def _setup_scene(self) -> None:
        xs = np.array([c.pos[0] for c in self.cells])
        ys = np.array([c.pos[1] for c in self.cells])

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

            # same floor edges only for clarity
            for c in floor_cells:
                for nb in self.neighbors[c.cid]:
                    ncell = self.cells[nb]
                    if ncell.floor != floor_idx or nb < c.cid:
                        continue
                    ax.plot([c.pos[0], ncell.pos[0]], [c.pos[1], ncell.pos[1]], color="gray", alpha=0.25, linewidth=1.0)

            self.cell_scatters.append(scatter)
            self.dir_quivers.append(quiver)

            intr_sc = ax.scatter([], [], c="black", marker="X", s=120)
            self.intruder_scatters.append(intr_sc)

        self.axes[-1].set_xlabel("Zone axis (corridor)")
        cbar = self.fig.colorbar(self.cell_scatters[0], ax=self.axes, fraction=0.016, pad=0.01)
        cbar.set_label("Vs energy")
        self.status_text = self.fig.text(0.01, 0.985, "", va="top", fontsize=10)

    def _compute_local_inputs(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        local_inputs = np.zeros(TOTAL_CELLS, dtype=float)
        local_dirs: List[np.ndarray] = [np.zeros(2, dtype=float) for _ in range(TOTAL_CELLS)]

        for cell in self.cells:
            risk_dir_sum = np.zeros(2, dtype=float)
            for intr in self.intruders:
                if not intr.active:
                    continue
                if intr.floor != cell.floor:
                    continue

                cpos = np.array(cell.pos, dtype=float)
                rel = intr.position - cpos
                d = float(np.linalg.norm(rel))

                if d > SENSOR_RANGE:
                    continue

                d_safe = max(d, SENSOR_EPS)
                intr_vel = (intr.position - intr.prev_position) / DT
                if d > 1e-8:
                    toward_cell = -rel / d
                else:
                    toward_cell = np.zeros(2, dtype=float)
                approach_speed = max(0.0, float(np.dot(intr_vel, toward_cell)))

                local_input = (LOCAL_A * (1.0 / d_safe)) + (LOCAL_B * approach_speed)
                local_inputs[cell.cid] += local_input

                risk = local_input
                dir_to_intr = normalize(rel)
                risk_dir_sum += risk * dir_to_intr

            local_dirs[cell.cid] = normalize(risk_dir_sum)

        return local_inputs, local_dirs

    def update(self, frame: int):
        self.frame_count += 1

        # 1) intruders move
        for intr in self.intruders:
            intr.update(DT)

        # 2) local sensing
        local_inputs, local_dirs = self._compute_local_inputs()

        # 3) each cell autonomous update (no central command)
        releases = np.zeros(TOTAL_CELLS, dtype=float)
        fired_cells = 0
        for cell in self.cells:
            released = cell.update(local_inputs[cell.cid], local_dirs[cell.cid])
            releases[cell.cid] = released
            if cell.fired_this_tick:
                fired_cells += 1

        # 4) propagation stage: fire된 셀만 전달
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

        # 5) draw update floor by floor
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
            if intr_points:
                pts = np.array(intr_points)
            else:
                pts = np.empty((0, 2))
            self.intruder_scatters[f].set_offsets(pts)

        active_intruders = sum(1 for intr in self.intruders if intr.active)
        mean_vs = float(np.mean([c.Vs for c in self.cells]))
        self.status_text.set_text(
            f"t={self.frame_count*DT:5.2f}s | active_intruders={active_intruders} | fired={fired_cells} | mean_Vs={mean_vs:.3f}"
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
