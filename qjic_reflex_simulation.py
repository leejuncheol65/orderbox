"""
QJIC Reflex Cell Network – Domino Energy Propagation Simulation

- 20개의 반사엔진 세포가 단독주택 구조에 배치된다.
- 침입자(도둑)가 대문에서 진입하여 집 안으로 이동한다.
- 각 세포는 중앙 제어 없이 독립적으로 입력을 받아 발화/감쇠/전달을 수행한다.
- 발화 에너지는 이웃 세포에 전달되어 도미노식 연쇄 반응을 만든다.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


ENERGY_TRANSFER_RATIO = 0.382  # 황금비 기반
SENSOR_RANGE = 1.7
DT = 0.08


@dataclass
class ReflexCell:
    """QJIC 반사엔진 세포 모델 (독립 판단 단위)."""

    cid: int
    pos: Tuple[float, float]
    Vs: float = 0.0
    VTH: float = 1.0
    A: float = 0.94
    K: float = 0.19
    LEAK: float = 0.03

    fired_last_tick: bool = False

    def receive_energy(self, e: float) -> None:
        """외부 자극(침입자/이웃 세포) 에너지를 누적한다."""
        self.Vs += max(0.0, e)

    def fire(self) -> float:
        """발화 시 잔류 에너지와 전달 에너지를 계산해 반환한다."""
        self.fired_last_tick = True
        released = self.Vs * ENERGY_TRANSFER_RATIO
        residue = self.Vs * (1.0 - ENERGY_TRANSFER_RATIO) * 0.38
        self.Vs = residue
        return released

    def update(self, dt: float) -> float:
        """세포의 자율 갱신: 임계 발화 or 누설/감쇠."""
        self.fired_last_tick = False

        if self.Vs >= self.VTH:
            return self.fire()

        # 발화하지 않았을 때는 감쇠 + 누설 + 소규모 내부 피드백만 적용
        self.Vs = max(0.0, (self.Vs * self.A) - self.LEAK * dt + self.K * 0.01)
        return 0.0


@dataclass
class Intruder:
    """대문에서 진입해 경로를 따라 움직이는 침입자 모델."""

    path: List[Tuple[float, float]]
    speed: float = 1.25
    position: Tuple[float, float] = (0.0, 0.0)
    segment_idx: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        self.position = self.path[0]

    def update(self, dt: float) -> None:
        if not self.active:
            return

        if self.segment_idx >= len(self.path) - 1:
            self.active = False
            return

        sx, sy = self.position
        tx, ty = self.path[self.segment_idx + 1]
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)

        if dist < 1e-9:
            self.segment_idx += 1
            return

        step = self.speed * dt
        if step >= dist:
            self.position = (tx, ty)
            self.segment_idx += 1
        else:
            self.position = (sx + dx / dist * step, sy + dy / dist * step)


class HouseSimulation:
    """단독주택 기반 QJIC 도미노 에너지 시뮬레이션 컨트롤러."""

    def __init__(self) -> None:
        self.cells = self._build_house_cells()
        self.neighbors = self._build_neighbor_graph(self.cells)
        self.intruder = Intruder(path=self._intruder_path())

        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.scatter = None
        self.intruder_dot = None
        self.time_text = None
        self.frame_count = 0

        self._setup_scene()

    @staticmethod
    def _build_house_cells() -> List[ReflexCell]:
        """입구→복도→방 구조를 반영해 총 20개 세포를 배치."""
        points = [
            # Entry / porch / foyer
            (0.5, 2.0), (1.3, 2.0), (2.0, 2.0), (2.7, 2.0),
            # Corridor spine
            (3.5, 2.0), (4.3, 2.0), (5.1, 2.0), (5.9, 2.0), (6.7, 2.0),
            # Living room cluster
            (4.1, 3.1), (4.9, 3.3), (5.7, 3.0),
            # Room 1
            (6.7, 3.3), (7.5, 3.3),
            # Room 2
            (6.7, 1.0), (7.5, 1.0),
            # Room 3
            (8.1, 2.0), (8.8, 2.6),
            # Kitchen / back door-window area
            (5.5, 0.7), (8.8, 0.9),
        ]
        return [ReflexCell(cid=i, pos=p) for i, p in enumerate(points)]

    @staticmethod
    def _intruder_path() -> List[Tuple[float, float]]:
        return [
            (0.2, 2.0),  # Gate opens
            (1.2, 2.0),  # Entry
            (2.6, 2.0),  # Foyer
            (4.0, 2.0),  # Corridor in
            (5.2, 2.9),  # Living room glance
            (6.8, 3.2),  # Room 1 side
            (6.7, 2.0),  # Back to corridor
            (7.6, 1.0),  # Room 2
            (8.7, 0.9),  # Back door area
            (8.1, 2.0),  # Room 3
        ]

    @staticmethod
    def _build_neighbor_graph(cells: List[ReflexCell]) -> Dict[int, List[int]]:
        """상하좌우/근접 연결 기반의 이웃 그래프를 구성."""
        graph: Dict[int, List[int]] = {c.cid: [] for c in cells}
        positions = np.array([c.pos for c in cells])

        for i, cell in enumerate(cells):
            dists = np.hypot(positions[:, 0] - cell.pos[0], positions[:, 1] - cell.pos[1])
            for j, d in enumerate(dists):
                if i == j:
                    continue
                if d <= 1.02:  # 복도/방 단위 격자에서 인접한 거리
                    graph[cell.cid].append(j)

        # 공간 분리 구간(방 입구) 연결을 조금 더 명시적으로 보강
        forced_links = [(8, 12), (8, 14), (16, 17), (7, 18), (16, 19)]
        for a, b in forced_links:
            if b not in graph[a]:
                graph[a].append(b)
            if a not in graph[b]:
                graph[b].append(a)

        return graph

    def _setup_scene(self) -> None:
        self.ax.set_title("QJIC Reflex Cell Network – Domino Energy Propagation")
        self.ax.set_xlim(-0.2, 9.4)
        self.ax.set_ylim(0.0, 4.2)
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.15)

        # 단독주택 도면 윤곽(간략)
        walls = [
            Rectangle((0.0, 1.5), 3.0, 1.0, fill=False, linewidth=1.2),   # entry/foyer
            Rectangle((3.0, 1.5), 4.0, 1.0, fill=False, linewidth=1.2),   # corridor
            Rectangle((3.5, 2.5), 2.6, 1.2, fill=False, linewidth=1.2),   # living room
            Rectangle((6.4, 2.7), 1.4, 0.9, fill=False, linewidth=1.2),   # room1
            Rectangle((6.4, 0.6), 1.4, 0.8, fill=False, linewidth=1.2),   # room2
            Rectangle((7.7, 1.6), 1.3, 1.6, fill=False, linewidth=1.2),   # room3
            Rectangle((5.0, 0.4), 1.2, 0.8, fill=False, linewidth=1.2),   # kitchen
        ]
        for w in walls:
            self.ax.add_patch(w)

        self.ax.text(0.0, 2.75, "Gate", fontsize=9)
        self.ax.text(3.2, 2.65, "Corridor", fontsize=9)
        self.ax.text(4.0, 3.8, "Living", fontsize=9)
        self.ax.text(6.6, 3.75, "Room1", fontsize=8)
        self.ax.text(6.6, 1.45, "Room2", fontsize=8)
        self.ax.text(8.0, 3.35, "Room3", fontsize=8)
        self.ax.text(5.1, 1.3, "Kitchen", fontsize=8)
        self.ax.text(8.45, 0.45, "Back door/window", fontsize=8)

        # 초기 세포 렌더링
        coords = np.array([c.pos for c in self.cells])
        self.scatter = self.ax.scatter(
            coords[:, 0], coords[:, 1], c=np.zeros(len(self.cells)), cmap="jet", vmin=0, vmax=1.8, s=260, edgecolors="black"
        )

        # 연결선(자극 전달 경로)
        for src, nbrs in self.neighbors.items():
            x0, y0 = self.cells[src].pos
            for nb in nbrs:
                if nb < src:
                    continue
                x1, y1 = self.cells[nb].pos
                self.ax.plot([x0, x1], [y0, y1], color="gray", alpha=0.18, linewidth=1.0)

        self.intruder_dot = self.ax.scatter([self.intruder.position[0]], [self.intruder.position[1]], c="black", s=110, marker="X")
        self.time_text = self.ax.text(0.05, 0.96, "", transform=self.ax.transAxes, va="top")

        cbar = self.fig.colorbar(self.scatter, ax=self.ax, fraction=0.028)
        cbar.set_label("Cell Energy Vs")

    def _inject_intruder_energy(self) -> None:
        ix, iy = self.intruder.position
        for cell in self.cells:
            cx, cy = cell.pos
            d = math.hypot(ix - cx, iy - cy)
            incoming = max(0.0, 1.0 - d / SENSOR_RANGE)
            cell.receive_energy(incoming * 0.28)

    def update(self, frame: int):
        """한 프레임 갱신: 침입자 이동 + 세포 자율 업데이트 + 도미노 전달."""
        self.frame_count += 1
        self.intruder.update(DT)

        if self.intruder.active:
            self._inject_intruder_energy()

        # 1단계: 각 세포가 개별적으로 발화 여부를 판단
        releases = np.zeros(len(self.cells))
        for i, cell in enumerate(self.cells):
            releases[i] = cell.update(DT)

        # 2단계: 발화된 세포의 에너지를 이웃에 전달 (명령 아닌 자극 전파)
        for i, released in enumerate(releases):
            if released <= 0:
                continue
            nbrs = self.neighbors[i]
            if not nbrs:
                continue
            share = released / len(nbrs)
            for nb in nbrs:
                self.cells[nb].receive_energy(share)

        # 시각 업데이트
        energies = np.array([c.Vs for c in self.cells])
        self.scatter.set_array(energies)
        self.intruder_dot.set_offsets(np.array([[self.intruder.position[0], self.intruder.position[1]]]))

        fired_count = sum(1 for c in self.cells if c.fired_last_tick)
        self.time_text.set_text(
            f"t={self.frame_count * DT:5.2f}s | intruder={'inside' if self.intruder.active else 'stopped'} | fired={fired_count}"
        )
        return self.scatter, self.intruder_dot, self.time_text

    def run(self) -> None:
        self.anim = FuncAnimation(self.fig, self.update, interval=55, blit=False)
        plt.tight_layout()
        plt.show()


def run_headless_steps(steps: int = 200) -> None:
    """CI/검증용: GUI 없이 지정 프레임만 계산."""
    sim = HouseSimulation()
    for i in range(steps):
        sim.update(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QJIC Reflex Cell domino energy animation")
    parser.add_argument("--headless-steps", type=int, default=0, help="Run simulation loop without GUI for testing")
    args = parser.parse_args()

    if args.headless_steps > 0:
        run_headless_steps(args.headless_steps)
    else:
        HouseSimulation().run()
