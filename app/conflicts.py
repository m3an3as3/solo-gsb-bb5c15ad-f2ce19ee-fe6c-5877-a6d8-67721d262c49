"""事件阶段窗口的冲突分析（sweep-line）。

每个事件按服务端 UTC 拥有两个「半开」阶段区间：

    commit: [created_at,      commit_deadline)
    reveal: [commit_deadline, reveal_deadline)

冲突定义：同一时刻有两个或更多事件处于**同一**阶段（不同阶段之间不
视为冲突）。半开边界语义下，右端点事件先「离场」、左端点事件再
「入场」，因此首尾相接（A 的结束恰为 B 的开始）不产生冲突，零长度
区间也不占据任何时刻。
"""

from dataclasses import dataclass
from datetime import datetime
from heapq import heappop, heappush
from typing import Iterable
from uuid import UUID

COMMIT = "commit"
REVEAL = "reveal"


@dataclass(frozen=True)
class PhaseWindow:
    event_id: UUID
    phase: str  # commit | reveal
    start: datetime  # 含（闭端）
    end: datetime  # 不含（开端）


@dataclass(frozen=True)
class ConflictSegment:
    start: datetime
    end: datetime
    phase: str
    event_ids: tuple[UUID, ...]

    @property
    def size(self) -> int:
        return len(self.event_ids)


@dataclass(frozen=True)
class TrackAssignment:
    event_id: UUID
    phase: str  # commit | reveal
    track: int  # 1 起编号


@dataclass(frozen=True)
class EventPair:
    """同一阶段内区间重叠的事件对（event_id_a < event_id_b）。"""

    phase: str
    event_id_a: UUID
    event_id_b: UUID


def sweep_conflicts(windows: Iterable[PhaseWindow]) -> list[ConflictSegment]:
    """扫描线计算各阶段轨道上的重叠片段。

    对每个阶段独立扫描：把每个区间展开为 (起点, 入场) / (终点, 离场)
    两个事件，同一时刻先处理离场再处理入场（半开边界）。相邻两个事件
    时刻之间活动事件集合不变，集合大小 >= 2 即为一个冲突片段。

    返回片段按 (start, end, phase) 排序；片段内 event_ids 按 UUID
    稳定排序。
    """
    windows_by_phase: dict[str, list[PhaseWindow]] = {}
    for window in windows:
        if window.end <= window.start:
            continue  # 退化（零长度或反向）区间不占任何时刻
        windows_by_phase.setdefault(window.phase, []).append(window)

    segments: list[ConflictSegment] = []
    for phase, phase_windows in windows_by_phase.items():
        # kind: 0 = 离场（排序在前，先处理），1 = 入场
        events: list[tuple[datetime, int, UUID]] = []
        for w in phase_windows:
            events.append((w.start, 1, w.event_id))
            events.append((w.end, 0, w.event_id))
        events.sort(key=lambda e: (e[0], e[1]))

        active: set[UUID] = set()
        prev_t: datetime | None = None
        i = 0
        while i < len(events):
            t = events[i][0]
            # [prev_t, t) 之间活动集合不变；长度为 0 的缝隙不产生片段
            if prev_t is not None and len(active) >= 2:
                segments.append(
                    ConflictSegment(prev_t, t, phase, tuple(sorted(active)))
                )
            # 同一时刻的所有离场/入场原子处理完毕，集合才属于 [t, 下一时刻)
            while i < len(events) and events[i][0] == t:
                _, kind, event_id = events[i]
                if kind == 1:
                    active.add(event_id)
                else:
                    active.discard(event_id)
                i += 1
            prev_t = t

    segments.sort(key=lambda s: (s.start, s.end, s.phase))
    return segments


def schedule_tracks(
    windows: Iterable[PhaseWindow],
) -> tuple[dict[str, list[TrackAssignment]], dict[str, int]]:
    """对每个阶段独立做最少轨道的确定性区间着色（interval graph coloring）。

    区间按 (start, end, event_id) 排序后贪心分配：把已释放（区间结束
    时刻不晚于当前起点，半开边界下首尾相接即可复用）的轨道回收，再取
    当前编号最小的空闲轨道。该策略对区间图是最优的——所需轨道数恰为
    最大重叠度，编号规则又使结果完全确定：相同输入恒得相同分配。

    退化（零长度或反向）区间不占据任何时刻，不参与着色（存储层保证
    事件的 commit / reveal 窗口恒为正长度，此处与 sweep_conflicts
    的处理保持一致）。

    返回 (assignments_by_phase, required_by_phase)：
      - 每个阶段的分配按 event_id 升序；轨道编号从 1 起连续；
      - required_by_phase[phase] 为该阶段实际用到的最大轨道编号。
    """
    windows_by_phase: dict[str, list[PhaseWindow]] = {}
    for window in windows:
        if window.end <= window.start:
            continue  # 退化区间不占任何时刻
        windows_by_phase.setdefault(window.phase, []).append(window)

    assignments_by_phase: dict[str, list[TrackAssignment]] = {}
    required_by_phase: dict[str, int] = {}
    for phase, phase_windows in windows_by_phase.items():
        # 确定性：先开始者优先；同时开始时先结束者优先；再以 event_id 兜底
        phase_windows.sort(key=lambda w: (w.start, w.end, w.event_id))
        # releasing: (释放时刻, 轨道编号) 堆，释放最早者先回收
        # free_tracks: 已回收轨道编号的最小堆，复用编号最小者
        releasing: list[tuple[datetime, int]] = []
        free_tracks: list[int] = []
        next_track = 1
        assignments: list[TrackAssignment] = []
        for w in phase_windows:
            while releasing and releasing[0][0] <= w.start:
                heappush(free_tracks, heappop(releasing)[1])
            if free_tracks:
                track = heappop(free_tracks)
            else:
                track = next_track
                next_track += 1
            heappush(releasing, (w.end, track))
            assignments.append(TrackAssignment(w.event_id, phase, track))

        assignments.sort(key=lambda a: a.event_id)
        assignments_by_phase[phase] = assignments
        required_by_phase[phase] = next_track - 1

    return assignments_by_phase, required_by_phase


def overload_pairs(
    windows: Iterable[PhaseWindow], track_limit: int
) -> list[EventPair]:
    """找出在给定轨道上限下「无法同时被容纳」的同阶段重叠事件对。

    复用 sweep-line：只收集活动事件数严格超过 track_limit 的时间片段；
    每个超载片段内活动集合的全部事件对都构成冲突对（半开语义与
    sweep_conflicts 一致，首尾相接不会同处一个活动集合）。事件对去重
    后按 (phase, event_id_a, event_id_b) 稳定排序，event_id_a 为
    两者中较小的 UUID。
    """
    windows_by_phase: dict[str, list[PhaseWindow]] = {}
    for window in windows:
        if window.end <= window.start:
            continue
        windows_by_phase.setdefault(window.phase, []).append(window)

    pairs: set[EventPair] = set()
    for phase, phase_windows in windows_by_phase.items():
        events: list[tuple[datetime, int, UUID]] = []
        for w in phase_windows:
            events.append((w.start, 1, w.event_id))
            events.append((w.end, 0, w.event_id))
        events.sort(key=lambda e: (e[0], e[1]))

        active: set[UUID] = set()
        prev_t: datetime | None = None
        i = 0
        while i < len(events):
            t = events[i][0]
            if prev_t is not None and len(active) > track_limit:
                ordered = sorted(active)
                for x in range(len(ordered)):
                    for y in range(x + 1, len(ordered)):
                        pairs.add(EventPair(phase, ordered[x], ordered[y]))
            while i < len(events) and events[i][0] == t:
                _, kind, event_id = events[i]
                if kind == 1:
                    active.add(event_id)
                else:
                    active.discard(event_id)
                i += 1
            prev_t = t

    return sorted(pairs, key=lambda p: (p.phase, p.event_id_a, p.event_id_b))
