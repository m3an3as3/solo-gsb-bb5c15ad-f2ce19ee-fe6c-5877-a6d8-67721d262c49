"""POST /events/schedule 资源排程测试。

覆盖：
  - 纯函数 schedule_tracks：最少轨道数、半开边界首尾相接复用、
    确定性（乱序输入结果一致）、跨阶段独立着色、零长度区间忽略
  - 纯函数 overload_pairs：仅超载片段内的事件对、半开语义与稳定排序
  - API：去重、未知 ID 404、非 UUID 422、非法 track_limit 400、
    空请求、按 event_id 排序的 assignment、确定性重放、只读不修改数据
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_schedule_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/schedule.db"

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_module  # noqa: E402
from app.conflicts import (  # noqa: E402
    COMMIT,
    REVEAL,
    EventPair,
    PhaseWindow,
    TrackAssignment,
    overload_pairs,
    schedule_tracks,
)
from app.database import SessionLocal  # noqa: E402
from app.models import Event  # noqa: E402

BASE = datetime(2030, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def t(minutes: int) -> datetime:
    return BASE + timedelta(minutes=minutes)


class FixedClock:
    def __call__(self) -> datetime:
        return BASE


client = TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    monkeypatch.setattr(main_module, "now_utc", FixedClock())
    with client:  # 触发 lifespan 建表（幂等）
        yield


def make_event(event_id: UUID, created: datetime, commit_dl: datetime,
               reveal_dl: datetime, title: str = "event") -> None:
    db = SessionLocal()
    try:
        db.add(
            Event(
                id=event_id,
                title=title,
                created_at=created,
                commit_deadline=commit_dl,
                reveal_deadline=reveal_dl,
            )
        )
        db.commit()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 纯函数：schedule_tracks
# --------------------------------------------------------------------------- #

def test_pure_minimum_tracks_and_numbering():
    # 同一阶段三区间：A[0,10) B[5,15) C[10,20)，最大重叠 2
    a, b, c = uuid4(), uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, COMMIT, t(5), t(15)),
        PhaseWindow(c, COMMIT, t(10), t(20)),
    ]
    assignments, required = schedule_tracks(windows)
    assert required == {COMMIT: 2}
    assert {a: 1, b: 2, c: 1} == {x.event_id: x.track for x in assignments[COMMIT]}
    # 分配按 event_id 排序
    ids = [x.event_id for x in assignments[COMMIT]]
    assert ids == sorted(ids)


def test_pure_touching_boundaries_share_track():
    # A[0,10) 与 B[10,20) 首尾相接：同一 track 即可
    a, b = uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, COMMIT, t(10), t(20)),
    ]
    assignments, required = schedule_tracks(windows)
    assert required == {COMMIT: 1}
    tracks = {x.event_id: x.track for x in assignments[COMMIT]}
    assert tracks == {a: 1, b: 1}


def test_pure_triple_overlap_needs_three_tracks():
    a, b, c = uuid4(), uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, COMMIT, t(0), t(10)),
        PhaseWindow(c, COMMIT, t(0), t(10)),
    ]
    _, required = schedule_tracks(windows)
    assert required == {COMMIT: 3}


def test_pure_phases_colored_independently():
    # 同时段 commit / reveal 各一区间：互不影响，各自只需 1 条
    a, b = uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, REVEAL, t(0), t(10)),
    ]
    assignments, required = schedule_tracks(windows)
    assert required == {COMMIT: 1, REVEAL: 1}
    assert assignments[COMMIT] == [TrackAssignment(a, COMMIT, 1)]
    assert assignments[REVEAL] == [TrackAssignment(b, REVEAL, 1)]


def test_pure_deterministic_under_input_permutation():
    a, b, c, d = (uuid4() for _ in range(4))
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, COMMIT, t(2), t(8)),
        PhaseWindow(c, COMMIT, t(1), t(12)),
        PhaseWindow(d, COMMIT, t(5), t(15)),
    ]
    first_assign, first_req = schedule_tracks(list(windows))
    second_assign, second_req = schedule_tracks(list(reversed(windows)))
    assert first_req == second_req
    key = lambda xs: sorted((x.event_id, x.track) for x in xs)
    assert key(first_assign[COMMIT]) == key(second_assign[COMMIT])


def test_pure_zero_length_window_ignored():
    a, b = uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(5), t(5)),
        PhaseWindow(b, COMMIT, t(0), t(10)),
    ]
    assignments, required = schedule_tracks(windows)
    assert required == {COMMIT: 1}
    assert [x.event_id for x in assignments[COMMIT]] == [b]


def test_pure_empty_input():
    assignments, required = schedule_tracks([])
    assert assignments == {} and required == {}


# --------------------------------------------------------------------------- #
# 纯函数：overload_pairs
# --------------------------------------------------------------------------- #

def test_pure_overload_pairs_within_limit_none():
    a, b = uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, COMMIT, t(0), t(10)),
    ]
    assert overload_pairs(windows, 2) == []


def test_pure_overload_pairs_triple_with_limit_one():
    small = UUID("00000000-0000-0000-0000-000000000001")
    middle = UUID("00000000-0000-0000-0000-000000000002")
    large = UUID("00000000-0000-0000-0000-000000000003")
    windows = [
        PhaseWindow(large, COMMIT, t(0), t(10)),
        PhaseWindow(small, COMMIT, t(0), t(10)),
        PhaseWindow(middle, COMMIT, t(0), t(10)),
    ]
    pairs = overload_pairs(windows, 1)
    assert pairs == [
        EventPair(COMMIT, small, middle),
        EventPair(COMMIT, small, large),
        EventPair(COMMIT, middle, large),
    ]


def test_pure_overload_pairs_partial_chain_limit_one():
    # A[0,10) B[5,15) C[10,20)：从不三者同时重叠，limit=1 时
    # [5,10) 超载对 (A,B)，[10,15) 超载对 (B,C)；A 与 C 首尾相接不算
    a, b, c = uuid4(), uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, COMMIT, t(5), t(15)),
        PhaseWindow(c, COMMIT, t(10), t(20)),
    ]
    pairs = overload_pairs(windows, 1)
    expected = {
        EventPair(COMMIT, min(a, b), max(a, b)),
        EventPair(COMMIT, min(b, c), max(b, c)),
    }
    assert set(pairs) == expected
    # 已按 (phase, event_id_a, event_id_b) 稳定排序
    assert pairs == sorted(pairs, key=lambda p: (p.phase, p.event_id_a, p.event_id_b))


def test_pure_overload_pairs_phase_separation():
    # 同时段但分属 commit / reveal：即使 limit=0 也不产生跨阶段对
    a, b = uuid4(), uuid4()
    windows = [
        PhaseWindow(a, COMMIT, t(0), t(10)),
        PhaseWindow(b, REVEAL, t(0), t(10)),
    ]
    assert overload_pairs(windows, 0) == []


# --------------------------------------------------------------------------- #
# API：POST /events/schedule
# --------------------------------------------------------------------------- #

def test_empty_request_requires_zero_tracks():
    r = client.post("/events/schedule", json={"event_ids": [], "track_limit": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["required_tracks"] == 0
    assert body["feasible"] is True
    assert body["assignments"] == []
    assert body["conflict_pairs"] == []
    assert body["phases"] == ["commit", "reveal"]
    assert body["track_limit"] == 1


def test_missing_event_ids_field_treated_as_empty():
    r = client.post("/events/schedule", json={"track_limit": 3})
    assert r.status_code == 200
    assert r.json()["required_tracks"] == 0


def test_unknown_event_returns_404():
    known = uuid4()
    make_event(known, t(0), t(10), t(20), title="known")
    missing = "00000000-0000-0000-0000-000000000000"
    r = client.post(
        "/events/schedule",
        json={"event_ids": [str(known), missing], "track_limit": 2},
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "EVENT_NOT_FOUND"
    assert missing in detail["message"]


def test_invalid_uuid_returns_422():
    r = client.post(
        "/events/schedule",
        json={"event_ids": ["not-a-uuid"], "track_limit": 2},
    )
    assert r.status_code == 422


@pytest.mark.parametrize("bad_limit", [0, -1])
def test_invalid_track_limit_returns_400(bad_limit):
    r = client.post(
        "/events/schedule",
        json={"event_ids": [], "track_limit": bad_limit},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "INVALID_TRACK_LIMIT"
    assert str(bad_limit) in detail["message"]


def test_track_limit_wrong_type_returns_422():
    r = client.post("/events/schedule", json={"event_ids": [], "track_limit": "two"})
    assert r.status_code == 422


def test_track_limit_missing_returns_422():
    r = client.post("/events/schedule", json={"event_ids": []})
    assert r.status_code == 422


def test_schedule_geometry_feasible_and_sorted_assignments():
    #   created  commit_dl  reveal_dl
    # A    0        10          20
    # B    5        15          25
    # C   10        20          30
    a, b, c = uuid4(), uuid4(), uuid4()
    make_event(a, t(0), t(10), t(20), title="A")
    make_event(b, t(5), t(15), t(25), title="B")
    make_event(c, t(10), t(20), t(30), title="C")

    r = client.post(
        "/events/schedule",
        json={"event_ids": [str(c), str(a), str(b)], "track_limit": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["required_tracks"] == 2
    assert body["feasible"] is True
    assert body["conflict_pairs"] == []

    assignments = body["assignments"]
    # 按 event_id 升序，与请求顺序无关
    ids = [item["event_id"] for item in assignments]
    assert ids == sorted(ids)
    by_id = {UUID(x): item for x, item in zip(ids, assignments)}
    # commit: A[0,10)->1 B[5,15)->2 C[10,20)->1（A 结束即释放，C 复用 1）
    assert by_id[a]["commit_track"] == 1
    assert by_id[b]["commit_track"] == 2
    assert by_id[c]["commit_track"] == 1
    # reveal: A[10,20)->1 B[15,25)->2 C[20,30)->1
    assert by_id[a]["reveal_track"] == 1
    assert by_id[b]["reveal_track"] == 2
    assert by_id[c]["reveal_track"] == 1
    for item in assignments:
        assert set(item) == {"event_id", "commit_track", "reveal_track"}


def test_touching_events_fit_single_track():
    # X: commit [100,110) reveal [110,120)；Y: commit [110,120) reveal [120,130)
    x, y = uuid4(), uuid4()
    make_event(x, t(100), t(110), t(120), title="X")
    make_event(y, t(110), t(120), t(130), title="Y")
    r = client.post(
        "/events/schedule",
        json={"event_ids": [str(x), str(y)], "track_limit": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["required_tracks"] == 1
    assert body["feasible"] is True
    assert body["conflict_pairs"] == []
    tracks = {UUID(item["event_id"]): item for item in body["assignments"]}
    assert tracks[x]["commit_track"] == tracks[y]["commit_track"] == 1
    assert tracks[x]["reveal_track"] == tracks[y]["reveal_track"] == 1


def test_infeasible_limit_returns_conflict_pairs():
    small = UUID("00000000-1000-0000-0000-000000000001")
    middle = UUID("00000000-1000-0000-0000-000000000002")
    large = UUID("00000000-1000-0000-0000-000000000003")
    # 三者 commit 与 reveal 窗口完全重叠 -> 每阶段需 3 条
    make_event(small, t(0), t(10), t(20), title="s")
    make_event(middle, t(0), t(10), t(20), title="m")
    make_event(large, t(0), t(10), t(20), title="l")

    r = client.post(
        "/events/schedule",
        json={"event_ids": [str(large), str(small), str(middle)], "track_limit": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["required_tracks"] == 3
    assert body["feasible"] is False

    pairs = body["conflict_pairs"]
    pair_set = {(p["phase"], UUID(p["event_id_a"]), UUID(p["event_id_b"]))
                for p in pairs}
    expected = {
        (COMMIT, small, middle),
        (COMMIT, small, large),
        (COMMIT, middle, large),
        (REVEAL, small, middle),
        (REVEAL, small, large),
        (REVEAL, middle, large),
    }
    assert pair_set == expected
    # 稳定排序：phase（commit 先）后按 ID 对
    ordered = [(p["phase"], p["event_id_a"], p["event_id_b"]) for p in pairs]
    assert ordered == sorted(ordered)
    for p in pairs:
        assert p["event_id_a"] < p["event_id_b"]


def test_duplicate_ids_deduplicated():
    x = uuid4()
    make_event(x, t(100), t(110), t(120), title="X")
    r = client.post(
        "/events/schedule",
        json={"event_ids": [str(x), str(x), str(x)], "track_limit": 1},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["assignments"]) == 1
    assert body["required_tracks"] == 1
    assert body["feasible"] is True


def test_repeated_requests_are_identical():
    a, b, c, d = (uuid4() for _ in range(4))
    make_event(a, t(0), t(10), t(20), title="A")
    make_event(b, t(2), t(8), t(18), title="B")
    make_event(c, t(1), t(12), t(22), title="C")
    make_event(d, t(5), t(15), t(25), title="D")

    payload = {"event_ids": [str(d), str(a), str(c), str(b)], "track_limit": 3}
    r1 = client.post("/events/schedule", json=payload)
    r2 = client.post("/events/schedule", json=payload)
    assert r1.status_code == r2.status_code == 200
    # 时间戳以外的排程结果完全一致
    b1, b2 = r1.json(), r2.json()
    assert b1["required_tracks"] == b2["required_tracks"]
    assert b1["assignments"] == b2["assignments"]
    assert b1["conflict_pairs"] == b2["conflict_pairs"]
    assert b1["feasible"] == b2["feasible"]


def test_schedule_does_not_modify_events():
    a, b = uuid4(), uuid4()
    make_event(a, t(0), t(10), t(20), title="A")
    make_event(b, t(0), t(10), t(20), title="B")
    db = SessionLocal()
    try:
        before = {
            e.id: (e.created_at, e.commit_deadline, e.reveal_deadline, e.title)
            for e in db.query(Event).all()
        }
    finally:
        db.close()

    r = client.post(
        "/events/schedule",
        json={"event_ids": [str(a), str(b)], "track_limit": 1},
    )
    assert r.status_code == 200

    db = SessionLocal()
    try:
        after = {
            e.id: (e.created_at, e.commit_deadline, e.reveal_deadline, e.title)
            for e in db.query(Event).all()
        }
    finally:
        db.close()
    assert before == after
