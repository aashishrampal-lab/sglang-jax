from types import SimpleNamespace

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqInput
from sgl_jax.srt.managers.scheduler_scoring_cache_mixin import SchedulerScoringCacheMixin
from sgl_jax.srt.managers.scheduler_scoring_dispatch_mixin import (
    SchedulerScoringDispatchMixin,
)
from sgl_jax.srt.managers.scheduler_scoring_execute_mixin import (
    SchedulerScoringExecuteMixin,
)
from sgl_jax.srt.managers.scheduler_scoring_state_mixin import SchedulerScoringStateMixin


class _DummyScheduler(
    SchedulerScoringCacheMixin,
    SchedulerScoringDispatchMixin,
    SchedulerScoringExecuteMixin,
    SchedulerScoringStateMixin,
):
    pass


def test_unpack_scoring_cache_entry_accepts_current_format():
    scheduler = _DummyScheduler()

    current = scheduler._unpack_scoring_cache_entry(("node", "swa", [1], [2], "k", 3.5))
    assert current == ("node", "swa", [1], [2], "k", 3.5)


def test_scoring_cache_metrics_snapshot_tracks_hits_and_misses():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(multi_item_prefill_extend_cache_timeout=1.0)
    )
    scheduler.scoring_cache_nodes = {"rid-1": ("node", "swa", [1], [2], "k", 0.0)}
    scheduler.scoring_cache_prefix_handles_by_key = {("k", (1,)): {"rid-1"}}
    scheduler.scoring_cache_handle_to_prefix_key = {"rid-1": ("k", (1,))}
    scheduler.scoring_cache_handles_created = 1

    scheduler._record_scoring_cache_lookup(path="extend", hit=True, lane_name="short")
    scheduler._record_scoring_cache_lookup(path="extend", hit=False, lane_name="long")

    metrics = scheduler._scoring_cache_metrics_snapshot()
    assert metrics["lookup_queries"] == 2
    assert metrics["lookup_hits"] == 1
    assert metrics["lookup_misses"] == 1
    assert metrics["lookup_by_path"]["extend"]["hits"] == 1


def test_score_from_cache_v2_scaffold_reports_validation_failure():
    scheduler = _DummyScheduler()
    scheduler.init_scoring_state(
        SimpleNamespace(multi_item_prefill_extend_cache_timeout=1.0)
    )
    scheduler.enable_overlap = False
    scheduler.server_args = type("ServerArgs", (), {})()

    result = scheduler.score_from_cache_v2(ScoreFromCacheReqInput(rid="rid-1"))

    assert result.success is False
    assert result.fallback_reason == "missing_cache_handle"
    assert scheduler.score_from_cache_v2_attempted == 1
    assert scheduler.score_from_cache_v2_fallback == 1
