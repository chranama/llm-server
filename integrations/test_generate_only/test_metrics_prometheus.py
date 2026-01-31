# integrations/test_generate_only/test_metrics_prometheus.py
from __future__ import annotations

import pytest

from integrations.lib.fixtures import load_prompt
from integrations.lib.metrics import fetch_metrics_text, parse_prometheus_text

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.generate_only,
    pytest.mark.requires_api_key,
    pytest.mark.requires_metrics,
]


def _sum_metrics_by_name_substrings(snap, needles: tuple[str, ...]) -> float:
    """
    Sum all series values for any metric name containing any substring in `needles`.
    Works with MetricsSnapshot(series: dict[name][labels_key]=value).
    """
    needles_l = tuple(n.lower() for n in needles)
    total = 0.0
    for name, series_map in snap.series.items():
        lname = name.lower()
        if any(n in lname for n in needles_l):
            for _lk, v in series_map.items():
                total += float(v)
    return total


def _delta_sum(before, after, needles: tuple[str, ...]) -> float:
    return _sum_metrics_by_name_substrings(after, needles) - _sum_metrics_by_name_substrings(before, needles)


async def test_metrics_reachable_and_parseable(client, assert_mode):
    assert assert_mode is True

    text = await fetch_metrics_text(client)
    snap = parse_prometheus_text(text)

    assert isinstance(snap.series, dict)
    assert snap.series, "expected some metrics to be present"


async def test_metrics_change_after_generate(client, assert_mode):
    """
    Avoid hardcoding metric names. We only assert that *something* changes after a request.
    """
    assert assert_mode is True

    before = parse_prometheus_text(await fetch_metrics_text(client))

    prompt = load_prompt("generate_ping.txt")
    r = await client.post("/v1/generate", json={"prompt": prompt, "max_new_tokens": 8, "temperature": 0.2})
    r.raise_for_status()

    after = parse_prometheus_text(await fetch_metrics_text(client))

    changed = False
    for name, series_map in after.series.items():
        prev_map = before.series.get(name)
        if prev_map is None:
            # new metric appeared
            changed = True
            break
        if prev_map != series_map:
            changed = True
            break

    assert changed, "expected at least one metric series to change after /v1/generate"


@pytest.mark.requires_redis
async def test_cache_metrics_increase_after_repeat_generate(client, assert_mode):
    """
    If redis caching is enabled, repeating identical /v1/generate calls should cause
    some cache-ish counters to move. We match by name substrings to avoid brittleness.

    Strong signal: any "hit"-like counter increases.
    Fallback signal: any "cache"/"redis" metric increases at all.
    """
    assert assert_mode is True

    prompt = load_prompt("generate_cache_key_stable.txt")
    payload = {"prompt": prompt, "max_new_tokens": 16, "temperature": 0.0}

    before = parse_prometheus_text(await fetch_metrics_text(client))

    r1 = await client.post("/v1/generate", json=payload)
    r1.raise_for_status()

    # Optional mid-snapshot retained for debugging / future tightening
    _mid = parse_prometheus_text(await fetch_metrics_text(client))

    r2 = await client.post("/v1/generate", json=payload)
    r2.raise_for_status()

    after = parse_prometheus_text(await fetch_metrics_text(client))

    hit_needles = ("cache_hit", "cache_hits", "hit_total", "hits_total")
    miss_needles = ("cache_miss", "cache_misses", "miss_total", "misses_total")
    generic_needles = ("cache", "redis")

    hit_delta = _delta_sum(before, after, hit_needles)
    miss_delta = _delta_sum(before, after, miss_needles)
    generic_delta = _delta_sum(before, after, generic_needles)

    if hit_delta > 0:
        return

    if generic_delta > 0:
        return

    raise AssertionError(
        "Expected cache-related metrics to increase after repeated identical /v1/generate "
        f"(hit_delta={hit_delta}, miss_delta={miss_delta}, generic_delta={generic_delta})."
    )