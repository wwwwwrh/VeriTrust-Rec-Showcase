import base64
import json
from dataclasses import replace

import httpx
import pytest

from data_pipeline.collectors.github_collector import (
    CollectorConfig,
    DiscoveryCandidate,
    GitHubAuthenticationError,
    GitHubClient,
    GitHubCollector,
    GitHubRequestError,
    select_balanced_candidates,
    stable_hash,
    load_config,
)


def test_load_config_uses_yaml_defaults():
    config = load_config()
    assert config.max_total_repos == 20
    assert config.queries
    assert config.collector_version == "0.2.0"
    assert load_config(profile_name="stage_a_55").max_total_repos == 55


def test_stage_c_uses_narrow_queries_without_losing_query_groups():
    stage_b = load_config(profile_name="stage_b_105")
    stage_c = load_config(profile_name="stage_c_205")
    assert set(stage_c.query_to_group.values()) == set(stage_c.group_shares)
    assert len(set(stage_c.query_to_group.values())) == 9
    assert "topic:data-science" in stage_b.queries
    assert "topic:data-science" not in stage_c.queries
    assert all("readme" not in query for query, group in stage_c.query_to_group.items() if group == "education")
    assert all(
        any(term in query for term in ("machine-learning", "deep-learning", "llm", "pytorch"))
        for query, group in stage_c.query_to_group.items() if group == "education"
    )


def test_client_adds_optional_token_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    config = replace(load_config(), cache_root=tmp_path / "cache")
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"ok": True}, headers={"x-ratelimit-remaining": "4999"})

    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler), headers=client.http.headers)
    assert client.get_json("/test")[0] == {"ok": True}
    assert client.get_json("/test")[2] is True
    assert calls == 1
    client.close()


def test_rate_limit_wait_is_bounded(tmp_path):
    config = replace(load_config(), cache_enabled=False, max_retries=0, rate_limit_max_wait_seconds=1, cache_root=tmp_path)
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "4102444800"})))
    with pytest.raises(GitHubRequestError):
        client.get_json("/limited")
    client.close()


def test_collector_writes_provenance_and_isolates_readme_failure(tmp_path):
    base = load_config()
    config = replace(
        base, queries=["topic:test"], max_total_repos=2, max_repos_per_query=2, per_page=2,
        cache_enabled=False, max_retries=0, raw_root=tmp_path / "raw", cache_root=tmp_path / "cache",
        failed_fetches=tmp_path / "failed.jsonl", logs_root=tmp_path / "logs",
    )
    logger = __import__("logging").getLogger("test")
    client = GitHubClient(config, logger, sleeper=lambda _: None)

    readme = base64.b64encode(b"# one").decode()
    def handler(request):
        path = request.url.path
        if path == "/search/repositories":
            return httpx.Response(200, json={"items": [{"id": 1, "full_name": "org/one"}, {"id": 2, "full_name": "org/two"}]})
        if path == "/repos/org/one":
            return httpx.Response(200, json={"id": 1, "full_name": "org/one", "html_url": "https://github.com/org/one"})
        if path == "/repos/org/two":
            return httpx.Response(200, json={"id": 2, "full_name": "org/two", "html_url": "https://github.com/org/two"})
        if path == "/repos/org/one/readme":
            return httpx.Response(200, json={"encoding": "base64", "content": readme, "html_url": "https://github.com/org/one/blob/main/README.md"})
        if path == "/repos/org/two/readme":
            return httpx.Response(404, json={"message": "Not Found"})
        raise AssertionError(path)

    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    summary = GitHubCollector(config, client=client).run()
    assert summary["repositories"] == 2
    assert summary["readme_success"] == 1
    assert summary["readme_failures"] == 1
    snapshots = list((config.raw_root).rglob("*.json"))
    repository = next(path for path in snapshots if "github_repository" in str(path))
    record = json.loads(repository.read_text(encoding="utf-8"))
    assert {"collector_version", "schema_version", "retrieved_at", "source_url", "source_type", "source_id", "source_hash"} <= record.keys()
    assert record["provenance"]["query_used"] == "topic:test"
    assert record["provenance"]["search_page"] == 1
    assert record["provenance"]["search_position"] == 1
    assert record["provenance"]["selection_method"] == "balanced_query_groups_v1"
    search = next(path for path in snapshots if "github_search" in str(path))
    assert json.loads(search.read_text(encoding="utf-8"))["provenance"]["result_positions"][0]["search_position"] == 1
    assert config.failed_fetches.exists()


def test_retries_transient_server_error(tmp_path):
    config = replace(load_config(), cache_enabled=False, max_retries=1, backoff_seconds=0, cache_root=tmp_path)
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    responses = iter([httpx.Response(500), httpx.Response(200, json={"ok": True})])
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: next(responses)))
    assert client.get_json("/flaky")[0] == {"ok": True}
    assert client.retry_count == 1
    client.close()


def test_stage_profiles_require_token_and_balanced_selection_is_stable(monkeypatch):
    stage = load_config(profile_name="stage_c_205")
    assert stage.max_total_repos == 205
    assert stage.discovery_cap == 400
    assert stage.requires_token is True
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(GitHubAuthenticationError):
        GitHubClient(stage, __import__("logging").getLogger("test"))
    candidates = [
        DiscoveryCandidate(3, "org/three", "a", 2, ()),
        DiscoveryCandidate(1, "org/one", "a", 1, ()),
        DiscoveryCandidate(2, "org/two", "b", 1, ()),
        DiscoveryCandidate(4, "org/four", "b", 2, ()),
    ]
    first = select_balanced_candidates(candidates, 3, {"a": 0.5, "b": 0.5}, set())
    second = select_balanced_candidates(list(reversed(candidates)), 3, {"a": 0.5, "b": 0.5}, set())
    assert [item.repository_id for item in first] == [item.repository_id for item in second] == [1, 3, 2]
    renamed = DiscoveryCandidate(1, "new-owner/renamed", "a", 1, ())
    assert select_balanced_candidates([renamed], 1, {"a": 1.0}, {1}) == []


def test_discovery_cap_does_not_starve_later_query_groups(tmp_path):
    config = replace(
        load_config(), queries=["a1", "a2", "b1", "c1"],
        query_to_group={"a1": "a", "a2": "a", "b1": "b", "c1": "c"},
        group_shares={"a": 0.4, "b": 0.3, "c": 0.3}, discovery_cap=1,
        max_repos_per_query=1, per_page=1, cache_enabled=False,
        raw_root=tmp_path / "raw", cache_root=tmp_path / "cache",
    )
    seen: list[str] = []

    def handler(request):
        query = request.url.params["q"]
        seen.append(query)
        repo_id = {"a1": 1, "a2": 2, "b1": 3, "c1": 4}[query]
        return httpx.Response(200, json={"items": [{"id": repo_id, "full_name": f"org/{query}"}]})

    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    collector = GitHubCollector(config, client=client)
    candidates = collector._discover()
    client.close()

    assert seen == ["a1", "b1", "c1"]
    assert {item.group for item in candidates} == {"a", "b", "c"}


def test_conditional_cache_revalidation_uses_etag(tmp_path):
    config = replace(load_config(), cache_root=tmp_path / "cache")
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    responses = []

    def handler(request):
        responses.append(request)
        if len(responses) == 1:
            return httpx.Response(200, json={"id": 1}, headers={"etag": '"abc"', "last-modified": "Mon, 01 Jan 2024 00:00:00 GMT"})
        assert request.headers["if-none-match"] == '"abc"'
        assert request.headers["if-modified-since"] == "Mon, 01 Jan 2024 00:00:00 GMT"
        return httpx.Response(304)

    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.get_json("/repos/org/repo", cache_mode="revalidate")[0] == {"id": 1}
    body, _, cached = client.get_json("/repos/org/repo", cache_mode="revalidate")
    assert body == {"id": 1} and cached is True
    assert client.cache_revalidations == 1
    client.close()


def test_authentication_is_fatal_and_secondary_limit_uses_bounded_retry_after(tmp_path):
    config = replace(load_config(), cache_enabled=False, max_retries=2, cache_root=tmp_path)
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    with pytest.raises(GitHubAuthenticationError):
        client.get_json("/unauthorized")
    assert client.request_count == 1 and client.retry_count == 0
    client.close()

    waits = []
    config = replace(config, max_retries=1, rate_limit_max_wait_seconds=5)
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=waits.append)
    responses = iter([httpx.Response(403, headers={"retry-after": "2"}), httpx.Response(200, json={"ok": True})])
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: next(responses)))
    assert client.get_json("/secondary")[0] == {"ok": True}
    assert waits == [2.0]
    client.close()


def test_rate_limit_preflight_tracks_core_and_search_buckets(tmp_path):
    config = replace(load_config(), cache_enabled=False, cache_root=tmp_path)
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    body = {"resources": {
        "core": {"limit": 5000, "remaining": 4900, "reset": 1, "used": 100},
        "search": {"limit": 30, "remaining": 29, "reset": 2, "used": 1},
    }}
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    assert client.preflight_rate_limit() == body
    assert client.rate_limits["core"]["x-ratelimit-remaining"] == "4900"
    assert client.rate_limits["search"]["x-ratelimit-limit"] == "30"
    client.close()


def _write_baseline_snapshot(run, source_type, source_id, payload):
    target = run / source_type / f"{source_id.replace('/', '__')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "collector_version": "0.1.0", "schema_version": "1.0.0", "run_id": run.name,
        "retrieved_at": "2026-01-01T00:00:00Z", "source_type": source_type,
        "source_id": source_id, "source_url": f"https://github.com/{source_id}",
        "source_hash": stable_hash(payload), "provenance": {
            "query_used": "topic:machine-learning", "search_page": 1, "search_position": 1,
        }, "payload": payload,
    }
    target.write_text(json.dumps(wrapper), encoding="utf-8")
    return wrapper


def test_stage_run_materializes_baseline_as_self_contained_snapshots(tmp_path):
    baseline = tmp_path / "baseline"
    metadata = {"id": 10, "full_name": "old/repo", "html_url": "https://github.com/old/repo"}
    readme = {"api_response": {"sha": "abc"}, "decoded_text": "# Evidence"}
    old_metadata = _write_baseline_snapshot(baseline, "github_repository", "old/repo", metadata)
    old_readme = _write_baseline_snapshot(baseline, "github_readme", "old/repo", readme)
    config = replace(
        load_config(), queries=[], query_to_group={}, group_shares={}, max_total_repos=1,
        raw_root=tmp_path / "raw", cache_enabled=False,
    )
    client = GitHubClient(config, __import__("logging").getLogger("test"), sleeper=lambda _: None)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError(request))))
    collector = GitHubCollector(config, client=client, baseline_run=baseline)
    summary = collector.run()
    assert summary["repositories"] == 1 and summary["self_contained"] is True
    new_metadata = json.loads(next((collector.run_dir / "github_repository").glob("*.json")).read_text(encoding="utf-8"))
    new_readme = json.loads(next((collector.run_dir / "github_readme").glob("*.json")).read_text(encoding="utf-8"))
    for original, materialized in ((old_metadata, new_metadata), (old_readme, new_readme)):
        assert materialized["run_id"] == collector.run_id
        assert materialized["payload"] == original["payload"]
        assert materialized["source_hash"] == original["source_hash"]
        assert materialized["retrieved_at"] == original["retrieved_at"]
        lineage = materialized["provenance"]["materialization"]
        assert lineage["materialized_from_run_id"] == baseline.name
        assert lineage["original_wrapper_hash"] == stable_hash(original)
