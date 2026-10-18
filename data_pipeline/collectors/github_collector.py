"""Provenance-preserving GitHub REST API collector for Phase 0-1 and S3.5."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from dotenv import load_dotenv


class GitHubRequestError(RuntimeError):
    """A finite-retry GitHub request failed."""


class GitHubAuthenticationError(GitHubRequestError):
    """The configured GitHub credential was rejected."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CollectorConfig:
    api_base_url: str
    queries: list[str]
    sort: str
    order: str
    max_repos_per_query: int
    max_total_repos: int
    per_page: int
    fetch_readme: bool
    timeout_seconds: float
    max_retries: int
    backoff_seconds: float
    rate_limit_max_wait_seconds: int
    cache_enabled: bool
    raw_root: Path
    cache_root: Path
    failed_fetches: Path
    logs_root: Path
    collector_version: str
    schema_version: str
    profile_name: str = "baseline_20"
    dataset_purpose: str = "phase_0_1_pipeline_validation"
    query_to_group: dict[str, str] | None = None
    group_shares: dict[str, float] | None = None
    discovery_cap: int = 50
    requires_token: bool = False
    preflight_rate_limit: bool = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | None = None, profile_name: str | None = None) -> CollectorConfig:
    root = project_root()
    path = config_path or root / "configs" / "github.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    github, paths, versions = raw["github"], raw["paths"], raw["versions"]
    profiles = github.get("profiles") or {}
    selected_name = profile_name or "baseline_20"
    if selected_name not in profiles:
        raise ValueError(f"unknown GitHub collector profile: {selected_name}")
    profile = profiles[selected_name]
    all_groups = github.get("query_groups") or {}
    selected_groups = list(profile.get("query_groups") or [])
    missing_groups = sorted(set(selected_groups) - set(all_groups))
    if missing_groups:
        raise ValueError(f"profile references unknown query groups: {missing_groups}")
    query_overrides = profile.get("query_overrides") or {}
    unknown_overrides = sorted(set(query_overrides) - set(selected_groups))
    if unknown_overrides:
        raise ValueError(f"profile query_overrides reference inactive groups: {unknown_overrides}")
    query_to_group = {
        str(query): str(group)
        for group in selected_groups
        for query in query_overrides.get(group, all_groups[group])
    }
    queries = list(query_to_group)
    shares = {str(key): float(value) for key, value in (profile.get("group_shares") or {}).items()}
    if set(shares) != set(selected_groups) or not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-6):
        raise ValueError("profile group_shares must cover selected groups and sum to 1.0")
    target = int(profile["target_total"])
    discovery_cap = int(profile["discovery_cap"])
    if target <= 0 or discovery_cap < target:
        raise ValueError("profile target_total must be positive and discovery_cap must cover it")
    return CollectorConfig(
        api_base_url=github["api_base_url"].rstrip("/"), queries=queries,
        sort=github["sort"], order=github["order"],
        max_repos_per_query=int(profile.get("max_repos_per_query", github["max_repos_per_query"])),
        max_total_repos=target, per_page=int(profile.get("per_page", github["per_page"])), fetch_readme=bool(github["fetch_readme"]),
        timeout_seconds=float(github["timeout_seconds"]), max_retries=int(github["max_retries"]),
        backoff_seconds=float(github["backoff_seconds"]),
        rate_limit_max_wait_seconds=int(github["rate_limit_max_wait_seconds"]),
        cache_enabled=bool(github["cache_enabled"]), raw_root=root / paths["raw_root"],
        cache_root=root / paths["cache_root"], failed_fetches=root / paths["failed_fetches"],
        logs_root=root / paths["logs_root"], collector_version=str(versions["collector_version"]),
        schema_version=str(versions["schema_version"]), profile_name=selected_name,
        dataset_purpose=str(profile["dataset_purpose"]), query_to_group=query_to_group,
        group_shares=shares, discovery_cap=discovery_cap,
        requires_token=bool(profile.get("requires_token", False)),
        preflight_rate_limit=bool(profile.get("preflight_rate_limit", False)),
    )


CacheMode = Literal["default", "bypass", "revalidate"]


class GitHubClient:
    def __init__(self, config: CollectorConfig, logger: logging.Logger, sleeper=time.sleep) -> None:
        self.config, self.logger, self.sleeper = config, logger, sleeper
        self.cache_hits = 0
        self.cache_revalidations = 0
        self.request_count = 0
        self.retry_count = 0
        self.rate_limits: dict[str, dict[str, str]] = {}
        token = os.getenv("GITHUB_TOKEN")
        if config.requires_token and not token:
            raise GitHubAuthenticationError(f"collector profile {config.profile_name} requires GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"VeriTrust-Rec-collector/{config.collector_version}",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = httpx.Client(timeout=httpx.Timeout(config.timeout_seconds), headers=headers, follow_redirects=True)

    @property
    def last_rate_limit(self) -> dict[str, str]:
        if "core" in self.rate_limits:
            return self.rate_limits["core"]
        return next(reversed(self.rate_limits.values()), {}) if self.rate_limits else {}

    def close(self) -> None:
        self.http.close()

    def _cache_file(self, endpoint: str, params: dict[str, Any] | None) -> Path:
        key = stable_hash({"endpoint": endpoint, "params": params or {}})
        return self.config.cache_root / f"{key}.json"

    def _capture_rate_limit(self, headers: httpx.Headers) -> None:
        values = {key.lower(): value for key, value in headers.items() if key.lower().startswith("x-ratelimit-")}
        bucket = values.get("x-ratelimit-resource", "unknown")
        if values:
            self.rate_limits[bucket] = values

    def _rate_limit_wait(self, headers: httpx.Headers) -> float:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        reset = headers.get("x-ratelimit-reset")
        if reset and reset.isdigit():
            return max(0.0, float(int(reset) - time.time()) + 1.0)
        return max(60.0, self.config.backoff_seconds)

    def _cached(self, cache_file: Path) -> dict[str, Any] | None:
        if not self.config.cache_enabled or not cache_file.exists():
            return None
        return json.loads(cache_file.read_text(encoding="utf-8"))

    def get_json(self, endpoint: str, params: dict[str, Any] | None = None, *, cache_mode: CacheMode = "default") -> tuple[Any, dict[str, str], bool]:
        cache_file = self._cache_file(endpoint, params)
        cached = self._cached(cache_file) if cache_mode != "bypass" else None
        if cached is not None and cache_mode == "default":
            self.cache_hits += 1
            return cached["body"], cached.get("headers", {}), True
        request_headers: dict[str, str] = {}
        if cached is not None and cache_mode == "revalidate":
            previous = {str(key).lower(): str(value) for key, value in cached.get("headers", {}).items()}
            if previous.get("etag"):
                request_headers["If-None-Match"] = previous["etag"]
            if previous.get("last-modified"):
                request_headers["If-Modified-Since"] = previous["last-modified"]
        url = f"{self.config.api_base_url}{endpoint}"
        last_error = "unknown error"
        for attempt in range(self.config.max_retries + 1):
            response: httpx.Response | None = None
            retry_delay: float | None = None
            try:
                self.request_count += 1
                response = self.http.get(url, params=params, headers=request_headers)
                self._capture_rate_limit(response.headers)
                if response.status_code == 401:
                    raise GitHubAuthenticationError("GitHub rejected GITHUB_TOKEN (HTTP 401)")
                if response.status_code == 304 and cached is not None:
                    self.cache_hits += 1
                    self.cache_revalidations += 1
                    return cached["body"], cached.get("headers", {}), True
                if response.status_code in (403, 429):
                    retry_delay = self._rate_limit_wait(response.headers)
                    last_error = f"rate limited ({response.status_code})"
                elif response.status_code >= 500:
                    last_error = f"server error {response.status_code}"
                else:
                    response.raise_for_status()
                    body = response.json()
                    headers = dict(response.headers)
                    if self.config.cache_enabled and cache_mode != "bypass":
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(json.dumps({"body": body, "headers": headers, "cached_at": utc_now()}, ensure_ascii=False), encoding="utf-8")
                    return body, headers, False
            except GitHubAuthenticationError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = str(exc)
            if attempt < self.config.max_retries:
                self.retry_count += 1
                delay = retry_delay if retry_delay is not None else self.config.backoff_seconds * (2 ** attempt) + random.uniform(0, 0.1)
                if delay > self.config.rate_limit_max_wait_seconds:
                    break
                self.logger.warning("Request failed (%s); retrying in %.1fs", last_error, delay)
                self.sleeper(delay)
        raise GitHubRequestError(f"GET {url} failed after {self.config.max_retries + 1} attempts: {last_error}")

    def preflight_rate_limit(self) -> dict[str, Any]:
        body, _, _ = self.get_json("/rate_limit", cache_mode="bypass")
        resources = body.get("resources") or {}
        for bucket in ("core", "search"):
            details = resources.get(bucket)
            if isinstance(details, dict):
                self.rate_limits[bucket] = {f"x-ratelimit-{key}": str(value) for key, value in details.items()}
        return body


@dataclass(frozen=True)
class DiscoveryCandidate:
    repository_id: int
    full_name: str
    group: str
    best_rank: int
    matches: tuple[dict[str, Any], ...]


def _wrapper_hash(wrapper: dict[str, Any]) -> str:
    return stable_hash(wrapper)


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.name


def load_baseline_snapshots(run_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    repositories: dict[int, dict[str, Any]] = {}
    readmes_by_name: dict[str, dict[str, Any]] = {}
    for source_type in ("github_repository", "github_readme"):
        directory = run_dir / source_type
        if not directory.is_dir():
            if source_type == "github_repository":
                raise ValueError("baseline run has no github_repository directory")
            continue
        for path in sorted(directory.glob("*.json")):
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            required = {"collector_version", "schema_version", "run_id", "retrieved_at", "source_type", "source_id", "source_url", "source_hash", "payload"}
            if required - wrapper.keys() or wrapper["source_type"] != source_type:
                raise ValueError(f"invalid baseline wrapper: {path}")
            if wrapper["run_id"] != run_dir.name:
                raise ValueError(f"baseline wrapper run_id does not match its run directory: {path}")
            if stable_hash(wrapper["payload"]) != wrapper["source_hash"]:
                raise ValueError(f"baseline payload hash mismatch: {path}")
            wrapper["_wrapper_hash"] = _wrapper_hash(wrapper)
            if source_type == "github_readme":
                if wrapper["source_id"] in readmes_by_name:
                    raise ValueError(f"duplicate baseline README identity: {wrapper['source_id']}")
                readmes_by_name[wrapper["source_id"]] = wrapper
                continue
            repo_id = wrapper["payload"].get("id")
            full_name = wrapper["payload"].get("full_name")
            if not isinstance(repo_id, int) or full_name != wrapper["source_id"]:
                raise ValueError(f"baseline repository identity conflict: {path}")
            if repo_id in repositories:
                raise ValueError(f"duplicate baseline numeric repository ID: {repo_id}")
            repositories[repo_id] = wrapper
    readmes: dict[int, dict[str, Any]] = {}
    for repo_id, repository in repositories.items():
        readme = readmes_by_name.get(repository["source_id"])
        if readme is not None:
            readmes[repo_id] = readme
    unmatched = sorted(set(readmes_by_name) - {item["source_id"] for item in repositories.values()})
    if unmatched:
        raise ValueError(f"baseline README has no matching repository: {unmatched}")
    return repositories, readmes


def select_balanced_candidates(candidates: list[DiscoveryCandidate], slots: int, shares: dict[str, float], excluded_ids: set[int]) -> list[DiscoveryCandidate]:
    if slots <= 0:
        return []
    grouped: dict[str, list[DiscoveryCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.repository_id not in excluded_ids:
            grouped[candidate.group].append(candidate)
    for group in grouped:
        grouped[group].sort(key=lambda item: (item.best_rank, item.repository_id))
    selected: list[DiscoveryCandidate] = []
    selected_ids = set(excluded_ids)
    quotas = {group: math.floor(slots * share) for group, share in shares.items()}
    remainders = sorted(shares, key=lambda group: (-(slots * shares[group] - quotas[group]), group))
    for group in remainders[: slots - sum(quotas.values())]:
        quotas[group] += 1
    for group in shares:
        for candidate in grouped.get(group, []):
            if quotas[group] <= 0:
                break
            if candidate.repository_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.repository_id)
            quotas[group] -= 1
    remainder = sorted((item for values in grouped.values() for item in values if item.repository_id not in selected_ids), key=lambda item: (item.best_rank, item.repository_id))
    for candidate in remainder:
        if len(selected) >= slots:
            break
        selected.append(candidate)
        selected_ids.add(candidate.repository_id)
    return selected[:slots]


class GitHubCollector:
    def __init__(self, config: CollectorConfig, client: GitHubClient | None = None, *, baseline_run: Path | None = None) -> None:
        self.config = config
        self.run_id = uuid.uuid4().hex
        self.retrieved_at = utc_now()
        self.run_dir = config.raw_root / self.retrieved_at[:10] / self.run_id
        self.baseline_run = baseline_run.resolve() if baseline_run else None
        self.logger = logging.getLogger("veritrust_rec.github")
        self.client = client or GitHubClient(config, self.logger)
        self.summary: dict[str, Any] = {
            "dataset_purpose": config.dataset_purpose, "profile": config.profile_name,
            "run_id": self.run_id, "retrieved_at": self.retrieved_at,
            "target_repositories": config.max_total_repos, "repositories": 0,
            "readme_success": 0, "readme_failures": 0,
            "materialized_repositories": 0, "materialized_readmes": 0,
        }

    def _write_wrapper(self, wrapper: dict[str, Any]) -> None:
        source_type, source_id = wrapper["source_type"], wrapper["source_id"]
        safe_id = source_id.replace("/", "__").replace(":", "_")
        target = self.run_dir / source_type / f"{safe_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), encoding="utf-8")

    def _snapshot(self, source_type: str, source_id: str, source_url: str, body: Any, *, provenance: dict[str, Any] | None = None, retrieved_at: str | None = None) -> dict[str, Any]:
        wrapper = {
            "collector_version": self.config.collector_version, "schema_version": self.config.schema_version,
            "run_id": self.run_id, "retrieved_at": retrieved_at or utc_now(), "source_type": source_type,
            "source_id": source_id, "source_url": source_url, "source_hash": stable_hash(body),
            "provenance": provenance or {}, "payload": body,
        }
        self._write_wrapper(wrapper)
        return wrapper

    def _materialize(self, original: dict[str, Any]) -> dict[str, Any]:
        provenance = dict(original.get("provenance") or {})
        provenance["materialization"] = {
            "materialization_method": "baseline_snapshot_rewrap_v1", "materialized_at": utc_now(),
            "materialized_from_run_id": original["run_id"],
            "original_collector_version": original["collector_version"],
            "original_schema_version": original["schema_version"],
            "original_wrapper_hash": original["_wrapper_hash"],
        }
        return self._snapshot(original["source_type"], original["source_id"], original["source_url"], original["payload"], provenance=provenance, retrieved_at=original["retrieved_at"])

    def _record_failure(self, source_type: str, source_url: str, error: Exception) -> None:
        self.config.failed_fetches.parent.mkdir(parents=True, exist_ok=True)
        record = {"retrieved_at": utc_now(), "run_id": self.run_id, "source_type": source_type, "source_url": source_url, "error_type": type(error).__name__, "error": str(error)}
        with self.config.failed_fetches.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _fetch_readme(self, metadata: dict[str, Any], provenance: dict[str, Any]) -> bool:
        full_name, html_url = metadata["full_name"], metadata["html_url"]
        endpoint = f"/repos/{full_name}/readme"
        try:
            body, _, _ = self.client.get_json(endpoint, cache_mode="revalidate")
            text = ""
            if body.get("encoding") == "base64" and body.get("content"):
                text = base64.b64decode(body["content"].replace("\n", "")).decode("utf-8", errors="replace")
            self._snapshot("github_readme", full_name, body.get("html_url", html_url), {"api_response": body, "decoded_text": text}, provenance=provenance)
            self.summary["readme_success"] += 1
            return True
        except (GitHubRequestError, ValueError, UnicodeError) as exc:
            self.summary["readme_failures"] += 1
            self._record_failure("github_readme", f"{self.config.api_base_url}{endpoint}", exc)
            self.logger.warning("README fetch failed for %s: %s", full_name, exc)
            return False

    def _discover(self) -> list[DiscoveryCandidate]:
        by_id: dict[int, dict[str, Any]] = {}
        query_to_group = self.config.query_to_group or {query: "legacy" for query in self.config.queries}
        grouped_queries: dict[str, list[str]] = defaultdict(list)
        for query in self.config.queries:
            grouped_queries[query_to_group.get(query, "legacy")].append(query)
        configured_order = list((self.config.group_shares or {}).keys())
        group_order = [group for group in configured_order if group in grouped_queries]
        group_order.extend(group for group in grouped_queries if group not in group_order)
        queries = [
            grouped_queries[group][index]
            for index in range(max((len(grouped_queries[group]) for group in group_order), default=0))
            for group in group_order
            if index < len(grouped_queries[group])
        ]
        required_groups = set(grouped_queries)
        attempted_groups: set[str] = set()
        for query in queries:
            group = query_to_group.get(query, "legacy")
            attempted_groups.add(group)
            query_ids: set[int] = set()
            for page in range(1, 1000):
                params = {"q": query, "sort": self.config.sort, "order": self.config.order, "per_page": self.config.per_page, "page": page}
                try:
                    search, _, _ = self.client.get_json("/search/repositories", params, cache_mode="bypass")
                except GitHubRequestError as exc:
                    self._record_failure("github_search", f"{self.config.api_base_url}/search/repositories", exc)
                    self.logger.error("Search query failed for %s: %s", query, exc)
                    break
                items = search.get("items", [])
                positions = [{"repository_id": item.get("id"), "full_name": item.get("full_name"), "search_position": position} for position, item in enumerate(items, start=1)]
                self._snapshot("github_search", f"{stable_hash(params)[:16]}_p{page}", f"{self.config.api_base_url}/search/repositories", search, provenance={"query_group": group, "query_used": query, "search_page": page, "result_positions": positions})
                for position, item in enumerate(items, start=1):
                    repo_id, full_name = item.get("id"), item.get("full_name")
                    if not isinstance(repo_id, int) or not isinstance(full_name, str):
                        continue
                    if repo_id not in query_ids and len(query_ids) >= self.config.max_repos_per_query:
                        continue
                    query_ids.add(repo_id)
                    match = {"query_group": group, "query_used": query, "search_page": page, "search_position": position}
                    existing = by_id.setdefault(repo_id, {"full_name": full_name, "matches": []})
                    existing["matches"].append(match)
                if len(by_id) >= self.config.discovery_cap or len(query_ids) >= self.config.max_repos_per_query or not items or len(items) < self.config.per_page:
                    break
            if len(by_id) >= self.config.discovery_cap and attempted_groups >= required_groups:
                break
        candidates: list[DiscoveryCandidate] = []
        for repo_id, item in by_id.items():
            matches = sorted(item["matches"], key=lambda value: (value["search_page"], value["search_position"], value["query_used"]))
            primary = matches[0]
            candidates.append(DiscoveryCandidate(repo_id, item["full_name"], primary["query_group"], (primary["search_page"] - 1) * self.config.per_page + primary["search_position"], tuple(matches)))
        return sorted(candidates, key=lambda item: (item.best_rank, item.repository_id))

    def run(self) -> dict[str, Any]:
        repositories: dict[int, dict[str, Any]] = {}
        readme_ids: set[int] = set()
        baseline_ids: list[int] = []
        try:
            if self.config.preflight_rate_limit:
                self.client.preflight_rate_limit()
            if self.baseline_run:
                baseline_repositories, baseline_readmes = load_baseline_snapshots(self.baseline_run)
                if len(baseline_repositories) > self.config.max_total_repos:
                    raise ValueError("baseline contains more repositories than the selected profile target")
                for repo_id in sorted(baseline_repositories):
                    repositories[repo_id] = self._materialize(baseline_repositories[repo_id])
                    self.summary["materialized_repositories"] += 1
                    if repo_id in baseline_readmes:
                        self._materialize(baseline_readmes[repo_id])
                        readme_ids.add(repo_id)
                        self.summary["materialized_readmes"] += 1
                baseline_ids = sorted(repositories)
            candidates = self._discover()
            selected = select_balanced_candidates(candidates, self.config.max_total_repos - len(repositories), self.config.group_shares or {"legacy": 1.0}, set(repositories))
            selected_ids = {item.repository_id for item in selected}
            fallback = sorted(
                (item for item in candidates if item.repository_id not in selected_ids and item.repository_id not in repositories),
                key=lambda item: (item.best_rank, item.repository_id),
            )
            for candidate in [*selected, *fallback]:
                if len(repositories) >= self.config.max_total_repos:
                    break
                endpoint = f"/repos/{candidate.full_name}"
                try:
                    metadata, _, _ = self.client.get_json(endpoint, cache_mode="revalidate")
                    if metadata.get("id") != candidate.repository_id or not isinstance(metadata.get("full_name"), str):
                        raise ValueError("hydrated metadata identity conflicts with discovery numeric ID")
                    provenance = {**candidate.matches[0], "discovery_matches": list(candidate.matches), "selection_method": "balanced_query_groups_v1"}
                    wrapper = self._snapshot("github_repository", metadata["full_name"], metadata["html_url"], metadata, provenance=provenance)
                    repositories[candidate.repository_id] = wrapper
                    if self.config.fetch_readme and self._fetch_readme(metadata, provenance):
                        readme_ids.add(candidate.repository_id)
                except (GitHubRequestError, ValueError, KeyError) as exc:
                    self._record_failure("github_repository", f"{self.config.api_base_url}{endpoint}", exc)
                    self.logger.warning("Repository hydration failed for %s: %s", candidate.repository_id, exc)
        finally:
            self.client.close()
        stage_wrappers = [
            json.loads(path.read_text(encoding="utf-8"))
            for source_type in ("github_repository", "github_readme")
            for path in sorted((self.run_dir / source_type).glob("*.json"))
        ]
        self.summary.update({
            "repositories": len(repositories), "readme_coverage": round(len(readme_ids) / len(repositories), 6) if repositories else 0.0,
            "fetched_repositories": len(repositories) - self.summary["materialized_repositories"],
            "available_readmes": len(readme_ids),
            "baseline_run": project_relative(self.baseline_run) if self.baseline_run else None,
            "baseline_repository_ids": baseline_ids, "repository_ids": sorted(repositories),
            "source_hashes": sorted(wrapper["source_hash"] for wrapper in stage_wrappers),
            "requests": self.client.request_count, "cache_hits": self.client.cache_hits,
            "cache_revalidations": self.client.cache_revalidations, "retries": self.client.retry_count,
            "rate_limits": self.client.rate_limits, "unique_repositories_seen": len(repositories),
            "unique_repositories_discovered": len(candidates),
            "query_group_selection_distribution": dict(sorted(Counter(
                (wrapper.get("provenance") or {}).get("query_group", "unknown")
                for wrapper in repositories.values()
            ).items())),
            "query_group_discovery_distribution": dict(sorted(Counter(
                match["query_group"]
                for candidate in candidates
                for match in {item["query_group"]: item for item in candidate.matches}.values()
            ).items())),
            "collector_version": self.config.collector_version, "schema_version": self.config.schema_version,
            "self_contained": True,
        })
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "validation_summary.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.summary


def configure_logging(config: CollectorConfig) -> None:
    config.logs_root.mkdir(parents=True, exist_ok=True)
    log_file = config.logs_root / f"collector_{datetime.now(UTC):%Y%m%d}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to a GitHub collector YAML configuration.")
    parser.add_argument("--profile", default="baseline_20", help="Collector profile from configs/github.yaml.")
    parser.add_argument("--baseline-run", type=Path, help="Prior self-contained raw run to materialize into this stage.")
    args = parser.parse_args()
    load_dotenv(project_root() / ".env")
    config = load_config(args.config, args.profile)
    configure_logging(config)
    summary = GitHubCollector(config, baseline_run=args.baseline_run).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
