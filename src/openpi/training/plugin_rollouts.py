"""Validated real-rollout inputs for plugin handoff and call losses.

The manifest is deliberately the only source of text and provenance.  NPZ
payloads contain current observations, plus a successful continuation action
chunk for handoff records.  Pickled arrays and undeclared payload keys are not
accepted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np

SUITE_ORDER = ("spatial", "object", "goal", "long")
POLICY_IDS = ("base", *SUITE_ORDER)
SPLITS = ("train", "validation")
FEATURE_SCHEMA = "libero_rgb_state_v1"
OBSERVATION_KEYS = (
    "observation/image",
    "observation/wrist_image",
    "observation/state",
)
_HANDOFF_KEYS = frozenset((*OBSERVATION_KEYS, "actions"))
_CALL_KEYS = frozenset(OBSERVATION_KEYS)
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_CONTINUATION_SOURCES = frozenset(("successful_rollout", "verified_expert_recovery"))
_CALL_TERMINATIONS = frozenset(("success", "budget_exhausted", "terminal_failure"))
_CENSORED_TERMINATIONS = frozenset(("interrupted", "infra_error"))


class RolloutManifestError(ValueError):
    """Raised when rollout metadata or a bound sample violates the contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class HandoffRecord:
    kind: Literal["handoff"]
    record_id: str
    root_episode_id: str
    split: Literal["train", "validation"]
    source_policy_id: str
    target_policy_id: str
    policy_bundle_id: str
    budget_steps: int
    provenance: Literal["simulator_rollout"]
    rollout_id: str
    source_step: int
    sample_path: str
    sample_sha256: str
    prompt: str
    action_schema: Literal["libero_raw_delta7"]
    action_reference: Literal["current_observation"]
    continuation_success: Literal[True]
    continuation_target_source: str


@dataclasses.dataclass(frozen=True, slots=True)
class CallRecord:
    kind: Literal["call"]
    record_id: str
    root_episode_id: str
    split: Literal["train", "validation"]
    source_policy_id: str
    target_policy_id: str
    policy_bundle_id: str
    budget_steps: int
    provenance: Literal["simulator_rollout"]
    rollout_id: str
    source_step: int
    sample_path: str
    sample_sha256: str
    prompt: str
    attempted_policy_id: str
    success: bool
    executed_steps: int
    termination: str
    execution_scope: Literal["single_policy_until_terminal_or_budget"]
    policy_switches: Literal[0]


RolloutRecord: TypeAlias = HandoffRecord | CallRecord


def _fail(message: str, *, record_id: str | None = None) -> RolloutManifestError:
    prefix = "rollout manifest"
    if record_id is not None:
        prefix += f" record {record_id!r}"
    return RolloutManifestError(f"{prefix}: {message}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise _fail(f"{context} keys differ; missing={missing}, extra={extra}")


def _nonempty_string(value: Any, name: str, *, record_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{name} must be a nonempty string", record_id=record_id)
    return value


def _plain_int(value: Any, name: str, *, minimum: int, record_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{name} must be an integer >= {minimum}", record_id=record_id)
    return value


def _sha256_string(value: Any, name: str, *, record_id: str | None = None) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a lowercase SHA-256 hex digest", record_id=record_id)
    return value


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_file(root: pathlib.Path, relative: Any, *, record_id: str) -> tuple[str, pathlib.Path]:
    relative = _nonempty_string(relative, "sample_path", record_id=record_id)
    raw = pathlib.PurePath(relative)
    if raw.is_absolute() or raw.suffix != ".npz":
        raise _fail("sample_path must be a relative .npz path", record_id=record_id)
    try:
        resolved = (root / raw).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise _fail("sample_path is missing or escapes the manifest directory", record_id=record_id) from error
    if not resolved.is_file():
        raise _fail("sample_path is not a regular file", record_id=record_id)
    return relative, resolved


def _validate_array(name: str, array: np.ndarray, *, record_id: str) -> None:
    if name in ("observation/image", "observation/wrist_image"):
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise _fail(f"{name} must be uint8 HWC with exactly 3 channels", record_id=record_id)
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise _fail(f"{name} spatial dimensions must be positive", record_id=record_id)
        return
    expected_shape = (8,) if name == "observation/state" else (10, 7)
    if array.dtype != np.float32 or array.shape != expected_shape:
        raise _fail(
            f"{name} must be float32 with shape {expected_shape}, got {array.dtype} {array.shape}",
            record_id=record_id,
        )
    if not np.isfinite(array).all():
        raise _fail(f"{name} contains non-finite values", record_id=record_id)


def _load_npz(path: pathlib.Path, *, kind: str, record_id: str) -> dict[str, np.ndarray]:
    expected = _HANDOFF_KEYS if kind == "handoff" else _CALL_KEYS
    try:
        with np.load(path, allow_pickle=False) as payload:
            actual = frozenset(payload.files)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise _fail(
                    f"NPZ keys differ; missing={missing}, extra={extra}", record_id=record_id
                )
            arrays = {name: np.asarray(payload[name]) for name in sorted(expected)}
    except RolloutManifestError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise _fail("NPZ is unreadable or contains a pickled/object array", record_id=record_id) from error
    for name, array in arrays.items():
        _validate_array(name, array, record_id=record_id)
    return arrays


def _validate_common(record: Mapping[str, Any]) -> dict[str, Any]:
    record_id = _nonempty_string(record.get("record_id"), "record_id")
    split = record.get("split")
    if split not in SPLITS:
        raise _fail("split must be 'train' or 'validation' (official_eval is forbidden)", record_id=record_id)
    source_policy_id = record.get("source_policy_id")
    target_policy_id = record.get("target_policy_id")
    if source_policy_id not in POLICY_IDS or target_policy_id not in POLICY_IDS:
        raise _fail(f"policy IDs must be in {POLICY_IDS}", record_id=record_id)
    if record.get("provenance") != "simulator_rollout":
        raise _fail("provenance must be 'simulator_rollout'", record_id=record_id)
    return {
        "record_id": record_id,
        "root_episode_id": _nonempty_string(
            record.get("root_episode_id"), "root_episode_id", record_id=record_id
        ),
        "split": split,
        "source_policy_id": source_policy_id,
        "target_policy_id": target_policy_id,
        "policy_bundle_id": _nonempty_string(
            record.get("policy_bundle_id"), "policy_bundle_id", record_id=record_id
        ),
        "budget_steps": _plain_int(
            record.get("budget_steps"), "budget_steps", minimum=1, record_id=record_id
        ),
        "provenance": "simulator_rollout",
        "rollout_id": _nonempty_string(record.get("rollout_id"), "rollout_id", record_id=record_id),
        "source_step": _plain_int(
            record.get("source_step"), "source_step", minimum=0, record_id=record_id
        ),
        "prompt": _nonempty_string(record.get("prompt"), "prompt", record_id=record_id),
    }


_COMMON_KEYS = frozenset(
    (
        "kind",
        "record_id",
        "root_episode_id",
        "split",
        "source_policy_id",
        "target_policy_id",
        "policy_bundle_id",
        "budget_steps",
        "provenance",
        "rollout_id",
        "source_step",
        "sample_path",
        "sample_sha256",
        "prompt",
    )
)
_HANDOFF_RECORD_KEYS = _COMMON_KEYS | frozenset(
    ("action_schema", "action_reference", "continuation_success", "continuation_target_source")
)
_CALL_RECORD_KEYS = _COMMON_KEYS | frozenset(
    (
        "attempted_policy_id",
        "success",
        "executed_steps",
        "termination",
        "execution_scope",
        "policy_switches",
    )
)


def _parse_record(record: Any, root: pathlib.Path) -> RolloutRecord:
    if not isinstance(record, dict):
        raise _fail("each metadata.records entry must be an object")
    kind = record.get("kind")
    expected_keys = _HANDOFF_RECORD_KEYS if kind == "handoff" else _CALL_RECORD_KEYS
    if kind not in ("handoff", "call"):
        raise _fail("record kind must be 'handoff' or 'call'", record_id=record.get("record_id"))
    record_id = record.get("record_id") if isinstance(record.get("record_id"), str) else None
    actual_keys = frozenset(record)
    if actual_keys != expected_keys:
        raise _fail(
            f"record keys differ; missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}",
            record_id=record_id,
        )
    common = _validate_common(record)
    record_id = common["record_id"]
    sample_path, resolved_path = _sample_file(root, record["sample_path"], record_id=record_id)
    sample_sha256 = _sha256_string(record["sample_sha256"], "sample_sha256", record_id=record_id)
    if _file_sha256(resolved_path) != sample_sha256:
        raise _fail("sample_sha256 does not match the NPZ file", record_id=record_id)
    _load_npz(resolved_path, kind=kind, record_id=record_id)
    common.update(sample_path=sample_path, sample_sha256=sample_sha256)

    if kind == "handoff":
        if common["target_policy_id"] == "base" or common["target_policy_id"] == common["source_policy_id"]:
            raise _fail(
                "handoff target must be one of the four adapters and differ from source",
                record_id=record_id,
            )
        if record["continuation_success"] is not True:
            raise _fail("handoff continuation_success must be true", record_id=record_id)
        if record["action_schema"] != "libero_raw_delta7":
            raise _fail("action_schema must be 'libero_raw_delta7'", record_id=record_id)
        if record["action_reference"] != "current_observation":
            raise _fail("action_reference must be 'current_observation'", record_id=record_id)
        continuation_source = record["continuation_target_source"]
        if continuation_source not in _CONTINUATION_SOURCES:
            raise _fail(
                f"continuation_target_source must be one of {sorted(_CONTINUATION_SOURCES)}",
                record_id=record_id,
            )
        return HandoffRecord(
            kind="handoff",
            **common,
            action_schema="libero_raw_delta7",
            action_reference="current_observation",
            continuation_success=True,
            continuation_target_source=continuation_source,
        )

    attempted = record["attempted_policy_id"]
    if attempted not in POLICY_IDS:
        raise _fail(f"attempted_policy_id must be in {POLICY_IDS}", record_id=record_id)
    success = record["success"]
    if not isinstance(success, bool):
        raise _fail("success must be a boolean", record_id=record_id)
    executed = _plain_int(record["executed_steps"], "executed_steps", minimum=0, record_id=record_id)
    if executed > common["budget_steps"]:
        raise _fail("executed_steps exceeds budget_steps", record_id=record_id)
    termination = record["termination"]
    if termination in _CENSORED_TERMINATIONS:
        raise _fail(
            "interrupted/infra_error calls are censored and belong only in exclusion counts",
            record_id=record_id,
        )
    if termination not in _CALL_TERMINATIONS:
        raise _fail(f"unsupported call termination {termination!r}", record_id=record_id)
    if success != (termination == "success"):
        raise _fail("success must be true exactly when termination is 'success'", record_id=record_id)
    if termination == "budget_exhausted" and executed != common["budget_steps"]:
        raise _fail("budget_exhausted requires executed_steps == budget_steps", record_id=record_id)
    if record["execution_scope"] != "single_policy_until_terminal_or_budget":
        raise _fail(
            "execution_scope must be 'single_policy_until_terminal_or_budget'",
            record_id=record_id,
        )
    if isinstance(record["policy_switches"], bool) or record["policy_switches"] != 0:
        raise _fail("policy_switches must be the integer 0", record_id=record_id)
    return CallRecord(
        kind="call",
        **common,
        attempted_policy_id=attempted,
        success=success,
        executed_steps=executed,
        termination=termination,
        execution_scope="single_policy_until_terminal_or_budget",
        policy_switches=0,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutStore:
    """Immutable index of a fully validated unified rollout manifest."""

    manifest_path: pathlib.Path
    source_base_inventory_sha256: str
    norm_stats_sha256: str
    records: tuple[RolloutRecord, ...]
    excluded_call_counts: Mapping[str, int]
    manifest_sha256: str

    def handoff_records(self, target_policy_id: str, split: str) -> tuple[HandoffRecord, ...]:
        if target_policy_id not in SUITE_ORDER:
            raise ValueError(f"target_policy_id must be one of {SUITE_ORDER}")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return tuple(
            record
            for record in self.records
            if isinstance(record, HandoffRecord)
            and record.target_policy_id == target_policy_id
            and record.split == split
        )

    def call_records(self, split: str) -> tuple[CallRecord, ...]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return tuple(
            record
            for record in self.records
            if isinstance(record, CallRecord) and record.split == split
        )

    def raw_sample(self, record: RolloutRecord) -> dict[str, np.ndarray | str]:
        """Re-verify and return one raw LiberoInputs-compatible sample."""
        if record not in self.records:
            raise ValueError("record does not belong to this rollout store")
        path = (self.manifest_path.parent / record.sample_path).resolve(strict=True)
        try:
            path.relative_to(self.manifest_path.parent)
        except ValueError as error:
            raise _fail("sample_path escapes the manifest directory", record_id=record.record_id) from error
        if _file_sha256(path) != record.sample_sha256:
            raise _fail("sample changed after manifest validation", record_id=record.record_id)
        sample: dict[str, np.ndarray | str] = _load_npz(
            path, kind=record.kind, record_id=record.record_id
        )
        sample["prompt"] = record.prompt
        return sample

    def call_targets(
        self,
        records: Sequence[CallRecord],
        policy_order: Sequence[str] = POLICY_IDS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return binary labels and observed masks without inventing negatives.

        Only the attempted policy is observed for each row.  Values under a
        false mask are padding and must not contribute to a loss.
        """
        ordered_policies = tuple(policy_order)
        if not ordered_policies or len(ordered_policies) != len(set(ordered_policies)):
            raise ValueError("policy_order must contain distinct policy IDs")
        if any(policy_id not in POLICY_IDS for policy_id in ordered_policies):
            raise ValueError(f"policy_order entries must be in {POLICY_IDS}")
        selected = tuple(records)
        if any(not isinstance(record, CallRecord) or record not in self.records for record in selected):
            raise ValueError("all call target records must belong to this rollout store")
        column_by_policy = {policy_id: index for index, policy_id in enumerate(ordered_policies)}
        labels = np.zeros((len(selected), len(ordered_policies)), dtype=np.float32)
        observed = np.zeros((len(selected), len(ordered_policies)), dtype=np.bool_)
        for row, record in enumerate(selected):
            try:
                column = column_by_policy[record.attempted_policy_id]
            except KeyError as error:
                raise ValueError(
                    f"policy_order omits attempted policy {record.attempted_policy_id!r}"
                ) from error
            labels[row, column] = np.float32(record.success)
            observed[row, column] = True
        return labels, observed

    def select_batch(
        self,
        records: Sequence[RolloutRecord],
        batch_size: int,
        *,
        seed: int,
        step: int = 0,
    ) -> tuple[RolloutRecord, ...]:
        """Choose a reproducible batch, cycling only when the batch is oversized."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        candidates = tuple(records)
        if not candidates:
            raise ValueError("cannot select a batch from no records")
        if any(record not in self.records for record in candidates):
            raise ValueError("all candidate records must belong to this rollout store")
        ordered = sorted(
            candidates,
            key=lambda record: hashlib.sha256(
                f"{seed}:{step}:{record.record_id}".encode()
            ).digest(),
        )
        return tuple(ordered[index % len(ordered)] for index in range(batch_size))

    def summary(self) -> dict[str, Any]:
        kind_counts = Counter(record.kind for record in self.records)
        split_counts = Counter(record.split for record in self.records)
        call_reasons = Counter(
            record.termination for record in self.records if isinstance(record, CallRecord)
        )
        return {
            "source": "real_rollouts",
            "feature_schema": FEATURE_SCHEMA,
            "suite_order": list(SUITE_ORDER),
            "manifest_sha256": self.manifest_sha256,
            "source_base_inventory_sha256": self.source_base_inventory_sha256,
            "norm_stats_sha256": self.norm_stats_sha256,
            "record_counts": {
                "total": len(self.records),
                "handoff": kind_counts["handoff"],
                "call": kind_counts["call"],
                "train": split_counts["train"],
                "validation": split_counts["validation"],
            },
            "call_termination_counts": {
                reason: call_reasons[reason] for reason in sorted(_CALL_TERMINATIONS)
            },
            "excluded_call_counts": dict(self.excluded_call_counts),
            "sample_sha256_by_record": {
                record.record_id: record.sample_sha256 for record in self.records
            },
            "policy_bundle_ids": sorted({record.policy_bundle_id for record in self.records}),
        }


_TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "source",
        "suite_order",
        "feature_schema",
        "source_base_inventory_sha256",
        "norm_stats_sha256",
        "metadata",
    )
)
_METADATA_KEYS = frozenset(("records", "excluded_call_counts"))


def load_rollout_manifest(path: str | pathlib.Path) -> RolloutStore:
    """Load and eagerly validate a schema-v1 unified real-rollout manifest.

    ``path`` must name the JSON file itself.  A missing file is an error; callers
    that make this feature optional must disable it before invoking this loader.
    """
    manifest_path = pathlib.Path(path).expanduser().resolve(strict=True)
    if not manifest_path.is_file():
        raise RolloutManifestError(f"rollout manifest is not a file: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes, object_pairs_hook=_json_object)
    except RolloutManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RolloutManifestError(f"invalid rollout manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise _fail("top level must be an object")
    _require_exact_keys(manifest, _TOP_LEVEL_KEYS, context="top-level")
    if manifest["schema_version"] != 1:
        raise _fail("schema_version must be 1")
    if manifest["source"] != "real_rollouts":
        raise _fail("source must be 'real_rollouts'")
    if manifest["suite_order"] != list(SUITE_ORDER):
        raise _fail(f"suite_order must be {list(SUITE_ORDER)}")
    if manifest["feature_schema"] != FEATURE_SCHEMA:
        raise _fail(f"feature_schema must be {FEATURE_SCHEMA!r}")
    source_base_hash = _sha256_string(
        manifest["source_base_inventory_sha256"], "source_base_inventory_sha256"
    )
    norm_stats_hash = _sha256_string(manifest["norm_stats_sha256"], "norm_stats_sha256")
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict):
        raise _fail("metadata must be an object")
    _require_exact_keys(metadata, _METADATA_KEYS, context="metadata")

    excluded = metadata["excluded_call_counts"]
    if not isinstance(excluded, dict) or frozenset(excluded) != _CENSORED_TERMINATIONS:
        raise _fail(
            "metadata.excluded_call_counts must contain exactly interrupted and infra_error"
        )
    for reason, count in excluded.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise _fail(f"excluded call count {reason!r} must be a non-negative integer")

    raw_records = metadata["records"]
    if not isinstance(raw_records, list):
        raise _fail("metadata.records must be a list")
    records = tuple(_parse_record(record, manifest_path.parent) for record in raw_records)
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        duplicates = sorted(record_id for record_id, count in Counter(record_ids).items() if count > 1)
        raise _fail(f"record_id values must be unique; duplicates={duplicates}")
    episode_splits: dict[str, str] = {}
    for record in records:
        prior = episode_splits.setdefault(record.root_episode_id, record.split)
        if prior != record.split:
            raise _fail(
                f"root episode {record.root_episode_id!r} crosses train/validation ({prior}, {record.split})"
            )

    return RolloutStore(
        manifest_path=manifest_path,
        source_base_inventory_sha256=source_base_hash,
        norm_stats_sha256=norm_stats_hash,
        records=records,
        excluded_call_counts=MappingProxyType(dict(excluded)),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
