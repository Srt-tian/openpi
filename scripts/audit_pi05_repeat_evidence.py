#!/usr/bin/env python3
"""Read-only integrity audit for the frozen 280-episode PI0.5 legacy repeat."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAPS = {"libero_spatial": 220, "libero_object": 280,
        "libero_goal": 300, "libero_10": 520}
PLUGIN = {"libero_spatial": "spatial", "libero_object": "object",
          "libero_goal": "goal", "libero_10": "long"}
CASES = {
    ("libero_10", 8, 0), ("libero_10", 8, 2), ("libero_10", 8, 3),
    ("libero_10", 8, 5), ("libero_10", 8, 8), ("libero_10", 9, 3),
    ("libero_spatial", 3, 6), ("libero_spatial", 7, 2),
    ("libero_spatial", 5, 2), ("libero_goal", 3, 4),
    ("libero_object", 4, 1), ("libero_object", 2, 7),
    ("libero_object", 5, 9), ("libero_goal", 9, 5),
}
EXPECTED_COMMIT = "98595c32938490526719c2348f32c6a73afdab24"
CALL_STRIDE = 1000003
REP_STRIDE = 1000000007


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint(suite: str, task: int) -> int:
    return SUITES.index(suite) * 10 + task


def identity(case: dict) -> tuple:
    return (case.get("suite"), case.get("task_id"), case.get("init_id"),
            case.get("replicate_id"), case.get("policy_id"))


class Audit:
    def __init__(self):
        self.errors: list[str] = []
        self.episodes: dict[tuple, dict] = {}
        self.batch_count = 0
        self.summary_successes = Counter()

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def case_protocol(self, case: dict, where: str) -> None:
        suite, task, init, rep, policy = identity(case)
        self.require((suite, task, init) in CASES, f"{where}: case outside frozen union")
        self.require(type(rep) is int and 0 <= rep <= 9, f"{where}: invalid replicate")
        self.require(policy in ("base", PLUGIN.get(suite)), f"{where}: invalid arm policy")
        number = joint(suite, task) if suite in SUITES and type(task) is int else -1
        ambient = 7 + number * 50 + init if type(init) is int else -1
        self.require(case.get("joint_task_number") == number, f"{where}: joint task mismatch")
        self.require(case.get("ambient_seed") == ambient, f"{where}: ambient seed mismatch")
        self.require(case.get("policy_seed") == ambient + rep * REP_STRIDE,
                     f"{where}: policy episode seed mismatch")
        self.require(case.get("id") == f"{suite}/{task}/{init}/{rep}", f"{where}: id mismatch")

    def audit_episode(self, path: Path, expected_case: dict, manifest_identity: dict,
                      summary_row: dict) -> None:
        try:
            value = json.loads(path.read_text())
        except Exception as exc:
            self.errors.append(f"{path}: unreadable episode ({type(exc).__name__})")
            return
        case = value.get("case", {})
        key = identity(case)
        self.case_protocol(case, str(path))
        self.require(case == expected_case == {k: summary_row[k] for k in expected_case},
                     f"{path}: manifest/summary/episode case mismatch")
        self.require(value.get("execution_backend") == "legacy", f"{path}: backend not legacy")
        legacy = value.get("legacy", {})
        self.require(legacy.get("service_metadata") == manifest_identity,
                     f"{path}: actual service identity mismatch")
        trace, payloads = legacy.get("trace"), legacy.get("payload_hashes")
        steps, calls = legacy.get("steps"), legacy.get("inference_calls")
        cap = CAPS.get(case.get("suite"), -1)
        self.require(isinstance(trace, list) and type(steps) is int and len(trace) == steps,
                     f"{path}: trace/steps mismatch")
        self.require(type(steps) is int and 0 < steps <= cap, f"{path}: step cap violation")
        success = legacy.get("success")
        self.require(type(success) is bool and summary_row.get("success") is success,
                     f"{path}: summary/episode success mismatch")
        expected_status = "success" if success else "failure"
        self.require(legacy.get("status") == summary_row.get("status") == expected_status,
                     f"{path}: status inconsistent with success")
        if isinstance(trace, list) and trace:
            self.require([row.get("step") for row in trace] == list(range(steps)),
                         f"{path}: nonconsecutive trace steps")
            done = [row.get("done") is True for row in trace]
            self.require(not any(done[:-1]), f"{path}: done before final executed step")
            self.require(done[-1] is success, f"{path}: final done does not match success")
            self.require(success or steps == cap, f"{path}: failure stopped before cap")
            for row in trace:
                call = row.get("inference_call")
                self.require(call == row.get("step") // 5 if type(row.get("step")) is int else False,
                             f"{path}: trace inference_call is not step//5")
                expected_seed = case.get("policy_seed") + call * CALL_STRIDE if type(call) is int else None
                self.require(row.get("policy_seed") == expected_seed,
                             f"{path}: trace call seed mismatch")
        self.require(isinstance(payloads, list) and type(calls) is int and len(payloads) == calls,
                     f"{path}: payload/inference-call count mismatch")
        if isinstance(payloads, list):
            self.require([row.get("inference_call") for row in payloads] == list(range(len(payloads))),
                         f"{path}: payload calls nonconsecutive")
            for row in payloads:
                call = row.get("inference_call")
                expected_seed = case.get("policy_seed") + call * CALL_STRIDE if type(call) is int else None
                self.require(row.get("policy_seed") == expected_seed,
                             f"{path}: payload call seed mismatch")
                self.require(row.get("policy_id") == case.get("policy_id") and row.get("status") == "ok",
                             f"{path}: payload policy/status mismatch")
                for name in ("observation_image", "observation_wrist_image",
                             "observation_state", "response_actions"):
                    descriptor = row.get(name, {})
                    hexdigest = descriptor.get("sha256")
                    self.require(isinstance(hexdigest, str) and len(hexdigest) == 64
                                 and all(c in "0123456789abcdef" for c in hexdigest),
                                 f"{path}: malformed {name} hash")
        if key in self.episodes:
            self.errors.append(f"duplicate episode identity: {key}")
        else:
            self.episodes[key] = summary_row
            self.summary_successes[case.get("policy_id")] += int(success is True)

    def audit_batch(self, manifest_path: Path, checkpoint: dict | None) -> None:
        root = manifest_path.parent
        summary_path = root / "summary.json"
        self.batch_count += 1
        try:
            manifest = json.loads(manifest_path.read_text())
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            self.errors.append(f"{root}: missing/unreadable manifest or summary ({type(exc).__name__})")
            return
        policy = manifest.get("fixed_policy_id")
        cases = manifest.get("cases", [])
        self.require(manifest.get("schema") == "pi05_harness_eval.manifest.v1",
                     f"{root}: manifest schema")
        self.require(manifest.get("mode") == manifest.get("execution_backend") == "legacy",
                     f"{root}: expected legacy mode/backend")
        self.require(manifest.get("replicate_protocol") == "fixed_case_replicate_0_9_stride_1000000007_v1"
                     and manifest.get("replicate_id_range") == list(range(10)),
                     f"{root}: replicate protocol mismatch")
        self.require(manifest.get("policy_episode_seed_formula") ==
                     "ambient_seed + replicate_id * 1000000007"
                     and manifest.get("inference_seed_formula") ==
                     "policy_episode_seed + inference_call * 1000003",
                     f"{root}: seed formulas mismatch")
        verified = manifest.get("verified_service_identity", {})
        self.require(verified.get("policy_id") == policy, f"{root}: fixed/verified policy mismatch")
        self.require(verified.get("policy_seed_protocol") ==
                     "paired_episode_plus_call_1000003_numpy_pcg64_noise_10x32_f32_v1",
                     f"{root}: model seed protocol mismatch")
        if policy == "base":
            self.require(verified.get("base_graph") == "original_pi05_libero"
                         and verified.get("adapter_sha256") is None,
                         f"{root}: base graph/adapter identity mismatch")
        else:
            self.require(verified.get("base_graph") == "pi05_lora",
                         f"{root}: plugin base_graph is not pi05_lora")
        if checkpoint is not None:
            self.require(verified.get("checkpoint_sha256") == checkpoint["sha256"],
                         f"{root}: checkpoint manifest hash mismatch")
            expected_adapter = None if policy == "base" else checkpoint["adapters"].get(policy)
            self.require(verified.get("adapter_sha256") == expected_adapter,
                         f"{root}: verified adapter differs from checkpoint bank")
        self.require(summary.get("schema") == "pi05_harness_eval.summary.v1"
                     and summary.get("mode") == "legacy", f"{root}: summary schema/mode")
        rows = summary.get("cases", [])
        self.require(summary.get("complete") is True and summary.get("errors") == 0
                     and summary.get("planned") == summary.get("completed") == len(rows) == len(cases),
                     f"{root}: batch incomplete/errors/count mismatch")
        self.require(summary.get("successes") == sum(row.get("success") is True for row in rows),
                     f"{root}: summary success total inaccurate")
        manifest_cases = {identity(case): case for case in cases}
        self.require(len(manifest_cases) == len(cases), f"{root}: duplicate manifest cases")
        for case in cases:
            self.case_protocol(case, str(manifest_path))
            self.require(case.get("policy_id") == policy, f"{root}: mixed policy batch")
        for row in rows:
            key = identity(row)
            expected = manifest_cases.get(key)
            relative = row.get("episode")
            if expected is None or not isinstance(relative, str):
                self.errors.append(f"{root}: summary row lacks manifest case/episode")
                continue
            episode_path = (root / relative).resolve()
            if not episode_path.is_relative_to(root.resolve()):
                self.errors.append(f"{root}: episode path escapes batch")
                continue
            self.audit_episode(episode_path, expected, verified, row)


def load_checkpoint(path: Path) -> dict:
    value = json.loads(path.read_text())
    banks = value.get("banks")
    if not isinstance(banks, dict):
        raise ValueError("checkpoint manifest requires banks")
    adapters = {}
    for policy in ("spatial", "object", "goal", "long"):
        adapter = banks.get(policy, {}).get("adapter_sha256")
        if not (isinstance(adapter, str) and len(adapter) == 64
                and all(c in "0123456789abcdef" for c in adapter)):
            raise ValueError(f"invalid checkpoint adapter for {policy}")
        adapters[policy] = adapter
    return {"sha256": sha256(path), "adapters": adapters}


def self_test() -> int:
    """Focused negative checks for identity, call mapping, and final done."""
    case = {"suite": "libero_10", "task_id": 8, "init_id": 0, "replicate_id": 0,
            "policy_id": "long", "joint_task_number": 38, "ambient_seed": 1907,
            "policy_seed": 1907, "id": "libero_10/8/0/0"}
    identity_value = {"policy_id": "long", "base_graph": "pi05_lora",
                      "adapter_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                      "policy_seed_protocol": "paired_episode_plus_call_1000003_numpy_pcg64_noise_10x32_f32_v1"}
    descriptor = {"dtype": "float64", "shape": [1], "sha256": "c" * 64}

    def evidence(call: int = 0, done: bool = True) -> dict:
        return {"case": case, "execution_backend": "legacy", "legacy": {
            "service_metadata": identity_value, "steps": 1, "inference_calls": 1,
            "success": True, "status": "success",
            "trace": [{"step": 0, "done": done, "inference_call": call,
                       "policy_seed": 1907 + call * CALL_STRIDE}],
            "payload_hashes": [{"inference_call": 0, "policy_seed": 1907,
                                "policy_id": "long", "status": "ok",
                                "observation_image": descriptor,
                                "observation_wrist_image": descriptor,
                                "observation_state": descriptor,
                                "response_actions": descriptor}]}}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, value, needle in (("wrong_call", evidence(1, True), "step//5"),
                                    ("wrong_done", evidence(0, False), "final done")):
            path = root / f"{name}.json"
            path.write_text(json.dumps(value))
            audit = Audit()
            audit.audit_episode(path, case, identity_value, {**case, "success": True, "status": "success"})
            if not any(needle in error for error in audit.errors):
                raise AssertionError(f"negative self-test did not reject {name}")
        checkpoint_path = root / "manifest.json"
        checkpoint_path.write_text(json.dumps({"banks": {name: {"adapter_sha256": "d" * 64}
                                          for name in ("spatial", "object", "goal", "long")}}))
        checkpoint = load_checkpoint(checkpoint_path)
        audit = Audit()
        verified = dict(identity_value)
        audit.require(verified["adapter_sha256"] == checkpoint["adapters"]["long"],
                      "verified adapter differs from checkpoint bank")
        if not any("adapter" in error for error in audit.errors):
            raise AssertionError("negative self-test did not reject wrong adapter")
    print("negative self-tests: wrong adapter, call mapping, final done: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path,
                        help="downloaded worker root; repeat for all five")
    parser.add_argument("--checkpoint-manifest", type=Path,
                        help="frozen plugin manifest whose byte SHA256 is the checkpoint identity")
    parser.add_argument("--aggregate", type=Path,
                        help="optional existing aggregate JSON for success-total cross-check")
    parser.add_argument("--output", type=Path, help="optional create-only JSON report")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.input:
        parser.error("--input is required unless --self-test is used")
    checkpoint = load_checkpoint(args.checkpoint_manifest) if args.checkpoint_manifest else None
    audit = Audit()
    manifests = sorted({path.resolve() for root in args.input
                        for path in root.rglob("manifest.json")})
    controllers = []
    for root in args.input:
        path = root / "controller.json"
        try:
            controller = json.loads(path.read_text())
            controllers.append(controller)
            audit.require(controller.get("commit") == EXPECTED_COMMIT,
                          f"{path}: frozen source commit mismatch")
            audit.require(controller.get("status") == "complete",
                          f"{path}: worker controller not complete")
        except Exception as exc:
            audit.errors.append(f"{path}: missing/unreadable controller ({type(exc).__name__})")
    for path in manifests:
        audit.audit_batch(path, checkpoint)
    expected = {(suite, task, init, rep, policy)
                for suite, task, init in CASES for rep in range(10)
                for policy in ("base", PLUGIN[suite])}
    observed = set(audit.episodes)
    audit.require(audit.batch_count == 12, "expected exactly 12 fixed-policy batches")
    audit.require(len(controllers) == 5, "expected exactly five worker controllers")
    audit.require(observed == expected, "episode coverage is not exact 14 x 10 x 2")
    aggregate_checked = False
    if args.aggregate:
        value = json.loads(args.aggregate.read_text())
        totals = value.get("totals", {})
        audit.require(totals.get("base_successes") == audit.summary_successes["base"],
                     "aggregate base success total differs from evidence")
        plugin_successes = sum(count for policy, count in audit.summary_successes.items()
                               if policy != "base")
        audit.require(totals.get("plugin_successes") == plugin_successes,
                     "aggregate plugin success total differs from evidence")
        aggregate_checked = True
    report = {"schema": "pi05_repeat_evidence_audit.v1", "pass": not audit.errors,
              "batches": audit.batch_count, "episodes": len(observed),
              "workers": len(controllers), "expected_source_commit": EXPECTED_COMMIT,
              "expected_episodes": 280, "missing": len(expected - observed),
              "unexpected": len(observed - expected), "checkpoint_manifest_checked": checkpoint is not None,
              "aggregate_totals_checked": aggregate_checked,
              "successes_by_policy": dict(audit.summary_successes), "errors": audit.errors}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("audit report output is create-only")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
