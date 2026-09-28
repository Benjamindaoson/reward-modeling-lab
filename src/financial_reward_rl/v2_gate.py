"""Promotion gate for Reward Modeling Lab V2.

The gate encodes the V2 objective directly: IID accuracy may stay flat or fall
slightly, but controlled shortcut-robustness must improve materially.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _accuracy_from_eval(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "overall" in payload:
        value = payload["overall"].get("pairwise_accuracy")
    else:
        value = payload.get("pairwise_accuracy")
    if value is None:
        raise ValueError(f"pairwise_accuracy missing from {path}")
    return float(value)


def _v1_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        return {
            "iid": float(payload["frozen_test"]["finetuned_pairwise_accuracy"]),
            "length_matched": float(
                payload["shortcut_audit"]["length_matched"]["finetuned"]
            ),
            "reversed_length": float(
                payload["shortcut_audit"]["reversed_length"]["finetuned"]
            ),
        }
    except KeyError as exc:
        raise ValueError(
            f"V1 summary missing required metric: {exc}"
        ) from exc


def evaluate_v2_promotion(
    *,
    v1: dict[str, float],
    v2: dict[str, float],
    max_iid_drop_pp: float = 5.0,
    min_matched_gain_pp: float = 5.0,
    min_reversed_gain_pp: float = 5.0,
) -> dict[str, Any]:
    required = {"iid", "length_matched", "reversed_length"}
    for name, metrics in (("v1", v1), ("v2", v2)):
        missing = sorted(required - set(metrics))
        if missing:
            raise ValueError(f"{name} metrics missing: {', '.join(missing)}")

    iid_delta_pp = (v2["iid"] - v1["iid"]) * 100.0
    matched_delta_pp = (
        v2["length_matched"] - v1["length_matched"]
    ) * 100.0
    reversed_delta_pp = (
        v2["reversed_length"] - v1["reversed_length"]
    ) * 100.0

    checks = {
        "iid_drop_within_budget": iid_delta_pp >= -max_iid_drop_pp,
        "length_matched_gain": matched_delta_pp >= min_matched_gain_pp,
        "reversed_length_gain": reversed_delta_pp >= min_reversed_gain_pp,
    }
    passed = all(checks.values())

    return {
        "schema_version": "2.0",
        "objective": (
            "improve shortcut robustness without catastrophic IID regression"
        ),
        "v1": v1,
        "v2": v2,
        "delta_percentage_points": {
            "iid": round(iid_delta_pp, 4),
            "length_matched": round(matched_delta_pp, 4),
            "reversed_length": round(reversed_delta_pp, 4),
        },
        "thresholds": {
            "max_iid_drop_pp": max_iid_drop_pp,
            "min_matched_gain_pp": min_matched_gain_pp,
            "min_reversed_gain_pp": min_reversed_gain_pp,
        },
        "checks": checks,
        "passed": passed,
        "interpretation": (
            "V2 improves reward validity under controlled length interventions."
            if passed
            else "V2 does not yet satisfy the shortcut-robustness promotion gate."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare V2 reward validity against the verified V1 baseline."
    )
    parser.add_argument(
        "--v1-summary",
        default=Path("docs/results/data/results_summary.json"),
        type=Path,
    )
    parser.add_argument("--v2-iid", required=True, type=Path)
    parser.add_argument("--v2-matched", required=True, type=Path)
    parser.add_argument("--v2-reversed", required=True, type=Path)
    parser.add_argument("--max-iid-drop-pp", type=float, default=5.0)
    parser.add_argument("--min-matched-gain-pp", type=float, default=5.0)
    parser.add_argument("--min-reversed-gain-pp", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)

    result = evaluate_v2_promotion(
        v1=_v1_metrics(args.v1_summary),
        v2={
            "iid": _accuracy_from_eval(args.v2_iid),
            "length_matched": _accuracy_from_eval(args.v2_matched),
            "reversed_length": _accuracy_from_eval(args.v2_reversed),
        },
        max_iid_drop_pp=args.max_iid_drop_pp,
        min_matched_gain_pp=args.min_matched_gain_pp,
        min_reversed_gain_pp=args.min_reversed_gain_pp,
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_pass and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
