import json
import tempfile
import unittest
from pathlib import Path

from financial_reward_rl.v2_data import (
    LengthBucket,
    audit_truncation,
    build_balanced_training_view,
    build_challenge_views,
    pair_length_features,
    prepare_v2_bundle,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _records() -> list[dict]:
    records = []
    for index in range(3):
        records.append(
            {
                "question": f"matched-{index}",
                "chosen": "good concise",
                "rejected": "weak reply",
                "quality_gap": 1,
            }
        )
    for index in range(5):
        records.append(
            {
                "question": f"aligned-{index}",
                "chosen": "good detailed answer with evidence",
                "rejected": "weak",
                "quality_gap": 2,
            }
        )
    records.append(
        {
            "question": "anti-0",
            "chosen": "correct",
            "rejected": "verbose but wrong answer here",
            "quality_gap": 1,
        }
    )
    return records


class V2DataTests(unittest.TestCase):
    def test_length_bucket_identifies_natural_anti_length_pair(self) -> None:
        features = pair_length_features(
            {
                "question": "q",
                "chosen": "correct",
                "rejected": "verbose but wrong answer here",
            },
            lambda text: len(text.split()),
        )
        self.assertEqual(features.bucket, LengthBucket.ANTI_LENGTH)

    def test_balanced_training_view_equalizes_three_length_strata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            output = root / "v2.jsonl"
            _write_jsonl(source, _records())

            summary = build_balanced_training_view(
                source,
                output,
                max_repeat=3,
                seed=7,
            )

            self.assertEqual(summary["source_pairs"], 9)
            self.assertEqual(summary["output_pairs"], 9)
            self.assertEqual(
                set(summary["counts_after"].values()),
                {3},
            )
            self.assertAlmostEqual(
                summary["anti_length_fraction_before"],
                1 / 9,
            )
            self.assertAlmostEqual(
                summary["anti_length_fraction_after"],
                1 / 3,
            )
            self.assertEqual(summary["repeated_output_instances"], 2)

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all("_v2" in row for row in rows))
            self.assertTrue(
                any(row["_v2"]["sampling_round"] > 0 for row in rows)
            )

    def test_challenge_views_are_fixed_subsets_of_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "test.jsonl"
            _write_jsonl(source, _records())

            summary = build_challenge_views(source, root / "challenges")

            self.assertEqual(summary["counts"]["iid"], 9)
            self.assertEqual(summary["counts"]["length_matched"], 3)
            self.assertEqual(summary["counts"]["reversed_length"], 1)

    def test_truncation_audit_compares_context_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            _write_jsonl(
                path,
                [
                    {
                        "question": "q",
                        "chosen": "one two three four five",
                        "rejected": "one two",
                    }
                ],
            )
            audit = audit_truncation(
                path,
                lambda text: len(text.split()),
                context_lengths=(2, 4),
            )
            self.assertEqual(audit["unique_responses"], 2)
            self.assertEqual(
                audit["contexts"]["2"]["truncated_responses"],
                1,
            )
            self.assertEqual(
                audit["contexts"]["4"]["truncated_responses"],
                1,
            )

    def test_prepare_v2_bundle_writes_manifest_and_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            test = root / "test.jsonl"
            _write_jsonl(train, _records())
            _write_jsonl(eval_path, _records())
            _write_jsonl(test, _records())

            manifest = prepare_v2_bundle(
                train_path=train,
                eval_path=eval_path,
                test_path=test,
                output_dir=root / "v2",
                monitor_size=4,
                seed=11,
            )

            self.assertFalse(
                manifest["policy"]["new_labels_synthesized"]
            )
            self.assertEqual(
                manifest["monitor"]["monitor_pairs"],
                4,
            )
            self.assertTrue((root / "v2" / "manifest.json").is_file())
            self.assertTrue(
                (root / "v2" / "train_balanced_hard.jsonl").is_file()
            )


if __name__ == "__main__":
    unittest.main()
