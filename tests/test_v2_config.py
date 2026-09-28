import json
import unittest
from pathlib import Path


class V2TrainingConfigTests(unittest.TestCase):
    def test_v2_config_keeps_effective_batch_and_doubles_context(self) -> None:
        path = Path("configs/training/v2_shortcut_robust_1024.json")
        config = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(config["max_length"], 1024)
        effective_batch = (
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
        )
        self.assertEqual(effective_batch, 8)
        self.assertEqual(config["training_mode"], "qlora_4bit")
        self.assertFalse(
            config["v2_contract"]["new_preference_labels_synthesized"]
        )
        self.assertIn(
            "train_balanced_hard.jsonl",
            config["train_file"],
        )


if __name__ == "__main__":
    unittest.main()
