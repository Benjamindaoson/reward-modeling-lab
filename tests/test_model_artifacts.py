import json
import tempfile
import unittest
from pathlib import Path

from financial_reward_rl.model_artifacts import resolve_model_artifact


class ModelArtifactTests(unittest.TestCase):
    def test_adapter_checkpoint_requires_and_uses_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            base = root / "base"
            adapter.mkdir()
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "ignored-by-explicit-path"}), encoding="utf-8")

            plan = resolve_model_artifact(adapter, base)

        self.assertTrue(plan["is_adapter"])
        self.assertEqual(plan["base_model_path"], base)
        self.assertEqual(plan["adapter_path"], adapter)
        self.assertTrue(plan["load_in_4bit"])


if __name__ == "__main__":
    unittest.main()
