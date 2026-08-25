import unittest

from financial_reward_rl.train import build_model_load_plan


class SingleGpuQloraPlanTests(unittest.TestCase):
    def test_qlora_plan_requests_4bit_lora_and_disables_full_precision_flags(self) -> None:
        plan = build_model_load_plan(
            {
                "training_mode": "qlora_4bit",
                "torch_dtype": "bfloat16",
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        )

        self.assertTrue(plan["load_in_4bit"])
        self.assertEqual(plan["quantization_config"]["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(plan["lora"]["r"], 16)
        self.assertFalse(plan["use_freeze_tuning"])


if __name__ == "__main__":
    unittest.main()
