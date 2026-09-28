import unittest

from financial_reward_rl.v2_gate import evaluate_v2_promotion


class V2PromotionGateTests(unittest.TestCase):
    def test_promotes_when_robustness_improves_despite_small_iid_drop(self) -> None:
        result = evaluate_v2_promotion(
            v1={
                "iid": 0.9135,
                "length_matched": 0.7778,
                "reversed_length": 0.7490,
            },
            v2={
                "iid": 0.8900,
                "length_matched": 0.8500,
                "reversed_length": 0.8600,
            },
        )

        self.assertTrue(result["passed"])
        self.assertLess(result["delta_percentage_points"]["iid"], 0)
        self.assertGreater(
            result["delta_percentage_points"]["reversed_length"],
            10,
        )

    def test_rejects_iid_only_improvement_without_challenge_gain(self) -> None:
        result = evaluate_v2_promotion(
            v1={
                "iid": 0.9135,
                "length_matched": 0.7778,
                "reversed_length": 0.7490,
            },
            v2={
                "iid": 0.9300,
                "length_matched": 0.7900,
                "reversed_length": 0.7600,
            },
        )

        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["iid_drop_within_budget"])
        self.assertFalse(result["checks"]["length_matched_gain"])
        self.assertFalse(result["checks"]["reversed_length_gain"])

    def test_rejects_catastrophic_iid_regression(self) -> None:
        result = evaluate_v2_promotion(
            v1={
                "iid": 0.9135,
                "length_matched": 0.7778,
                "reversed_length": 0.7490,
            },
            v2={
                "iid": 0.8000,
                "length_matched": 0.9000,
                "reversed_length": 0.9000,
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["iid_drop_within_budget"])


if __name__ == "__main__":
    unittest.main()
