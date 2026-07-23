import unittest

from scripts.run_chartqa_conflict import classify, extract_final_answer, normalize_answer


class ChartQAAttributionTests(unittest.TestCase):
    def test_numeric_normalization_is_exact(self):
        self.assertEqual(normalize_answer("1,200.00", "usd"),
                         normalize_answer("$1,200", "usd"))
        self.assertEqual(normalize_answer("1/2", "unitless"),
                         normalize_answer("0.5", "unitless"))
        self.assertEqual(normalize_answer("−12.0%", "percent"),
                         normalize_answer("-12%", "percent"))

    def test_boolean_synonyms(self):
        self.assertEqual(normalize_answer("True"), ("boolean", "yes"))
        self.assertEqual(normalize_answer("incorrect"), ("boolean", "no"))

    def test_strict_attribution_outcomes(self):
        row = {"image_answer": "1,200.00", "text_answer": "1,250", "unit_class": "usd"}
        self.assertEqual(classify("#### $1,200", row)[0], "image")
        self.assertEqual(classify("#### $1,250", row)[0], "text")
        self.assertEqual(classify("#### $1,225", row)[0], "neither")
        self.assertEqual(
            classify("Reasoning mentions 1200 and 1250 without a final marker", row)[0],
            "invalid",
        )

    def test_identical_candidates_are_ambiguous(self):
        row = {"image_answer": "20%", "text_answer": "20.0%", "unit_class": "percent"}
        self.assertEqual(classify("#### 20%", row)[0], "ambiguous")

    def test_explicit_answer_markers_without_hashes(self):
        self.assertEqual(extract_final_answer("Answer: 0.9"), "0.9")
        self.assertEqual(
            extract_final_answer("The sources conflict. Therefore, the answer is 883."),
            "883",
        )

    def test_does_not_mine_unmarked_reasoning(self):
        self.assertIsNone(
            extract_final_answer("The chart says 11.8 while the report says 12.8."),
        )


if __name__ == "__main__":
    unittest.main()
