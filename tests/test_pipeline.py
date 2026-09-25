import unittest

import jax.numpy as jnp
import numpy as np

from noelle.model import Arch, BackboneFamily
from noelle.pipeline import DecisionPipeline, available_models, load_pipeline
from train import collate, encode_row


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return type("Encoding", (), {"ids": [2 + ord(char) % 100 for char in text]})()


class RuntimeStub:
    arch = Arch.SmollmDB135m

    def logits(self, ids, ids_mask, ctx_ids, ctx_mask, *, choice_count):
        return jnp.arange(choice_count, dtype=jnp.float32)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = DecisionPipeline(
            RuntimeStub(), TokenizerStub(),
            max_prompt_tokens=96, max_option_tokens=8,
        )

    def test_preprocessing_matches_training_format_and_preserves_option_order(self):
        state = {"z": "world", "a": "hello"}
        options = ["first", "second", "third"]
        row = {
            "id": "example", "kind": "score", "state": state,
            "question": "Rate?", "options": options,
            "target": [0.1, 0.2, 0.7],
        }
        expected = encode_row(row, TokenizerStub(), 96, 8, 64)
        prepared = self.pipeline.preprocess(
            state=state, question="Rate?", options=options, kind="score",
        )
        training_batch = collate([expected], 96, 8)
        for name in ("ids", "ids_mask", "ctx_ids", "ctx_mask"):
            np.testing.assert_array_equal(np.asarray(getattr(prepared, name)), np.asarray(training_batch[name]))
        self.assertEqual(prepared.choice_count, 3)
        self.assertEqual(prepared.choices_truncated, expected.choices_truncated)
        self.assertEqual(prepared.prompt_truncated, expected.prompt_truncated)

    def test_prediction_returns_one_probability_per_input_option(self):
        with self.assertWarnsRegex(UserWarning, "Every option is one token"):
            result = self.pipeline.predict(
                state="hello", question="Choose?", options=["A", "B", "C"],
            )
        self.assertEqual(result.options, ("A", "B", "C"))
        self.assertEqual(result.selected_index, 2)
        self.assertEqual(result.selected_option, "C")
        self.assertAlmostEqual(sum(result.probabilities), 1.0)
        self.assertLess(result.probabilities[0], result.probabilities[1])
        self.assertLess(result.probabilities[1], result.probabilities[2])

    def test_rejects_bad_inputs_and_unavailable_model(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.pipeline.predict(state="x", question="q", options=["only one"])
        with self.assertRaisesRegex(ValueError, "kind"):
            self.pipeline.predict(state="x", question="q", options=["a", "b"], kind="other")
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            self.pipeline.predict(state={1, 2}, question="q", options=["a", "b"])
        self.assertEqual(available_models(), (Arch.SmollmDB135m,))
        self.assertIs(Arch.SmollmDB135m.family, BackboneFamily.DB)
        with self.assertRaisesRegex(TypeError, "Arch value"):
            load_pipeline("eb/unknown")


if __name__ == "__main__":
    unittest.main()
