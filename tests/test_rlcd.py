import gzip
import json
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from train_rlcd import UniqueRows, check_split_groups, proper_score, sampled_policy_loss


class RLCDTest(unittest.TestCase):
    def test_policy_gradient_ignores_padded_choice(self):
        logits = jnp.array([[0.2, -0.1, 0.0]])
        valid = jnp.array([[True, True, False]])
        target = jnp.array([[0.7, 0.3, 0.0]])
        ordinal = jnp.array([False])

        def loss(value):
            return sampled_policy_loss(
                value, valid, target, ordinal, jax.random.PRNGKey(0),
                candidates=16, sigma=0.3, spherical_weight=0.2, rps_weight=0.2,
            )[0]

        gradient = np.asarray(jax.grad(loss)(logits))
        self.assertTrue(np.isfinite(gradient).all())
        self.assertEqual(float(gradient[0, 2]), 0.0)

    def test_ordinal_score_rewards_matching_distribution(self):
        target = jnp.array([[1.0, 0.0, 0.0]])
        valid = jnp.ones_like(target, dtype=bool)
        ordinal = jnp.array([True])
        good = jnp.array([[0.9, 0.05, 0.05]])
        bad = jnp.array([[0.05, 0.05, 0.9]])
        score = lambda p: proper_score(jnp.log(p), p, target, ordinal, valid, 0.2, 0.2)
        self.assertGreater(float(score(good)[0]), float(score(bad)[0]))

    def test_deduplicates_configs_and_detects_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rows.jsonl.gz"
            row = {
                "id": "item", "group_id": "group", "source": "test", "kind": "choice",
                "question": "Choose", "options": ["a", "b"], "target": [1.0, 0.0],
                "state": "state", "split": "train",
            }
            with gzip.open(source, "wt") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.write(json.dumps({**row, "id": "heldout", "split": "validation"}) + "\n")
            training = UniqueRows(root / "train.sqlite", [source, source], "train")
            validation = UniqueRows(root / "validation.sqlite", [source], "validation")
            try:
                self.assertEqual(len(training), 1)
                self.assertEqual(training.duplicates, 1)
                with self.assertRaisesRegex(ValueError, "group overlap"):
                    check_split_groups(training, validation)
            finally:
                training.close()
                validation.close()


if __name__ == "__main__":
    unittest.main()
