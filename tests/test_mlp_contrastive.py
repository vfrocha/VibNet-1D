import unittest

import numpy as np
import torch

from src.models.build_mlp_contrastive import (
    MLPClassifier,
    supervised_contrastive_loss,
    train_and_evaluate,
)
from src.models.build_tabnet_resnet_balance import _balance_target_training


class ContrastiveMLPTests(unittest.TestCase):
    def test_supcon_is_finite_and_differentiable(self):
        embeddings = torch.randn(6, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = supervised_contrastive_loss(embeddings, labels, temperature=0.2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(embeddings.grad)
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_domains_have_independent_class_vocabularies(self):
        model = MLPClassifier(10, {"source": 3, "target": 2}, encoder_profile="expanded")
        x = torch.randn(4, 10)
        self.assertEqual(model(x, "source").shape, (4, 3))
        self.assertEqual(model(x, "target").shape, (4, 2))
        self.assertIsNot(model.classifiers["source"][0], model.classifiers["target"][0])

    def test_hybrid_and_smote_counts_explain_prior_identical_outputs(self):
        # A tied majority makes the median equal the maximum, so both methods
        # target the same counts; a skewed fold distinguishes the strategies.
        tied_counts = [3, 8, 8]
        X_tied = np.vstack([np.full((n, 2), c, dtype=np.float32) for c, n in enumerate(tied_counts)])
        y_tied = np.concatenate([np.full(n, c) for c, n in enumerate(tied_counts)])
        _, _, _, tied_smote = _balance_target_training(X_tied, y_tied, "smote", 42)
        _, _, _, tied_hybrid = _balance_target_training(X_tied, y_tied, "hybrid", 42)
        self.assertEqual(tied_smote, tied_hybrid)

        skew_counts = [2, 3, 9]
        X_skew = np.vstack([np.full((n, 2), c, dtype=np.float32) for c, n in enumerate(skew_counts)])
        y_skew = np.concatenate([np.full(n, c) for c, n in enumerate(skew_counts)])
        _, _, _, skew_smote = _balance_target_training(X_skew, y_skew, "smote", 42)
        _, _, _, skew_hybrid = _balance_target_training(X_skew, y_skew, "hybrid", 42)
        self.assertNotEqual(skew_smote, skew_hybrid)
        self.assertEqual(set(skew_smote.values()), {9})
        self.assertEqual(set(skew_hybrid.values()), {3})

    def test_synthetic_loco_training_returns_correct_metrics(self):
        rng = np.random.default_rng(17)
        source_x = rng.normal(size=(24, 6)).astype(np.float32)
        source_y = np.tile([0, 1], 12)
        target_x = rng.normal(size=(20, 6)).astype(np.float32)
        target_y = np.tile([0, 1], 10)
        test_x = rng.normal(size=(8, 6)).astype(np.float32)
        test_y = np.tile([0, 1], 4)
        result = train_and_evaluate(
            {"source": (source_x, source_y), "target": (target_x, target_y)},
            "target", test_x, test_y, mode="joint", epochs=1,
            finetune_epochs=0, batch_size=8, seed=5,
        )
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            self.assertTrue(0.0 <= result[key] <= 1.0)
        self.assertEqual(len(result["y_pred"]), len(test_y))
        self.assertEqual(result["probabilities"].shape, (len(test_y), 2))


if __name__ == "__main__":
    unittest.main()
