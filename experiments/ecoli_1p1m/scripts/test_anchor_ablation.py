#!/usr/bin/env python3

import random
import tempfile
import unittest
from pathlib import Path

import anchor_ablation


class AnchorAblationTest(unittest.TestCase):
    def test_anchor_envelope_preserves_source_and_tightens_with_more_anchors(self) -> None:
        rng = random.Random(123)
        reference = "".join(rng.choice("ACGT") for _ in range(1200))
        windows = anchor_ablation.make_windows(reference, 40, 10)
        source_idx = 12
        query = anchor_ablation.mutate_substitutions(
            windows[source_idx].sequence, 2, random.Random(7)
        )
        anchor_distances = anchor_ablation.precompute_anchor_distances(windows)

        one_anchor = anchor_ablation.select_anchor_ids(
            "proximal", query, source_idx, windows, anchor_distances, 1, random.Random(1)
        )
        four_anchors = anchor_ablation.select_anchor_ids(
            "proximal", query, source_idx, windows, anchor_distances, 4, random.Random(1)
        )

        one_metrics = anchor_ablation.evaluate_anchor_set(
            query, source_idx, windows, anchor_distances, one_anchor, 2
        )
        four_metrics = anchor_ablation.evaluate_anchor_set(
            query, source_idx, windows, anchor_distances, four_anchors, 2
        )

        self.assertEqual(one_metrics.false_negative_count, 0)
        self.assertEqual(four_metrics.false_negative_count, 0)
        self.assertLessEqual(four_metrics.envelope_size, one_metrics.envelope_size)

    def test_run_experiment_writes_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = root / "ref.fa"
            ref.write_text(">ref\n" + ("ACGT" * 300) + "\n", encoding="utf-8")
            out = root / "anchors.tsv"

            rows = anchor_ablation.run_experiment(
                reference_path=ref,
                out_path=out,
                window_length=40,
                stride=20,
                query_count=4,
                query_edits=2,
                tolerance=2,
                anchor_counts=[1, 2],
                strategies=["random", "proximal", "far"],
                seed=11,
            )

            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("strategy\tanchor_count\tquery_count", text)
            self.assertIn("proximal\t2\t4", text)
            self.assertTrue(rows)
            self.assertTrue(all(row["false_negative_count_total"] == "0" for row in rows))


if __name__ == "__main__":
    unittest.main()
