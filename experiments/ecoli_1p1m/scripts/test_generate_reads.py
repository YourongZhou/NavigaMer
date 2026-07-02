#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import generate_reads


REFERENCE = ">ecoli\n" + ("ACGT" * 30_000) + "\n"


class GenerateReadsTest(unittest.TestCase):
    def test_build_dataset_shapes_and_lengths(self) -> None:
        reference = "ACGT" * 100

        exact_reads, exact_truth = generate_reads.build_dataset(
            reference, "exact", 0, 4, 11
        )
        self.assertEqual(len(exact_reads), 4)
        self.assertTrue(all(len(sequence) == generate_reads.READ_LENGTH
                            for _, sequence in exact_reads))
        self.assertTrue(all(row["edit_script"] == [] for row in exact_truth))

        substitution_reads, substitution_truth = generate_reads.build_dataset(
            reference, "substitution", 3, 4, 17
        )
        self.assertEqual(len(substitution_reads), 4)
        self.assertTrue(all(len(sequence) == generate_reads.READ_LENGTH
                            for _, sequence in substitution_reads))
        self.assertTrue(all(len(row["edit_script"]) == 3
                            for row in substitution_truth))
        self.assertTrue(all(op["op"] == "sub"
                            for row in substitution_truth
                            for op in row["edit_script"]))

        mixed_reads, mixed_truth = generate_reads.build_dataset(
            reference, "mixed", 5, 8, 23
        )
        self.assertEqual(len(mixed_reads), 8)
        self.assertTrue(all(145 <= len(sequence) <= 155
                            for _, sequence in mixed_reads))
        self.assertTrue(all(len(row["edit_script"]) == 5 for row in mixed_truth))

    def test_generate_all_writes_manifest_and_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_path = root / "ref.fa"
            out_dir = root / "reads"
            reference_path.write_text(REFERENCE, encoding="utf-8")

            generate_reads.generate_all(reference_path, out_dir, 20260625)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["reference_path"], str(reference_path))
            self.assertEqual(manifest["read_length"], generate_reads.READ_LENGTH)
            self.assertEqual(manifest["seed"], 20260625)
            self.assertEqual(len(manifest["datasets"]), 19)

            dataset_names = {row["dataset"] for row in manifest["datasets"]}
            self.assertIn("main_exact_10000", dataset_names)
            self.assertIn("main_mixed_tau5_10000", dataset_names)
            self.assertIn("main_hard_tau2_5000", dataset_names)
            self.assertIn("oracle_prefix50000_mixed_tau3_1000", dataset_names)
            self.assertIn("oracle_prefix100000_mixed_tau5_1000", dataset_names)

            sample_fastq = (out_dir / "main_exact_10000.fq").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(sample_fastq[0], "@exact_tau0_read00000")
            self.assertEqual(len(sample_fastq[1]), generate_reads.READ_LENGTH)
            self.assertEqual(sample_fastq[2], "+")
            self.assertEqual(len(sample_fastq[3]), generate_reads.READ_LENGTH)

            mixed_truth_lines = (
                out_dir / "oracle_prefix50000_mixed_tau1_1000.truth.tsv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                mixed_truth_lines[0],
                "read_id\tfamily\ttau\tsource_start\tsource_end\tseed\tsequence_length\tedit_script_json",
            )
            self.assertEqual(len(mixed_truth_lines), 1001)

    def test_mixed_dataset_covers_all_edit_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_path = root / "ref.fa"
            out_dir = root / "reads"
            reference_path.write_text(REFERENCE, encoding="utf-8")

            generate_reads.generate_all(reference_path, out_dir, 20260625)

            truth_lines = (
                out_dir / "main_mixed_tau1_10000.truth.tsv"
            ).read_text(encoding="utf-8").splitlines()[1:]
            coverage = set()
            for line in truth_lines:
                payload = line.split("\t")[-1]
                for op in json.loads(payload):
                    coverage.add(op["op"])
            self.assertEqual(coverage, {"sub", "ins", "del"})


if __name__ == "__main__":
    unittest.main()
