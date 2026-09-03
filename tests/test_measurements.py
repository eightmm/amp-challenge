from __future__ import annotations

import unittest

from amp_challenge.curation import concentration_to_micromolar, molecular_weight
from amp_challenge.measurements import (
    normalize_organism,
    parse_hc50_entries,
    parse_mic_entries,
    threshold_label,
)


class MeasurementTests(unittest.TestCase):
    def test_mass_units_and_terminal_modifications(self) -> None:
        sequence = "KWKLFKKIGAVLKVL"
        free_mass = molecular_weight(sequence)
        amidated_mass = molecular_weight(sequence, c_terminal="amidated")
        self.assertLess(amidated_mass, free_mass)
        converted = concentration_to_micromolar(
            8.0,
            "μg/mL",
            sequence,
            c_terminal="amidated",
        )
        self.assertGreater(converted, 0.0)
        self.assertAlmostEqual(
            concentration_to_micromolar(500.0, "nM", sequence),
            0.5,
        )

    def test_mic_parser_preserves_censoring_ranges_and_nested_strain(self) -> None:
        sequence = "KWKLFKKIGAVLKVL"
        text = (
            "Gram-negative bacteria: Escherichia coli (strain: ATCC 25922, clinical) "
            "(MIC≤8 μg/ml), P. aeruginosa PA14 (MIC=4-16 μM)."
        )
        entries, errors = parse_mic_entries(
            text,
            sequence,
            n_terminal="free",
            c_terminal="free",
            activity="Antibacterial",
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].organism.name, "Escherichia coli")
        self.assertIn("ATCC 25922", entries[0].organism.strain or "")
        self.assertEqual(entries[0].interval.relation, "le")
        self.assertIsNone(entries[0].interval.lower_um)
        self.assertEqual(entries[1].organism.name, "Pseudomonas aeruginosa")
        self.assertEqual(entries[1].interval.relation, "interval")
        self.assertEqual(threshold_label(entries[1].interval, 16.0), "1")
        self.assertEqual(threshold_label(entries[1].interval, 4.0), "")

    def test_fungal_abbreviations_are_not_bacterial(self) -> None:
        organism = normalize_organism("C. albicans ATCC 90028", activity="Antibacterial")
        self.assertFalse(organism.is_bacterial)
        mrsa = normalize_organism("MRSA CCARM 3090", activity="Antibacterial")
        self.assertEqual(mrsa.name, "Staphylococcus aureus")
        self.assertTrue(mrsa.is_bacterial)
        fungal_full_name = normalize_organism(
            "Leptosphaeria maculans",
            activity="Antibacterial",
        )
        self.assertFalse(fungal_full_name.is_bacterial)
        generic = normalize_organism("Pathogenic bacteria", activity="Antibacterial")
        self.assertFalse(generic.is_bacterial)

    def test_hc50_parser(self) -> None:
        entries, errors = parse_hc50_entries(
            "[Ref.1] HC50 > 128 μM against human O+ red blood cells",
            "KWKLFKKIGAVLKVL",
            n_terminal="free",
            c_terminal="free",
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].interval.relation, "gt")
        self.assertEqual(entries[0].cell_source, "human red blood cells")
        self.assertEqual(threshold_label(entries[0].interval, 128.0, higher_is_one=True), "1")

    def test_strict_censoring_proves_threshold_edges(self) -> None:
        mic_entries, _ = parse_mic_entries(
            "E. coli (MIC>64 μM)",
            "KWKLFKKIGAVLKVL",
            n_terminal="free",
            c_terminal="free",
        )
        self.assertEqual(threshold_label(mic_entries[0].interval, 64.0), "0")
        hc50_entries, _ = parse_hc50_entries(
            "HC50<128 μM",
            "KWKLFKKIGAVLKVL",
            n_terminal="free",
            c_terminal="free",
        )
        self.assertEqual(
            threshold_label(hc50_entries[0].interval, 128.0, higher_is_one=True),
            "0",
        )


if __name__ == "__main__":
    unittest.main()
