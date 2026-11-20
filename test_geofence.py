"""Behavior checks for the pinned SpeciesNet geographic postprocessing."""

import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import numpy as np

from wildlife_speciesnet.constants import Classification
from wildlife_speciesnet.geofence_utils import should_geofence_animal_classification
from wildlife_speciesnet.postprocess import GeographicRollup, display_name


class GeographicRollupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = GeographicRollup("USA", "TX")

    def label(self, text):
        return next(x for x in self.geo.labels if x.endswith(";" + text))

    def score(self, name, score=0.95):
        p = np.zeros(len(self.geo.labels), dtype=np.float32)
        p[self.geo.labels.index(self.label(name))] = score
        return p

    def test_known_texas_wildlife_remains_specific(self):
        for name in (
            "grey fox",
            "striped skunk",
            "virginia opossum",
            "northern raccoon",
            "nine-banded armadillo",
            "american black bear",
        ):
            with self.subTest(name=name):
                label, score, source = self.geo.predict(self.score(name))
                self.assertEqual(label, self.label(name))
                self.assertEqual(source, "classifier")
                self.assertFalse(
                    should_geofence_animal_classification(
                        label, "USA", "TX", self.geo.geofence, True
                    )
                )
        self.assertEqual(display_name(self.label("grey fox")), "gray fox")

    def test_out_of_range_species_rolls_up_not_to_a_different_species(self):
        label = self.label("tiger")
        self.assertTrue(
            should_geofence_animal_classification(
                label, "USA", "TX", self.geo.geofence, True
            )
        )
        result, score, source = self.geo.predict(self.score("tiger"))
        self.assertIn("geofence", source)
        self.assertNotEqual(result, label)
        self.assertEqual(result.split(";")[1:4], label.split(";")[1:4])
        self.assertEqual(result.split(";")[4:6], ["", ""])
        self.assertGreater(score, 0.9)

    def test_blank_remains_none_and_no_location_disables_feature(self):
        p = np.zeros(len(self.geo.labels), dtype=np.float32)
        p[self.geo.labels.index(Classification.BLANK)] = 1
        label, _, source = self.geo.predict(p)
        self.assertEqual(label, Classification.BLANK)
        self.assertEqual(display_name(label), "none")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(GeographicRollup.from_environment())
        with patch.dict(os.environ, {"FRIGATE_SPECIESNET_STATE": "TX"}, clear=True):
            with self.assertRaises(ValueError):
                GeographicRollup.from_environment()

    def test_index_mapping_matches_deployed_readable_labels(self):
        readable = (
            (Path(__file__).resolve().parent / "speciesnet-labelmap.txt")
            .read_text()
            .splitlines()
        )
        self.assertEqual(len(readable), len(self.geo.labels))
        norm = lambda s: re.sub("[^a-z0-9]", "", s.lower().replace("gray", "grey"))
        mismatches = []
        for index, (human, raw) in enumerate(zip(readable, self.geo.labels)):
            parts = raw.split(";")
            allowed = {norm(parts[-1]), norm(parts[4] + parts[5])}
            if norm(human) not in allowed:
                mismatches.append((index, human))
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
