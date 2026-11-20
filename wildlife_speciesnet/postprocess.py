"""Apply the official SpeciesNet 4.0.2 geographic rollup to classifier scores."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from wildlife_speciesnet.constants import Classification
from wildlife_speciesnet.geofence_utils import geofence_animal_classification

DATA = Path(__file__).resolve().parent / "data"
EXPECTED = {
    "classifier-labels.txt": "e7f14eb90514af42a3351db648dc1358daa6f1175b71e548e65a9e7678c79664",
    "taxonomy.txt": "156673d5f66be4e898fdb1cd2ed518a35a36d3f51578b26833d1d3f5deebc815",
    "geofence.json": "bf08da6263405a125caf49d1d221e82834e3d6f23580619945627c88e31d2fba",
}


def checked(name, compressed=False):
    path = DATA / (name + ".gz" if compressed else name)
    data = gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED[name]:
        raise ValueError(f"SpeciesNet geographic data checksum mismatch: {name}")
    return data


class GeographicRollup:
    def __init__(self, country, region):
        if len(country) != 3 or (region and len(region) != 2):
            raise ValueError(
                "SpeciesNet country must be ISO alpha-3; region must be a two-letter code"
            )
        self.country = country.upper()
        self.region = region.upper() if region else None
        self.labels = checked("classifier-labels.txt").decode().splitlines()
        if len(self.labels) != 2498 or any(
            len(label.split(";")) != 7 for label in self.labels
        ):
            raise ValueError("Invalid SpeciesNet classifier label data")
        taxonomy = checked("taxonomy.txt").decode().splitlines()
        self.taxonomy = {";".join(label.split(";")[1:6]): label for label in taxonomy}
        for label in (
            Classification.BLANK,
            Classification.VEHICLE,
            Classification.UNKNOWN,
        ):
            self.taxonomy.pop(";".join(label.split(";")[1:6]), None)
        for label in (Classification.HUMAN, Classification.ANIMAL):
            self.taxonomy[";".join(label.split(";")[1:6])] = label
        self.geofence = json.loads(checked("geofence.json", compressed=True))

    @classmethod
    def from_environment(cls):
        country = os.environ.get("FRIGATE_SPECIESNET_COUNTRY", "").strip()
        region = os.environ.get("FRIGATE_SPECIESNET_STATE", "").strip()
        if not country and not region:
            return None
        if not country:
            raise ValueError("SpeciesNet country is required when setting a state")
        return cls(country, region)

    def predict(self, probs):
        if len(probs) != len(self.labels):
            raise ValueError("SpeciesNet label count does not match model output")
        indices = np.argsort(probs)[-5:][::-1]
        labels = [self.labels[int(i)] for i in indices]
        scores = [float(probs[int(i)]) for i in indices]
        if labels[0] in (
            Classification.BLANK,
            Classification.HUMAN,
            Classification.VEHICLE,
            Classification.UNKNOWN,
        ):
            return labels[0], scores[0], "classifier"
        return geofence_animal_classification(
            labels=labels,
            scores=scores,
            country=self.country,
            admin1_region=self.region,
            taxonomy_map=self.taxonomy,
            geofence_map=self.geofence,
            enable_geofence=True,
        )


def display_name(label):
    if label in (
        Classification.BLANK,
        Classification.HUMAN,
        Classification.VEHICLE,
        Classification.UNKNOWN,
        Classification.ANIMAL,
    ):
        return "none"
    parts = label.split(";")
    if len(parts) != 7:
        raise ValueError("Invalid SpeciesNet result")
    name = parts[-1].strip() or " ".join(p for p in parts[1:6] if p)
    if name.lower() == "grey fox":
        name = "gray fox"
    return name or "none"
