"""Build a readable, index-preserving Frigate label map for SpeciesNet-Crop."""

from collections import defaultdict
import argparse
from pathlib import Path
import re

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--crop-labels", type=Path, required=True)
parser.add_argument("--full-labels", type=Path, required=True)
parser.add_argument("--output", type=Path, default=here / "speciesnet-labelmap.txt")
args = parser.parse_args()
normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
by_common = defaultdict(set)
by_scientific = defaultdict(set)
for line in args.full_labels.read_text().splitlines():
    parts = line.split(";")
    if len(parts) < 7:
        raise ValueError("Unexpected taxonomy format")
    common = parts[-1].strip()
    if common:
        by_common[normalize(common)].add(common)
    if parts[4] and parts[5]:
        by_scientific[normalize(parts[4] + parts[5])].add(parts[4] + " " + parts[5])

labels = [None] * 2498
unmatched = []
for line in args.crop_labels.read_text().splitlines():
    raw, index = line.rsplit(",", 1)
    index = int(index)
    choices = by_common.get(normalize(raw), set()) or by_scientific.get(
        normalize(raw), set()
    )
    if len(choices) > 1:
        raise ValueError("Ambiguous label: " + raw)
    if not choices:
        unmatched.append(raw)
    name = next(iter(choices), raw)
    if name.lower() == "grey fox":
        name = "gray fox"
    if labels[index] is not None:
        raise ValueError("Duplicate index")
    labels[index] = name
if any(not x for x in labels):
    raise ValueError("Incomplete label map")
dest = args.output
dest.write_text("\n".join(labels) + "\n")
print(
    f"Wrote {len(labels)} labels to {dest}; original compact label retained for: {unmatched}"
)
