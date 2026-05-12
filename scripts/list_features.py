#!/usr/bin/env python3
"""List all features in the trained model."""

from pathlib import Path

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import book_split
from latin_ebm.meters import Hexameter
from latin_ebm.train import train_nll

# Load
result = parse_xml(Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml")
train_examples, test_examples = book_split(result.examples, ("1", "2"))

# Train
model, _ = train_nll(
    train_examples,
    l2_lambda=0.01,
    max_iter=200,
)

# List features
names = model.feature_index.names
print(f"Total features: {len(names)}")
print()

# Group by prefix
from collections import defaultdict
groups = defaultdict(list)
for name in names:
    prefix = name.split(":")[0] if ":" in name else name
    groups[prefix].append(name)

for prefix in sorted(groups.keys()):
    feats = groups[prefix]
    print(f"{prefix}: {len(feats)} features")
    for feat in sorted(feats)[:10]:  # show first 10
        print(f"  {feat}")
    if len(feats) > 10:
        print(f"  ... and {len(feats) - 10} more")
    print()
