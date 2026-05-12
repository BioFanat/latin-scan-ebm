#!/usr/bin/env python3
"""Check training data: how many lines are skipped and why."""

from pathlib import Path

from latin_ebm.corpus.pedecerto import parse_xml
from latin_ebm.enumerate import enumerate_parses
from latin_ebm.evaluate import book_split
from latin_ebm.meters import Hexameter
from latin_ebm.train import precompute_training_data
from latin_ebm.features import build_feature_index

# Load
result = parse_xml(Path(__file__).parent.parent.parent / "pedecerto-raw" / "VERG-aene.xml")
print(f"Loaded {len(result.examples)} examples ({result.skipped} skipped during XML parsing)")
print()

# Split
train_examples, test_examples = book_split(result.examples, ("1", "2"))
print(f"After split:")
print(f"  Train (books 3-12): {len(train_examples)} lines")
print(f"  Test (books 1-2): {len(test_examples)} lines")
print()

# Check training data preprocessing
meter = Hexameter()
print("Training data preprocessing:")

no_candidates = 0
no_gold_match = 0
ok = 0

for i, ex in enumerate(train_examples):
    candidates = enumerate_parses(ex.line, meter)
    if not candidates:
        no_candidates += 1
        continue
    
    # Check if gold is in candidate set
    gold_match = False
    for c in candidates:
        if c.foot_types == ex.gold_parse.foot_types and c.slots == ex.gold_parse.slots:
            gold_match = True
            break
    
    if not gold_match:
        no_gold_match += 1
    else:
        ok += 1

print(f"  OK (gold reachable): {ok}")
print(f"  No candidates: {no_candidates}")
print(f"  Gold not in candidates: {no_gold_match}")
print(f"  Total skipped: {no_candidates + no_gold_match}")
print(f"  Usable for training: {ok} ({100*ok/len(train_examples):.1f}%)")
print()

# Check test data
print("Test data:")
test_candidates = []
test_gold_reachable = 0
test_no_candidates = 0

for ex in test_examples:
    candidates = enumerate_parses(ex.line, meter)
    if not candidates:
        test_no_candidates += 1
        continue
    
    test_candidates.append(len(candidates))
    
    # Check gold
    for c in candidates:
        if c.foot_types == ex.gold_parse.foot_types and c.slots == ex.gold_parse.slots:
            test_gold_reachable += 1
            break

import numpy as np
print(f"  Lines with candidates: {len(test_candidates)}")
print(f"  Gold reachable: {test_gold_reachable}/{len(test_candidates)} ({100*test_gold_reachable/len(test_candidates):.1f}%)")
print(f"  Candidate set sizes: mean={np.mean(test_candidates):.1f}, median={np.median(test_candidates):.0f}, "
      f"min={np.min(test_candidates)}, max={np.max(test_candidates)}")
