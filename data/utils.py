import os
from collections import Counter

def count_labels(dataset):
  # dataset must yield (img, label) tuples
  counts = Counter()
  for i in range(len(dataset)):
    _, label = dataset[i]
    counts[int(label)] += 1
  return counts

def balance_label_file(infile, outfile):
  # simple balancing: read, group by class, downsample to minority count
  import random
  lines = []
  with open(infile, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) >= 2:
        lines.append((parts[0], parts[1]))
  groups = {}
  for fn, lab in lines:
    groups.setdefault(lab, []).append(fn)
  minc = min(len(v) for v in groups.values())
  out = []
  for lab, arr in groups.items():
    sampled = random.sample(arr, minc)
    out += [(s, lab) for s in sampled]
  with open(outfile, 'w') as f:
    f.write('filename label\n')
    for fn, lab in out:
      f.write(f"{fn} {lab}\n")
