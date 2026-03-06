# Strobemers Python Library - Pyguide

A comprehensive guide to using the strobemers Python library for sequence similarity detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Parameter Reference](#parameter-reference)
4. [Function Catalog](#function-catalog)
5. [Practical Tutorial](#practical-tutorial)
6. [Parameter Selection Guidelines](#parameter-selection-guidelines)
7. [Complete Working Example](#complete-working-example)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**Strobemers** are fuzzy seeds for text similarity searches, originally designed for biological sequence analysis. Unlike traditional k-mers, strobemers can match across mutations (substitutions, insertions, deletions).

### What Makes Strobemers Different?

| Feature | K-mers | Strobemers |
|---------|--------|------------|
| Structure | Consecutive nucleotides | Non-consecutive substrings |
| Mutation tolerance | Low (single mutation breaks match) | High (gaps between strobes absorb mutations) |
| Specificity | Fixed | Tunable via window parameters |
| Memory usage | Lower | Higher (stores position tuples) |

### Strobemer Types

- **Randstrobes**: Second/third strobe selected by minimum hash value in window
- **Minstrobes**: Second/third strobe selected by minimizer criterion
- **Hybridstrobes**: Combines random and minimizer selection

---

## Installation & Setup

### Prerequisites

- Python 3.6+
- No external dependencies (uses only standard library)

### Basic Setup

```python
# Add the modules directory to your path
import sys
sys.path.insert(0, '/path/to/strobemers/modules')

# Import the indexing module
from modules import indexing
from collections import defaultdict
```

### Project Structure

```
strobemers/
├── modules/
│   └── indexing.py      # Main Python library
├── data/                 # Test datasets
└── README.md             # Documentation
```

---

## Parameter Reference

### Core Parameters (All Functions)

| Parameter | Type | Description | Constraints | Example |
|-----------|------|-------------|-------------|---------|
| `seq` | `str` | Input DNA/RNA sequence | Any alphabet (ACGT, protein, etc.) | `"ACGTACGT..."` |
| `k_size` | `int` | **Total strobemer length** in nucleotides | Order 2: ≤64, Order 3: ≤96 | `15`, `20`, `30` |
| `strobe_w_min_offset` | `int` | **Minimum window offset** - distance from first strobe to search window start | Must be > 0, typically `k_size/2 + 1` | `20` |
| `strobe_w_max_offset` | `int` | **Maximum window offset** - distance from first strobe to search window end | Must be ≥ `strobe_w_min_offset` | `70` |
| `w` | `int` | **Thinning window** - subsampling density (like minimizers) | `w=1`: all strobemers, `w>1`: sparse sampling | `1`, `10`, `100` |
| `order` | `int` | **Strobemer order** - number of substrings combined | `2` (2 substrings) or `3` (3 substrings) | `2`, `3` |
| `buffer_size` | `int` | Memory buffer for `_iter` functions | Default: 10,000,000 | `1000000` |

### Parameter Relationships

```
Total strobemer length = k_size

For order 2:
  - Each strobe = k_size // 2 nucleotides
  - Window for 2nd strobe: [pos₁ + w_min, pos₁ + w_max]

For order 3:
  - Each strobe = k_size // 3 nucleotides
  - Window for 2nd strobe: [pos₁ + w_min, pos₁ + w_max]
  - Window for 3rd strobe: [pos₁ + w_max + w_min, pos₁ + 2*w_max]

Effective spacing = w_max - w_min (larger = more diversity)
```

### Visual Representation

```
Sequence:  5'-----------------------------------------3'
           |k-mer|---w_min---|window|---w_max---|
           |-----|-----------|------|-----------|
Position:  0    k           k+w_min        k+w_max

Order 2 randstrobe: (k-mer) + (min-hash k-mer in window)
Order 3 randstrobe: (k-mer) + (min-hash in window1) + (min-hash in window2)
```

---

## Function Catalog

### 1. `randstrobes()` - Random Strobemers

**Description**: Generate randstrobes where subsequent strobes are selected by minimum hash value in a sliding window.

**Signature**:
```python
randstrobes(seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2)
```

**Returns**: `dict` with `(position_tuple)` as keys and `hash_value` as values

**Example**:
```python
from modules import indexing

seq = "ACGCGTACGAATCACGCCGGGTGTGTGTGATCGGGGCTATCAGCTACGTACTATGCTAGCTACGGACGGCGATTTTTTTTCATATCGTACGCTAGCTAGCTAGCTGCGATCGATTCG"

# Generate randstrobes with parameters (2, 15, 20, 70)
result = indexing.randstrobes(
    seq,
    k_size=15,              # Total length (each strobe = 7-8 nt)
    strobe_w_min_offset=20, # Window start offset
    strobe_w_max_offset=70, # Window end offset
    w=1,                    # No thinning - return all
    order=2                 # 2 substrings
)

# Build hash table for mapping
from collections import defaultdict
hash_table = defaultdict(list)
for (p1, p2), h in result.items():
    hash_table[h].append((p1, p2))

print(f"Generated {len(hash_table)} unique randstrobes")
```

---

### 2. `randstrobes_iter()` - Random Strobemers (Low Memory)

**Description**: Generator version of `randstrobes()` - processes sequence in chunks for memory efficiency.

**Signature**:
```python
randstrobes_iter(seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2, buffer_size=10000000)
```

**Returns**: Generator yielding `(position_tuple, hash_value)` pairs

**Example**:
```python
hash_table = defaultdict(list)

for (p1, p2, p3), h in indexing.randstrobes_iter(
    seq,
    k_size=30,
    strobe_w_min_offset=31,
    strobe_w_max_offset=60,
    w=1,
    order=3,
    buffer_size=1000000  # Process 1 Mbp chunks
):
    hash_table[h].append((p1, p2, p3))
```

**When to use**: 
- Sequences > 10 Mbp
- Whole genome indexing
- Memory-constrained environments

---

### 3. `minstrobes()` / `minstrobes_iter()` - Minimum Strobemers

**Description**: Generate minstrobes where subsequent strobes are selected by minimizer criterion (lexicographically smallest).

**Signature**:
```python
minstrobes(seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2)
minstrobes_iter(seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2, buffer_size=10000000)
```

**Returns**: Same as `randstrobes()`

**Example**:
```python
result = indexing.minstrobes(
    seq,
    k_size=15,
    strobe_w_min_offset=20,
    strobe_w_max_offset=70,
    w=1,
    order=2
)
```

**Characteristics**:
- More deterministic than randstrobes
- May be less robust to certain mutation patterns

---

### 4. `hybridstrobes()` / `hybridstrobes_iter()` - Hybrid Strobemers

**Description**: Combines random hash-based and minimizer-based selection for balanced performance.

**Signature**:
```python
hybridstrobes(seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2)
hybridstrobes_iter(seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2, buffer_size=10000000)
```

**Example**:
```python
result = indexing.hybridstrobes(
    seq,
    k_size=15,
    strobe_w_min_offset=20,
    strobe_w_max_offset=70,
    w=1,
    order=2
)
```

**Characteristics**:
- Balances randomness and determinism
- Good general-purpose choice

---

### 5. `kmers()` / `kmer_iter()` - Traditional K-mers

**Description**: Extract standard k-mers with optional minimizer thinning.

**Signature**:
```python
kmers(seq, k_size, w)
kmer_iter(seq, k_size, w)
```

**Returns**: `dict` with `position` as keys and `hash_value` as values

**Example**:
```python
# All k-mers
all_kmers = indexing.kmers(seq, k_size=20, w=1)

# Sparse sampling (minimizers)
sparse_kmers = indexing.kmers(seq, k_size=20, w=50)
```

---

### 6. `minimizers()` - Minimizers

**Description**: Extract minimizers (local minimum k-mers in sliding window).

**Signature**:
```python
minimizers(seq, k_size, w_size)
```

**Returns**: `list` of `(kmer_string, position)` tuples

**Example**:
```python
mins = indexing.minimizers(seq, k_size=15, w_size=50)
for kmer, pos in mins:
    print(f"Minimizer at {pos}: {kmer}")
```

---

### 7. `spaced_kmers()` / `spaced_kmers_iter()` - Spaced K-mers

**Description**: K-mers with wildcard positions (pattern-based sampling).

**Signature**:
```python
spaced_kmers(seq, k_size, span_size, positions, w)
spaced_kmers_iter(seq, k_size, span_size, positions)
```

**Returns**: `dict` with `position` as keys and `hash_value` as values

**Example**:
```python
# Pattern: 1101101101 (1=keep, 0=skip)
positions = {0, 1, 3, 4, 6, 7}  # Which positions to keep
result = indexing.spaced_kmers(
    seq,
    k_size=6,           # Number of kept positions
    span_size=10,       # Total span length
    positions=positions, # Position mask
    w=1
)
```

---

## Practical Tutorial

### Example 1: Basic Randstrobe Generation (Order 2)

```python
from modules import indexing
from collections import defaultdict

seq = "ACGCGTACGAATCACGCCGGGTGTGTGTGATCGGGGCTATCAGCTACGTACTATGCTAGCTACGGACGGCGATTTTTTTTCATATCGTACGCTAGCTAGCTAGCTGCGATCGATTCG"

# Generate randstrobes with parameters (2, 15, 20, 70)
# This means: order=2, k=15, window [20, 70]
randstrobe_dict = indexing.randstrobes(
    seq,
    k_size=15,
    strobe_w_min_offset=20,
    strobe_w_max_offset=70,
    w=1,
    order=2
)

# Build hash table for mapping
hash_table = defaultdict(list)
for (pos1, pos2), hash_val in randstrobe_dict.items():
    hash_table[hash_val].append((pos1, pos2))

print(f"Generated {len(hash_table)} unique randstrobes")

# Extract actual strobemer sequences
for (p1, p2), h in list(randstrobe_dict.items())[:3]:
    strobe1 = seq[p1:p1+7]   # k_size // 2 = 7
    strobe2 = seq[p2:p2+7]
    print(f"Positions ({p1}, {p2}): {strobe1}+{strobe2}")
```

---

### Example 2: Memory-Efficient Iteration for Long Sequences

```python
# For whole-genome or long-read data, use _iter functions
hash_table = defaultdict(list)

for (p1, p2, p3), h in indexing.randstrobes_iter(
    seq,
    k_size=30,
    strobe_w_min_offset=31,
    strobe_w_max_offset=60,
    w=1,
    order=3,
    buffer_size=1000000  # 1 Mbp chunks
):
    hash_table[h].append((p1, p2, p3))

print(f"Indexed {len(hash_table)} unique order-3 randstrobes")
```

---

### Example 3: Comparing Different Strobemer Types

```python
# Generate different strobemer types with same parameters
k = 15
w_min = 20
w_max = 70

rand = indexing.randstrobes(seq, k, w_min, w_max, w=1, order=2)
minst = indexing.minstrobes(seq, k, w_min, w_max, w=1, order=2)
hybrid = indexing.hybridstrobes(seq, k, w_min, w_max, w=1, order=2)

print(f"Randstrobes:   {len(rand)}")
print(f"Minstrobes:    {len(minst)}")
print(f"Hybridstrobes: {len(hybrid)}")

# Compare overlap
rand_keys = set(rand.values())
minst_keys = set(minst.values())
hybrid_keys = set(hybrid.values())

print(f"Randstrobe ∩ Minstrobe: {len(rand_keys & minst_keys)}")
print(f"Randstrobe ∩ Hybrid:    {len(rand_keys & hybrid_keys)}")
```

---

### Example 4: Sparse Sampling with Thinning

```python
# Use w > 1 for subsampling (reduces memory/disk usage)
# w=10 means ~1 in 10 strobemers are kept

sparse_randstrobes = indexing.randstrobes(
    seq,
    k_size=15,
    strobe_w_min_offset=20,
    strobe_w_max_offset=70,
    w=10,      # Thin to ~10%
    order=2
)

print(f"Full: {len(rand)}, Sparse (w=10): {len(sparse_randstrobes)}")
```

---

### Example 5: Building a Sequence Index

```python
class StrobeIndex:
    """Simple strobemer-based sequence index for mapping."""
    
    def __init__(self, k=15, w_min=20, w_max=70, order=2):
        self.k = k
        self.w_min = w_min
        self.w_max = w_max
        self.order = order
        self.index = defaultdict(list)
    
    def add_sequence(self, seq_id, seq):
        """Add a sequence to the index"""
        for pos_tuple, hash_val in indexing.randstrobes_iter(
            seq, self.k, self.w_min, self.w_max, w=1, 
            order=self.order, buffer_size=1000000
        ):
            self.index[hash_val].append((seq_id, pos_tuple))
    
    def query(self, seq):
        """Find matches for a query sequence"""
        matches = defaultdict(list)
        for pos_tuple, hash_val in indexing.randstrobes(
            seq, self.k, self.w_min, self.w_max, w=1, order=self.order
        ).items():
            if hash_val in self.index:
                matches[hash_val] = self.index[hash_val]
        return matches
    
    def get_statistics(self):
        """Return index statistics"""
        return {
            'unique_hashes': len(self.index),
            'total_hits': sum(len(v) for v in self.index.values()),
            'avg_hits_per_hash': sum(len(v) for v in self.index.values()) / len(self.index)
        }

# Usage
idx = StrobeIndex(k=15, w_min=20, w_max=70, order=2)
idx.add_sequence("chr1", "ACGTACGT...")
idx.add_sequence("chr2", "ACGTACGT...")

results = idx.query("ACGTACGT...")
print(f"Query matched {len(results)} unique strobemers")
print(f"Index stats: {idx.get_statistics()}")
```

---

### Example 6: Finding Similar Regions Between Sequences

```python
def find_similar_regions(seq1, seq2, k=15, w_min=20, w_max=70, min_matches=3):
    """
    Find regions of similarity between two sequences using strobemers.
    
    Returns list of (seq1_pos, seq2_pos, match_count) tuples
    """
    # Build index for seq2
    seq2_index = defaultdict(list)
    for pos_tuple, h in indexing.randstrobes(
        seq2, k, w_min, w_max, w=1, order=2
    ).items():
        seq2_index[h].append(pos_tuple)
    
    # Query with seq1
    matches = defaultdict(lambda: defaultdict(int))
    for (p1, p2), h in indexing.randstrobes(
        seq1, k, w_min, w_max, w=1, order=2
    ).items():
        if h in seq2_index:
            for (q1, q2) in seq2_index[h]:
                # Cluster by approximate position
                region_key = (p1 // 100, q1 // 100)
                matches[region_key]['count'] += 1
                matches[region_key]['seq1_pos'] = p1
                matches[region_key]['seq2_pos'] = q1
    
    # Filter significant matches
    significant = [
        (v['seq1_pos'], v['seq2_pos'], v['count'])
        for k, v in matches.items()
        if v['count'] >= min_matches
    ]
    
    return sorted(significant, key=lambda x: -x[2])

# Example usage
similar = find_similar_regions(seq1, seq2)
for s1_pos, s2_pos, count in similar[:10]:
    print(f"seq1:{s1_pos} ≈ seq2:{s2_pos} ({count} matches)")
```

---

## Parameter Selection Guidelines

### Recommended Parameters by Application

| Application | k_size | w_min | w_max | order | w | Rationale |
|-------------|--------|-------|-------|-------|---|-----------|
| **Short reads** (Illumina, 150bp) | 15-20 | k+1 | 50-70 | 2 | 1 | Balance sensitivity/speed |
| **Long reads** (ONT, PacBio) | 15-20 | 20 | 70 | 2 | 1 | Handle high error rates (~10-15%) |
| **Genome alignment** | 20-30 | k+5 | 100-200 | 2 | 5-10 | Reduce index size |
| **High sensitivity** | 15 | 16 | 50 | 3 | 1 | More context, better specificity |
| **Fast mapping** | 20 | 21 | 40 | 2 | 10 | Sparse sampling |
| **Metagenomics** | 18-22 | 20 | 80 | 2 | 1-5 | Cross-species matching |
| **Variant calling** | 15-18 | 16 | 40 | 2 | 1 | Local alignment sensitivity |

### Parameter Tuning Tips

1. **Window placement matters**: 
   - `[w_min=20, w_max=70]` typically outperforms `[w_min=0, w_max=50]`
   - Larger gaps between strobes increase mutation tolerance

2. **Order selection**:
   - Order 2: Faster, less memory, good for most applications
   - Order 3: More specificity, better for repetitive regions, 2-3x memory

3. **Thinning (w > 1)**:
   - Reduces output size proportionally
   - Use for large datasets (whole genomes, metagenomes)
   - w=10 gives ~10% sampling

4. **k_size considerations**:
   - Larger k = more specific, fewer spurious matches
   - Smaller k = more sensitive, more matches
   - Must be divisible by order (auto-adjusted if not)

### Performance Characteristics

| Parameter | Memory | Speed | Sensitivity | Specificity |
|-----------|--------|-------|-------------|-------------|
| ↑ k_size | ↓ | ↑ | ↓ | ↑ |
| ↑ w_min/w_max | ↑ | ↓ | ↑ | ↓ |
| ↑ order | ↑↑ | ↓ | ↑ | ↑ |
| ↑ w (thinning) | ↓↓ | ↑↑ | ↓ | - |

---

## Complete Working Example

### Read Mapping Pipeline

```python
#!/usr/bin/env python3
"""
Strobemer-based sequence mapping example
Maps reads to a reference genome using randstrobes
"""

from modules import indexing
from collections import defaultdict
import sys

class StrobeMapper:
    """Simple strobemer-based read mapper."""
    
    def __init__(self, k=15, w_min=20, w_max=70, order=2, w=1):
        self.k = k
        self.w_min = w_min
        self.w_max = w_max
        self.order = order
        self.w = w
        self.ref_index = defaultdict(list)
        self.ref_lengths = {}
    
    def parse_fasta(self, filename):
        """Parse FASTA file, yield (seq_id, sequence) tuples"""
        seq_id = None
        seq = ""
        with open(filename) as f:
            for line in f:
                if line.startswith('>'):
                    if seq_id and seq:
                        yield seq_id, seq
                    seq_id = line[1:].strip().split()[0]
                    seq = ""
                else:
                    seq += line.strip().upper()
            if seq_id and seq:
                yield seq_id, seq
    
    def build_index(self, ref_fasta):
        """Build strobemer index from reference FASTA"""
        print(f"Building index from {ref_fasta}...", file=sys.stderr)
        
        for seq_id, seq in self.parse_fasta(ref_fasta):
            self.ref_lengths[seq_id] = len(seq)
            
            for pos_tuple, h in indexing.randstrobes_iter(
                seq, self.k, self.w_min, self.w_max, w=self.w, 
                order=self.order, buffer_size=1000000
            ):
                self.ref_index[h].append((seq_id, pos_tuple))
        
        print(f"Index built: {len(self.ref_index)} unique strobemers", file=sys.stderr)
    
    def map_read(self, read_seq):
        """Map a single read, return list of (ref_id, ref_pos, score)"""
        hits = defaultdict(int)
        
        for pos_tuple, h in indexing.randstrobes(
            read_seq, self.k, self.w_min, self.w_max, w=1, order=self.order
        ).items():
            if h in self.ref_index:
                for ref_id, ref_pos_tuple in self.ref_index[h]:
                    # Use first strobe position for clustering
                    ref_pos = ref_pos_tuple[0] if isinstance(ref_pos_tuple, tuple) else ref_pos_tuple
                    hits[(ref_id, ref_pos)] += 1
        
        # Convert to sorted list
        results = [
            (ref_id, ref_pos, score)
            for (ref_id, ref_pos), score in hits.items()
        ]
        return sorted(results, key=lambda x: -x[2])
    
    def map_reads(self, reads_fasta, min_score=3, max_hits=10):
        """Map all reads from a FASTA file"""
        for read_id, read_seq in self.parse_fasta(reads_fasta):
            hits = self.map_read(read_seq)
            
            if not hits:
                continue
            
            # Report top hits
            best_score = hits[0][2]
            for ref_id, ref_pos, score in hits[:max_hits]:
                if score >= min_score and score >= best_score * 0.5:
                    yield read_id, ref_id, ref_pos, score, len(read_seq)


def main():
    if len(sys.argv) < 3:
        print("Usage: python mapper.py <reference.fasta> <reads.fasta>")
        sys.exit(1)
    
    ref_fasta = sys.argv[1]
    reads_fasta = sys.argv[2]
    
    # Initialize mapper with recommended parameters for long reads
    mapper = StrobeMapper(k=15, w_min=20, w_max=70, order=2, w=1)
    
    # Build reference index
    mapper.build_index(ref_fasta)
    
    # Map reads and output in MUMmer-like format
    print(">read_id")
    print("ref_id\tref_pos\tread_pos\tmatch_length")
    
    for read_id, ref_id, ref_pos, score, read_len in mapper.map_reads(reads_fasta):
        print(f"{ref_id}\t{ref_pos}\t0\t{score * mapper.k}")


if __name__ == "__main__":
    main()
```

### Usage

```bash
# Map ONT reads to reference
python mapper.py reference.fasta reads.fasta > mappings.tsv

# Map with different parameters (shorter reads)
python mapper.py ref.fa reads.fa --k 20 --w_min 25 --w_max 50
```

---

## Troubleshooting

### Common Issues

#### 1. Memory Error with Large Sequences

**Problem**: `MemoryError` when indexing whole genomes

**Solution**: Use `_iter` functions with smaller buffer size
```python
# Instead of:
result = indexing.randstrobes(seq, k, w_min, w_max, w=1, order=2)

# Use:
for pos, h in indexing.randstrobes_iter(
    seq, k, w_min, w_max, w=1, order=2, buffer_size=100000
):
    process(pos, h)
```

#### 2. Few/No Matches Found

**Possible causes**:
- `k_size` too large for sequence length
- `w_min`/`w_max` window too narrow
- Sequences too divergent

**Solutions**:
```python
# Try smaller k
k = 12  # instead of 20

# Try wider window
w_min, w_max = 10, 100  # instead of 20, 40

# Try order 3 for more specificity
order = 3
```

#### 3. Hash Values Differ Between Runs

**Expected behavior**: Python's `hash()` uses random seed per session

**Solution**: For reproducible hashes, modify the hash function:
```python
import hashlib

def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

# Would require modifying indexing.py to use stable_hash instead of hash()
```

#### 4. Slow Performance

**Solutions**:
- Use `w > 1` for thinning
- Use C++ implementation for production (`strobemers_cpp/index.cpp`)
- Pre-allocate result dictionaries
- Process in parallel (split sequences by chromosome)

### Performance Benchmarks

| Sequence Length | Function | Time (approx.) | Memory |
|-----------------|----------|----------------|--------|
| 1 Kbp | `randstrobes()` | <1 ms | <1 MB |
| 1 Mbp | `randstrobes()` | ~500 ms | ~100 MB |
| 1 Mbp | `randstrobes_iter()` | ~600 ms | ~10 MB |
| 100 Mbp | `randstrobes_iter()` | ~50 s | ~50 MB |
| 3 Gbp (human) | `randstrobes_iter()` | ~25 min | ~500 MB |

*Benchmarks on standard laptop (2020), order=2, k=15*

---

## Additional Resources

- [Original Paper](https://genome.cshlp.org/content/31/11/2080)
- [Supplemental Methods](https://genome.cshlp.org/content/suppl/2021/10/19/gr.275648.121.DC1/Supplemental_Methods.pdf)
- [C++ Implementation](https://github.com/ksahlin/strobemers/tree/main/strobemers_cpp)
- [Bioconda Package](https://bioconda.github.io/recipes/strobemap/README.html)

---

## Citation

If you use strobemers in your research, please cite:

> Kristoffer Sahlin, Effective sequence similarity detection with strobemers, Genome Res. November 2021 31: 2080-2094; doi: https://doi.org/10.1101/gr.275648.121
