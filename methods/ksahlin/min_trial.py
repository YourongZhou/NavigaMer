import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from strobemers.modules import indexing
from collections import defaultdict


def generate_strobes(seq, k_size=15, w_min=20, w_max=70, w=1, order=2):
    """
    Generate strobemers from a DNA sequence.
    
    Parameters
    ----------
    seq : str
        DNA sequence (e.g., "ACGTACGT...")
    k_size : int
        Total strobemer length in nucleotides
    w_min : int
        Minimum window offset (distance from first strobe to window start)
    w_max : int
        Maximum window offset (distance from first strobe to window end)
    w : int
        Thinning window (w=1: all strobemers, w>1: sparse sampling)
    order : int
        Strobemer order (2 or 3)
    
    Returns
    -------
    dict
        Dictionary with (position_tuple) as keys and hash_value as values
        - order=2: keys are (p1, p2)
        - order=3: keys are (p1, p2, p3)
    """
    return indexing.randstrobes(seq, k_size, w_min, w_max, w, order=order)


def process_fasta_sequences(fasta_file, k_size=15, w_min=20, w_max=70, w=1, order=2):
    """
    Process all sequences in a FASTA file and generate strobemers.
    
    Parameters
    ----------
    fasta_file : str
        Path to FASTA file
    k_size : int
        Total strobemer length
    w_min : int
        Minimum window offset
    w_max : int
        Maximum window offset
    w : int
        Thinning window
    order : int
        Strobemer order
    
    Returns
    -------
    dict
        Nested dictionary: {seq_id: {position_tuple: hash_value}}
    """
    results = {}
    
    # Parse FASTA file
    seq_id = None
    seq = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if seq_id and seq:
                    results[seq_id] = generate_strobes(
                        seq, k_size, w_min, w_max, w, order=order
                    )
                # New sequence header
                seq_id = line[1:].split()[0]  # Get first word after '>'
                seq = ""
            else:
                seq += line.upper()
        
        # Don't forget the last sequence
        if seq_id and seq:
            results[seq_id] = generate_strobes(
                seq, k_size, w_min, w_max, w, order=order
            )
    
    return results


def print_strobe_summary(results):
    """Print summary statistics for strobemer results."""
    print("=" * 60)
    print("STROBEMER GENERATION SUMMARY")
    print("=" * 60)
    print(f"{'Sequence ID':<50} {'Strobemers':>10}")
    print("-" * 60)
    
    total_strobes = 0
    for seq_id, strobes in results.items():
        n_strobes = len(strobes)
        total_strobes += n_strobes
        # Truncate long IDs for display
        display_id = seq_id[:47] + "..." if len(seq_id) > 50 else seq_id
        print(f"{display_id:<50} {n_strobes:>10}")
    
    print("-" * 60)
    print(f"{'TOTAL':<50} {total_strobes:>10}")
    print(f"{'Unique sequences':<50} {len(results):>10}")
    print("=" * 60)


def print_sample_strobes(results, n_samples=3):
    """Print sample strobemers from first few sequences."""
    print("\nSAMPLE STROBEMERS (first 3 per sequence):\n")
    
    count = 0
    for seq_id, strobes in results.items():
        if count >= 5:  # Limit to first 5 sequences
            break
            
        print(f"> {seq_id[:60]}...")
        print(f"  Total strobemers: {len(strobes)}")
        
        # Show first n_samples strobemers
        for i, ((positions), hash_val) in enumerate(list(strobes.items())[:n_samples]):
            if len(positions) == 2:
                p1, p2 = positions
                print(f"    #{i+1}: positions=({p1}, {p2}), hash={hash_val}")
            else:
                p1, p2, p3 = positions
                print(f"    #{i+1}: positions=({p1}, {p2}, {p3}), hash={hash_val}")
        
        print()
        count += 1


if __name__ == "__main__":
    # Path to the ONT SIRV reads data
    data_dir = Path(__file__).parent / "strobemers" / "data"
    fasta_file = data_dir / "ONT_sirv_cDNA_seqs.fasta"
    
    # Check if file exists
    if not fasta_file.exists():
        print(f"ERROR: File not found: {fasta_file}")
        sys.exit(1)
    
    print(f"Processing: {fasta_file}")
    print()
    
    # Process all sequences with default parameters (2, 15, 20, 70)
    results = process_fasta_sequences(
        fasta_file,
        k_size=15,
        w_min=20,
        w_max=70,
        w=1,
        order=2
    )
    
    # Print summary
    print_strobe_summary(results)
    
    # Print sample strobemers
    print_sample_strobes(results, n_samples=3)
    
    # Example: Access specific sequence results
    print("\n" + "=" * 60)
    print("EXAMPLE: Accessing specific sequence data")
    print("=" * 60)
    
    first_seq_id = list(results.keys())[0]
    first_seq_strobes = results[first_seq_id]
    
    print(f"\nSequence: {first_seq_id}")
    print(f"Number of strobemers: {len(first_seq_strobes)}")
    
    # Show first 5 strobemers with sequence context
    print("\nFirst 5 strobemers (with sequence extraction):")
    
    # We need to re-parse to get the sequence for demonstration
    with open(fasta_file, 'r') as f:
        seq = ""
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id == first_seq_id and seq:
                    break
                current_id = line[1:].split()[0]
                seq = ""
            elif current_id == first_seq_id:
                seq += line.upper()
    
    k = 15  # k_size
    for i, ((p1, p2), hash_val) in enumerate(list(first_seq_strobes.items())[:5]):
        strobe1_seq = seq[p1:p1 + k//2]
        strobe2_seq = seq[p2:p2 + k//2]
        print(f"  {i+1}. ({p1:4d}, {p2:4d}): {strobe1_seq}+{strobe2_seq}")
