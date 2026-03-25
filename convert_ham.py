"""
Batch convert all Pauli Hamiltonian datasets from HDF5 files in ham/ to NPZ files,
organized by transformation type (JW, BK, molec, parity).

Output structure:
    ham/JW/H2_4q.npz
    ham/BK/H2_4q.npz
    ham/parity/H2_4q.npz

Each .npz contains 'mat' and 'w' arrays compatible with Pauli.load().

Note: 'molec' type uses second-quantization (fermion) notation, NOT Pauli strings,
so it is saved in a different format with 'terms' and 'coeffs' arrays.
"""

import re
import os
import glob
import sys
import h5py
import numpy as np


PAULI_MAP = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def parse_complex_coeff(s: str) -> float:
    """Parse a coefficient string, handling various formats:
    - '(0.123+0j)'         -> 0.123
    - '(-0.123+0j)'        -> -0.123
    - '(-0-1.03e-08j)'     -> -0.0  (imaginary-only, treat as 0)
    - '0.123'              -> 0.123
    - '-0.123'             -> -0.123
    """
    s = s.strip()
    # Remove surrounding parens if present
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    # Handle complex numbers
    if "j" in s:
        try:
            c = complex(s)
            return c.real
        except ValueError:
            # Handle malformed like '-0-1.03e-08j'
            # Try to extract real part before the imaginary
            m = re.match(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", s)
            if m:
                return float(m.group(1))
            return 0.0
    # Plain float
    return float(s)


def parse_pauli_jw_bk(raw: str):
    """Parse JW/BK format: (coeff+0j) [X0 Y1 Z2 ...] +
    
    Returns (mat, w) numpy arrays.
    """
    # Match both formats:
    #   (0.123+0j) [X0 Z1]       — JW/BK with parens
    #   0.123 [X0 Z1]            — parity without parens
    pattern = r"(?:\(([^)]+)\)|([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?))\s*\[([^\]]*)\]"
    terms = []
    for match in re.finditer(pattern, raw):
        coeff_str = match.group(1) if match.group(1) else match.group(2)
        coeff = parse_complex_coeff(coeff_str)
        ops_str = match.group(3).strip()
        if not ops_str:
            continue
        # Check if this is second-quantization (has ^ for creation operators)
        if "^" in ops_str:
            return None  # Not a Pauli string format
        pauli_map = {}
        n_q = 0
        for op in ops_str.split():
            gate = op[0]
            qubit = int(op[1:])
            pauli_map[qubit] = gate
            n_q = max(n_q, qubit + 1)
        terms.append((pauli_map, coeff, n_q))

    if not terms:
        return None

    n_q = max(t[2] for t in terms)
    n_p = len(terms)

    mat = np.zeros((n_p, n_q, 4), dtype=np.float64)
    w = np.zeros(n_p, dtype=np.float64)

    for i, (pauli_map, coeff, _) in enumerate(terms):
        w[i] = coeff
        for q in range(n_q):
            gate = pauli_map.get(q, "I")
            mat[i, q, PAULI_MAP[gate]] = 1.0

    return mat, w


def parse_molec(raw: str):
    """Parse molecular (second quantization) format: coeff [0^ 0^ 2 2] +
    
    Returns (coeffs, terms_list) where terms_list is a list of operator tuples.
    Each operator tuple is (index, is_creation).
    Saved as sparse representation.
    """
    pattern = r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\[([^\]]*)\]"
    coeffs = []
    terms = []
    for match in re.finditer(pattern, raw):
        coeff = float(match.group(1))
        ops_str = match.group(2).strip()
        if not ops_str:
            continue
        ops = []
        for token in ops_str.split():
            if token.endswith("^"):
                ops.append((int(token[:-1]), 1))  # creation
            else:
                ops.append((int(token), 0))        # annihilation
        coeffs.append(coeff)
        terms.append(ops)

    if not coeffs:
        return None

    # Convert to arrays: pad terms to max length
    max_ops = max(len(t) for t in terms)
    n_terms = len(terms)
    # ops_indices[i, j] = orbital index, ops_types[i, j] = 0/1 (annihilation/creation)
    # -1 for padding
    ops_indices = np.full((n_terms, max_ops), -1, dtype=np.int32)
    ops_types = np.full((n_terms, max_ops), -1, dtype=np.int32)
    ops_lengths = np.zeros(n_terms, dtype=np.int32)

    for i, ops in enumerate(terms):
        ops_lengths[i] = len(ops)
        for j, (idx, is_creation) in enumerate(ops):
            ops_indices[i, j] = idx
            ops_types[i, j] = is_creation

    return np.array(coeffs, dtype=np.float64), ops_indices, ops_types, ops_lengths


def main():
    ham_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ham")
    hdf5_files = sorted(glob.glob(os.path.join(ham_dir, "*.hdf5")))
    print(f"Found {len(hdf5_files)} HDF5 files in {ham_dir}")

    # Clean previous incomplete results
    force = "--force" in sys.argv

    total = 0
    skipped = 0
    errors = []

    for fpath in hdf5_files:
        molname = os.path.basename(fpath).replace(".hdf5", "")
        print(f"\n=== {molname} ===")
        with h5py.File(fpath, "r") as f:
            for key in sorted(f.keys()):
                m = re.match(r"ham_([A-Za-z]+)-?(\d+)", key)
                if not m:
                    print(f"  SKIP: {key} (unrecognized format)")
                    skipped += 1
                    continue

                ham_type = m.group(1)
                n_qubits = m.group(2)

                out_dir = os.path.join(ham_dir, ham_type)
                out_path = os.path.join(out_dir, f"{molname}_{n_qubits}q.npz")

                if os.path.exists(out_path) and not force:
                    total += 1
                    continue

                raw = f[key][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                try:
                    if ham_type == "molec":
                        result = parse_molec(raw)
                        if result is None:
                            raise ValueError("No fermion terms found")
                        coeffs, ops_indices, ops_types, ops_lengths = result
                        os.makedirs(out_dir, exist_ok=True)
                        np.savez(out_path,
                                 coeffs=coeffs,
                                 ops_indices=ops_indices,
                                 ops_types=ops_types,
                                 ops_lengths=ops_lengths)
                        total += 1
                        print(f"  ✓ {key} → {ham_type}/{molname}_{n_qubits}q.npz  "
                              f"(n_terms={len(coeffs)}, max_ops={ops_indices.shape[1]})")
                    else:
                        result = parse_pauli_jw_bk(raw)
                        if result is None:
                            raise ValueError("No Pauli terms found")
                        mat, w = result
                        os.makedirs(out_dir, exist_ok=True)
                        np.savez(out_path, mat=mat, w=w)
                        total += 1
                        print(f"  ✓ {key} → {ham_type}/{molname}_{n_qubits}q.npz  "
                              f"(n_p={mat.shape[0]}, n_q={mat.shape[1]})")
                except Exception as e:
                    errors.append(f"{molname}/{key}: {e}")
                    print(f"  ERROR: {key}: {e}")

    print(f"\n{'='*50}")
    print(f"Done! Total: {total} datasets.")
    if skipped:
        print(f"Skipped: {skipped}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("No errors!")


if __name__ == "__main__":
    main()
