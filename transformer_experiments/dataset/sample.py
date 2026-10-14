"""Balanced-cell dataset sampler.

Samples sentences via `generate.generate_sentence` and arranges them into a
6-D cube so analyses can slice by condition:

    tokens[tg, subj, n_attr, n_mis, last_mis, k]

The first 5 axes identify a Linzen-style cell; the 6th axis is the per-cell
sample index. 38 of the 324 nominal cells are structurally valid; the rest
are impossible combinations and are padded with sentinels + a validity mask.

Hard balance (50/50 subject number, 50/50 last-attractor mismatch among
attractor-bearing cells) is a free consequence of giving every valid cell
the same number of samples.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .generate import generate_sentence


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------

TEMPLATE_GROUPS: list[str] = [
    "simple", "pp", "relative_clause", "object_rc", "copula", "pp_copula",
]
SUBJECT_NUMBERS: list[str] = ["singular", "plural"]
LAST_MISMATCH_VALUES: list[Optional[bool]] = [None, False, True]

CUBE_AXES_SHAPE: tuple[int, int, int, int, int] = (
    len(TEMPLATE_GROUPS), len(SUBJECT_NUMBERS), 3, 3, 3
)

# (template_group, n_attractors) -> name registered in generate.TEMPLATES.
# relative_clause is the only group whose generator covers more than one
# n_attractors value; rejection sampling filters its branch.
TG_NATTR_TO_GEN: dict[tuple[str, int], str] = {
    ("simple",          0): "simple",
    ("pp",              1): "pp_x1",
    ("pp",              2): "pp_x2",
    ("relative_clause", 0): "relative_clause",
    ("relative_clause", 1): "relative_clause",
    ("object_rc",       1): "object_rc",
    ("copula",          0): "copula",
    ("pp_copula",       1): "pp_copula_x1",
    ("pp_copula",       2): "pp_copula_x2",
}

# Structurally valid (template_group, n_attr, n_mis, last_mis) patterns.
# These are multiplied by {singular, plural} to get the 38 cells.
VALID_PATTERNS: list[tuple[str, int, int, Optional[bool]]] = [
    ("simple",          0, 0, None),
    ("pp",              1, 0, False),
    ("pp",              1, 1, True),
    ("pp",              2, 0, False),
    ("pp",              2, 1, False),
    ("pp",              2, 1, True),
    ("pp",              2, 2, True),
    ("relative_clause", 0, 0, None),
    ("relative_clause", 1, 0, False),
    ("relative_clause", 1, 1, True),
    ("object_rc",       1, 0, False),
    ("object_rc",       1, 1, True),
    ("copula",          0, 0, None),
    ("pp_copula",       1, 0, False),
    ("pp_copula",       1, 1, True),
    ("pp_copula",       2, 0, False),
    ("pp_copula",       2, 1, False),
    ("pp_copula",       2, 1, True),
    ("pp_copula",       2, 2, True),
]


@dataclass(frozen=True)
class Cell:
    """One cell in the first 5 axes of the cube."""
    template_group: str
    subject_number: str
    n_attractors: int
    n_mismatched_attractors: int
    last_attractor_mismatch: Optional[bool]
    generator_name: str

    @property
    def index(self) -> tuple[int, int, int, int, int]:
        return (
            TEMPLATE_GROUPS.index(self.template_group),
            SUBJECT_NUMBERS.index(self.subject_number),
            self.n_attractors,
            self.n_mismatched_attractors,
            LAST_MISMATCH_VALUES.index(self.last_attractor_mismatch),
        )


def enumerate_valid_cells() -> list[Cell]:
    cells: list[Cell] = []
    for tg, n_attr, n_mis, last_mis in VALID_PATTERNS:
        gen = TG_NATTR_TO_GEN[(tg, n_attr)]
        for subj in SUBJECT_NUMBERS:
            cells.append(Cell(
                template_group=tg,
                subject_number=subj,
                n_attractors=n_attr,
                n_mismatched_attractors=n_mis,
                last_attractor_mismatch=last_mis,
                generator_name=gen,
            ))
    return cells


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

# Fixed-width int columns copied straight from the record.
INT_COLUMNS: list[str] = [
    "length", "subject_position", "verb_position",
    "n_attractors", "n_mismatched_attractors",
]

# Object columns copied straight from the record (or synthesized).
OBJECT_COLUMNS: list[str] = [
    "tokens", "subject_number", "template", "template_group",
    "intervening_phrase_type", "verb_correct", "verb_incorrect",
    "attractor_numbers", "last_attractor_mismatch", "content_tags",
]


def _empty_cube(n_per_cell: int) -> dict:
    """Allocate the 6-D arrays with sentinel fills."""
    sample_shape = CUBE_AXES_SHAPE + (n_per_cell,)
    cube: dict = {}
    for col in INT_COLUMNS:
        cube[col] = np.zeros(sample_shape, dtype=np.int32)
    for col in OBJECT_COLUMNS:
        cube[col] = np.full(sample_shape, None, dtype=object)
    cube["cell_valid"] = np.zeros(CUBE_AXES_SHAPE, dtype=bool)
    cube["cell_count"] = np.zeros(CUBE_AXES_SHAPE, dtype=np.int32)
    # Axis labels, saved as 0-d object arrays for round-tripping.
    cube["axis_template_group"] = np.array(TEMPLATE_GROUPS, dtype=object)
    cube["axis_subject_number"] = np.array(SUBJECT_NUMBERS, dtype=object)
    cube["axis_last_mismatch"] = np.array(LAST_MISMATCH_VALUES, dtype=object)
    return cube


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------

def sample_balanced(
    n_per_cell: Optional[int] = None,
    total_samples: Optional[int] = None,
    seed: Optional[int] = None,
    max_attempts_per_cell: Optional[int] = None,
) -> dict:
    """Rejection-sample a balanced 6-D cube of sentence records.

    Pass exactly one of `n_per_cell` or `total_samples`. With `total_samples`,
    `n_per_cell = total_samples // n_valid_cells`.
    """
    cells = enumerate_valid_cells()
    n_cells = len(cells)
    assert n_cells == 38, f"expected 38 valid cells, got {n_cells}"

    if (n_per_cell is None) == (total_samples is None):
        raise ValueError("pass exactly one of n_per_cell or total_samples")
    if total_samples is not None:
        if total_samples < n_cells:
            raise ValueError(
                f"total_samples={total_samples} too small for {n_cells} cells"
            )
        n_per_cell = total_samples // n_cells
    assert n_per_cell is not None and n_per_cell > 0

    if max_attempts_per_cell is None:
        # Every valid cell is reachable with p >= ~1/8 under the uniform
        # grammar choices, so 50x is a comfortable safety net.
        max_attempts_per_cell = max(10_000, 50 * n_per_cell)

    rng = random.Random(seed)
    cube = _empty_cube(n_per_cell)

    for cell in cells:
        idx = cell.index
        cube["cell_valid"][idx] = True
        cube["cell_count"][idx] = n_per_cell
        _fill_cell(cube, cell, n_per_cell, rng, max_attempts_per_cell)

    return cube


def _fill_cell(cube: dict, cell: Cell, n_per_cell: int,
               rng: random.Random, max_attempts: int) -> None:
    """Rejection-sample `n_per_cell` records for one cell."""
    k = 0
    attempts = 0
    n_attr = cell.n_attractors
    n_mis = cell.n_mismatched_attractors
    target_last_mis = cell.last_attractor_mismatch
    seen = set()

    while k < n_per_cell:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"could not fill cell {cell} after {attempts} attempts; "
                f"got {k}/{n_per_cell} samples"
            )
        attempts += 1
        rec = generate_sentence(cell.generator_name, cell.subject_number, rng)

        tok_key = tuple(rec["tokens"])
        if tok_key in seen:
            continue
    
        if rec["n_attractors"] != n_attr:
            continue
        rec_n_mis = sum(
            1 for a in rec["attractor_numbers"] if a != rec["subject_number"]
        )
        if rec_n_mis != n_mis:
            continue
        if rec["last_attractor_mismatch"] != target_last_mis:
            continue

        seen.add(tok_key)
        _write_record(cube, rec, (*cell.index, k), cell.template_group,
                      rec_n_mis)
        k += 1


def _write_record(cube: dict, rec: dict, slot: tuple[int, ...],
                  template_group: str, n_mis: int) -> None:
    """Place one record's fields at the given 6-D slot."""
    cube["length"][slot] = rec["length"]
    cube["subject_position"][slot] = (
        rec["subject_position"] if rec["subject_position"] is not None else -1
    )
    cube["verb_position"][slot] = (
        rec["verb_position"] if rec["verb_position"] is not None else -1
    )
    cube["n_attractors"][slot] = rec["n_attractors"]
    cube["n_mismatched_attractors"][slot] = n_mis

    cube["tokens"][slot] = list(rec["tokens"])
    cube["subject_number"][slot] = rec["subject_number"]
    cube["template"][slot] = rec["template"]
    cube["template_group"][slot] = template_group
    cube["intervening_phrase_type"][slot] = rec["intervening_phrase_type"]
    cube["verb_correct"][slot] = rec["verb_correct"]
    cube["verb_incorrect"][slot] = rec["verb_incorrect"]
    cube["attractor_numbers"][slot] = list(rec["attractor_numbers"])
    cube["last_attractor_mismatch"][slot] = rec["last_attractor_mismatch"]
    cube["content_tags"][slot] = list(rec["content_tags"])


# ---------------------------------------------------------------------------
# Splitting along the sample axis (stratified-by-cell by construction)
# ---------------------------------------------------------------------------

# All columns with the per-sample `k` axis.
_PER_SAMPLE_COLUMNS: list[str] = INT_COLUMNS + OBJECT_COLUMNS


def split_cube(cube: dict, splits: dict[str, float],
               seed: Optional[int] = None) -> dict[str, dict]:
    """Split a cube along the sample axis per cell. Each returned sub-cube has
    the same 5 structural axes; only the last (sample) axis is shortened.

    `splits` maps a name to a fraction; fractions should sum to 1.0 (within
    floating-point tolerance). Each cell is partitioned deterministically with
    the same proportions, so balance is preserved in every sub-cube.

    If `seed` is given, each cell's samples are shuffled independently before
    the split (useful if generation order correlates with sampling attempt
    count — in practice samples within a cell are already i.i.d., but shuffling
    makes probe train/eval splits reproducible across code changes).
    """
    total = sum(splits.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    n_per_cell = int(cube["cell_count"].max())
    names = list(splits)
    # Integer counts per split; distribute the remainder to earlier splits.
    counts = [int(splits[name] * n_per_cell) for name in names]
    remainder = n_per_cell - sum(counts)
    for i in range(remainder):
        counts[i] += 1
    assert sum(counts) == n_per_cell, (counts, n_per_cell)

    # Per-cell permutations (or identity ranges).
    rng = np.random.default_rng(seed)
    valid_idx = np.argwhere(cube["cell_valid"])
    perms: dict[tuple, np.ndarray] = {}
    for idx in valid_idx:
        key = tuple(int(x) for x in idx)
        perms[key] = (rng.permutation(n_per_cell) if seed is not None
                      else np.arange(n_per_cell))

    # Allocate each sub-cube with the right sample-axis size.
    sub_cubes: dict[str, dict] = {}
    offsets = np.cumsum([0] + counts)
    for name, size, start in zip(names, counts, offsets[:-1]):
        sub = _empty_cube(size)
        # Carry over mask + axis labels verbatim.
        sub["cell_valid"] = cube["cell_valid"].copy()
        sub["cell_count"] = np.where(cube["cell_valid"], size, 0).astype(np.int32)
        for k in ("axis_template_group", "axis_subject_number",
                  "axis_last_mismatch"):
            sub[k] = cube[k].copy()
        sub_cubes[name] = sub

        for cell_key, perm in perms.items():
            cell_slice = perm[start:start + size]
            for col in _PER_SAMPLE_COLUMNS:
                src = cube[col][cell_key]           # shape (n_per_cell,)
                sub[col][cell_key] = src[cell_slice]
    return sub_cubes


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_npz(cube: dict, path: str | Path) -> None:
    """Compressed NPZ with object arrays for variable-shape fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cube)


def load_npz(path: str | Path) -> dict:
    """Load a cube saved by save_npz. Requires allow_pickle for object cols."""
    with np.load(path, allow_pickle=True) as loaded:
        return {k: loaded[k] for k in loaded.files}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_cube(cube: dict) -> None:
    """Assert balance and structural invariants."""
    expected_sample_shape = CUBE_AXES_SHAPE + (cube["cell_count"].max(),)
    assert cube["tokens"].shape == expected_sample_shape, cube["tokens"].shape
    assert cube["cell_valid"].sum() == 38

    # Every valid pattern is marked valid; every invalid cell is marked invalid.
    expected_valid = np.zeros(CUBE_AXES_SHAPE, dtype=bool)
    for cell in enumerate_valid_cells():
        expected_valid[cell.index] = True
    assert np.array_equal(cube["cell_valid"], expected_valid)

    # Content checks over valid cells only.
    valid_mask = cube["cell_valid"]
    subj_axis = cube["subject_number"]  # (6,2,3,3,3,N)
    last_axis = cube["last_attractor_mismatch"]

    n_singular = n_plural = 0
    n_last_true = n_last_false = 0
    for idx in np.argwhere(valid_mask):
        tg, subj, na, nm, lm = idx
        n = int(cube["cell_count"][tg, subj, na, nm, lm])
        for k in range(n):
            slot = (tg, subj, na, nm, lm, k)
            assert subj_axis[slot] is not None, f"null record at {slot}"
            if subj_axis[slot] == "singular":
                n_singular += 1
            else:
                n_plural += 1
            v = last_axis[slot]
            if v is True:
                n_last_true += 1
            elif v is False:
                n_last_false += 1

    assert abs(n_singular - n_plural) <= 1, (n_singular, n_plural)
    assert abs(n_last_true - n_last_false) <= 1, (n_last_true, n_last_false)

    # Every invalid cell slot is None (object) / 0 (int).
    invalid_mask = ~valid_mask
    for idx in np.argwhere(invalid_mask):
        tg, subj, na, nm, lm = idx
        # Spot-check: subject_number and tokens should be None across all k.
        assert np.all(cube["subject_number"][tg, subj, na, nm, lm] == None), idx
        assert np.all(cube["tokens"][tg, subj, na, nm, lm] == None), idx


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    t0 = time.time()
    cube = sample_balanced(n_per_cell=10, seed=0)
    dt = time.time() - t0
    print(f"sampled 38 cells x 10 = 380 records in {dt:.2f}s")
    verify_cube(cube)
    print("verify_cube OK")

    # Spot-check a slice:
    tg = TEMPLATE_GROUPS.index("pp")
    subj = SUBJECT_NUMBERS.index("plural")
    print("\nexample [pp, plural, n_attr=2, n_mis=1, last=False, k=0]:")
    print("  tokens:", cube["tokens"][tg, subj, 2, 1, 1, 0])
    print("  verb_correct:", cube["verb_correct"][tg, subj, 2, 1, 1, 0])

    print("\nexample [simple, singular, n_attr=0, n_mis=0, last=None, k=0]:")
    print("  tokens:",
          cube["tokens"][TEMPLATE_GROUPS.index("simple"), 0, 0, 0, 0, 0])

    print("\ninvalid slot [simple, singular, n_attr=2, n_mis=0, last=None]:")
    print("  tokens:", cube["tokens"][0, 0, 2, 0, 0, 0])
