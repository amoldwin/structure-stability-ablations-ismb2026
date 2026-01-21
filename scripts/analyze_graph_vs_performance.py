#!/usr/bin/env python3
"""
analyze_graph_vs_performance_tmalign_global_strict_multiSphere.py

Global TM-align mapping + Kabsch superposition (strict; no silent failures)
and mutation-centered multi-radius sphere RMSD.

Key metrics computed for each WT vs MUT structure:
  - ca_rmsd: RMSD over aligned residues after Kabsch superposition (Å)
  - ca_rmsd_sphere_{R}A: RMSD over aligned residues within R Å of mutation site (Å),
    for each radius in --sphere_radii (default: 8,10,12,15)
  - ca_pair_mean_abs_diff: mean |Δ pairwise CA distance| over aligned residues (Å)
  - n_contact_gained / n_contact_lost: contact changes over aligned residues

Replicate variability metric (UPDATED):
  - rep_pair_rmsd_mean/std/max: pairwise global RMSD among replicate mutant structures
    (global TM-align mapping + Kabsch)
  - rep_pair_rmsd_sphere_{R}A_mean/std/max: pairwise sphere RMSD (same mapping + Kabsch),
    for each radius R, centered at mutation position mut_pos0 in the "A" replicate.

Strictness:
  - No fallback alignments.
  - No per-variant try/except. Any issue aborts immediately.
  - Cache loads with allow_pickle=False; legacy caches with object dtype are
    deterministically rebuilt (without pickle).

Usage:
  python analyze_graph_vs_performance_tmalign_global_strict_multiSphere.py \
    --features_dir /projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/ \
    --out_dir analysis_outputs_graph_tmalign_global_strict_multiSphere \
    --tmalign_bin /home/amoldwin/bin/TMalign \
    --sphere_radii 8,10,12,15
"""

import os
import re
import argparse
import subprocess
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1


# -----------------------------
# Constants / defaults
# -----------------------------
PARSER = PDBParser(QUIET=True)

NON_STANDARD_SUBSTITUTIONS = {
    "GLH": "GLU", "ASH": "ASP", "CYX": "CYS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
}

DEFAULT_WT_PDB = "wt_esmf.pdb"
DEFAULT_MUT_E2E_ESMF = "mut_esmf.pdb"
DEFAULT_MUT_ROSETTA = "mut_rosetta.pdb"


# -----------------------------
# PDB parsing (sequence + CA)
# -----------------------------
def three_to_one_aug(three: str) -> str:
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    try:
        return protein_letters_3to1[three]
    except Exception:
        return "X"


def parse_pdb_seq_and_ca(pdb_path: str) -> Tuple[str, np.ndarray]:
    """
    Parse PDB -> (seq, ca_coords).
    Chooses chain with most residues.
    Requires backbone N/CA/C present.
    """
    if pdb_path is None or (not os.path.exists(pdb_path)):
        raise FileNotFoundError(f"PDB missing: {pdb_path}")

    structure = PARSER.get_structure(None, pdb_path)
    model = structure[0]

    best_chain = None
    best_cnt = -1
    for ch in model:
        cnt = sum(1 for r in ch if r.get_id()[0] == " ")
        if cnt > best_cnt:
            best_cnt = cnt
            best_chain = ch
    if best_chain is None:
        raise ValueError(f"No chain found in PDB: {pdb_path}")

    seq: List[str] = []
    ca: List[np.ndarray] = []
    for res in best_chain:
        hetflag, resseq, icode = res.get_id()
        if hetflag != " ":
            continue

        aa1 = three_to_one_aug(res.get_resname())

        # require backbone atoms
        if not (res.has_id("N") and res.has_id("CA") and res.has_id("C")):
            continue

        ca_coord = res["CA"].get_coord()
        seq.append(aa1)
        ca.append(np.asarray(ca_coord, dtype=float))

    if len(seq) == 0:
        raise ValueError(f"No residues parsed (after filtering) from: {pdb_path}")

    return "".join(seq), np.vstack(ca)


# -----------------------------
# TM-align mapping (pairs)
# -----------------------------
def _extract_tmalign_aln_strings(stdout_text: str) -> Tuple[str, str]:
    """
    Extract two gapped alignment strings from TM-align stdout.
    Uses last two lines matching [A-Za-z-]+.
    """
    lines = [ln.strip() for ln in stdout_text.splitlines()]
    aln_lines = [ln for ln in lines if re.fullmatch(r"[A-Za-z\-]+", ln) and len(ln) >= 20]
    if len(aln_lines) < 2:
        tail = "\n".join(lines[-60:])
        raise ValueError(
            "Could not find alignment strings in TM-align output.\n"
            "Last ~60 lines of stdout:\n" + tail
        )
    aln1, aln2 = aln_lines[-2].upper(), aln_lines[-1].upper()
    if len(aln1) != len(aln2):
        raise ValueError("TM-align alignment strings have different lengths.")
    return aln1, aln2


def build_index_pairs_from_alignment(aln1: str, aln2: str) -> List[Tuple[int, int]]:
    """Return (i,j) residue indices for ungapped columns; indices are 0-based."""
    i = j = 0
    pairs: List[Tuple[int, int]] = []
    for c1, c2 in zip(aln1, aln2):
        g1 = (c1 == "-")
        g2 = (c2 == "-")
        if (not g1) and (not g2):
            pairs.append((i, j))
        if not g1:
            i += 1
        if not g2:
            j += 1
    return pairs


def run_tmalign_pairs(tmalign_bin: str, pdb_a: str, pdb_b: str) -> Tuple[List[Tuple[int, int]], str, str]:
    """
    Run TM-align; return (pairs, aln1, aln2).
    No -m. Superposition is done via Kabsch downstream.
    """
    cmd = [tmalign_bin, pdb_a, pdb_b]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"TM-align failed (code {p.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr (first 300 chars): {p.stderr.strip()[:300]}"
        )
    aln1, aln2 = _extract_tmalign_aln_strings(p.stdout)
    pairs = build_index_pairs_from_alignment(aln1, aln2)
    if len(pairs) < 5:
        raise ValueError(f"TM-align produced too few aligned pairs: {len(pairs)}")
    return pairs, aln1, aln2


# -----------------------------
# Rigid superposition (Kabsch)
# -----------------------------
def kabsch_rt(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute R,t that best superposes P onto Q (least-squares).
    P,Q: (N,3), N>=3
    """
    if P.shape != Q.shape or P.shape[0] < 3:
        raise ValueError("Kabsch requires >=3 paired points with matching shape.")

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    R = (V @ Wt).T

    if np.linalg.det(R) < 0:
        Wt[-1, :] *= -1
        R = (V @ Wt).T

    t = Qc - (R @ Pc)
    return R, t


def apply_rt(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (X @ R.T) + t[None, :]


# -----------------------------
# Structural metrics
# -----------------------------
def ca_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape != B.shape or A.shape[0] == 0:
        return np.nan
    D = A - B
    return float(np.sqrt((D * D).sum() / A.shape[0]))


def pairwise_dist_mat(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def ca_pair_mean_abs_diff(ca_a: np.ndarray, ca_b: np.ndarray) -> float:
    Da = pairwise_dist_mat(ca_a)
    Db = pairwise_dist_mat(ca_b)
    return float(np.mean(np.abs(Da - Db)))


def contact_graph_from_ca(ca: np.ndarray, thresh: float) -> np.ndarray:
    D = pairwise_dist_mat(ca)
    C = (D < thresh).astype(np.uint8)
    np.fill_diagonal(C, 0)
    return C


def contact_delta_counts(C_a: np.ndarray, C_b: np.ndarray) -> Tuple[int, int]:
    gained = np.logical_and(C_b == 1, C_a == 0)
    lost = np.logical_and(C_b == 0, C_a == 1)
    n_gained = int(gained.sum() // 2)
    n_lost = int(lost.sum() // 2)
    return n_gained, n_lost


def sphere_rmsd_about_index(
    a_sup: np.ndarray,
    b_aln: np.ndarray,
    a_idx: List[int],
    center_pos0: int,
    radius_A: float,
) -> float:
    """
    RMSD on aligned residues whose (superposed) A coords are within radius_A
    of the center residue (by index center_pos0 in the original A CA indexing).

    a_sup, b_aln: (L,3) aligned coordinates
    a_idx: length L list mapping aligned positions -> A CA index
    center_pos0: 0-based center residue index in A CA indexing
    """
    if a_sup.shape != b_aln.shape or a_sup.shape[0] != len(a_idx):
        raise ValueError("sphere_rmsd_about_index: shape mismatch.")

    k = None
    for aligned_pos, ai in enumerate(a_idx):
        if ai == center_pos0:
            k = aligned_pos
            break
    if k is None:
        return np.nan

    center = a_sup[k]
    d = np.linalg.norm(a_sup - center, axis=1)
    mask = d <= radius_A
    if mask.sum() < 3:
        return np.nan

    diff = a_sup[mask] - b_aln[mask]
    return float(np.sqrt((diff * diff).sum() / mask.sum()))


# -----------------------------
# Caching
# -----------------------------
@dataclass
class CachedPDB:
    seq: str
    ca: np.ndarray
    contact: np.ndarray
    pdb_path: str


def cache_key_for_path(p: str) -> str:
    st = os.stat(p)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.strip("/"))
    return f"{safe}__{st.st_size}__{int(st.st_mtime)}"


def load_or_build_cache(pdb_path: str, cache_dir: str, contact_thresh: float) -> CachedPDB:
    """
    Strict load with allow_pickle=False.
    Legacy caches storing seq as object dtype are deterministically rebuilt.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key_for_path(pdb_path)
    cache_fn = os.path.join(cache_dir, key + ".npz")

    def _build_and_write() -> CachedPDB:
        seq, ca = parse_pdb_seq_and_ca(pdb_path)
        contact = contact_graph_from_ca(ca, contact_thresh)
        np.savez_compressed(
            cache_fn,
            seq=np.array(seq, dtype="U"),  # unicode scalar, not object
            ca=ca.astype(np.float32, copy=False),
            contact=contact.astype(np.uint8, copy=False),
        )
        return CachedPDB(seq=seq, ca=ca, contact=contact, pdb_path=pdb_path)

    if os.path.exists(cache_fn):
        try:
            z = np.load(cache_fn, allow_pickle=False)
            seq_arr = z["seq"]
            if isinstance(seq_arr, np.ndarray) and seq_arr.shape == ():
                seq = str(seq_arr.item())
            else:
                seq = str(seq_arr.tolist())
            ca = z["ca"]
            contact = z["contact"]
            return CachedPDB(seq=seq, ca=ca, contact=contact, pdb_path=pdb_path)
        except ValueError as e:
            if "Object arrays cannot be loaded" in str(e):
                os.remove(cache_fn)
                return _build_and_write()
            raise

    return _build_and_write()


# -----------------------------
# Variant path helpers
# -----------------------------
def find_variant_dir(features_dir: str, var_name: str) -> str:
    return os.path.join(features_dir, var_name)


def wt_pdb_path(features_dir: str, var_name: str) -> str:
    p = os.path.join(find_variant_dir(features_dir, var_name), "wt_data", DEFAULT_WT_PDB)
    if not os.path.exists(p):
        raise FileNotFoundError(f"WT PDB missing for {var_name}: {p}")
    return p


def esmfold_mut_paths(features_dir: str, var_name: str) -> Dict[str, str]:
    mut_dir = os.path.join(find_variant_dir(features_dir, var_name), "mut_data")
    out: Dict[str, str] = {}
    p_single = os.path.join(mut_dir, DEFAULT_MUT_E2E_ESMF)
    if os.path.exists(p_single):
        out["esm_single"] = p_single
    for k in range(5):
        p = os.path.join(mut_dir, f"mut_esmf_rep{k}.pdb")
        if os.path.exists(p):
            out[f"esm_rep{k}"] = p
    if len(out) == 0:
        raise FileNotFoundError(f"No ESMFold mutant PDBs found for {var_name} in {mut_dir}")
    return out


def rosetta_mut_path(features_dir: str, var_name: str) -> Optional[str]:
    p = os.path.join(find_variant_dir(features_dir, var_name), "mut_data", DEFAULT_MUT_ROSETTA)
    return p if os.path.exists(p) else None


def rosetta_rep_paths(features_dir: str, var_name: str, geo_pilot_map: Dict[str, str], rosetta_work_root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if var_name not in geo_pilot_map:
        return out
    pilot_name = geo_pilot_map[var_name]
    parts = pilot_name.split("_")
    if len(parts) < 2:
        return out
    protein, chain = parts[0], parts[1]
    base_dir = os.path.join(rosetta_work_root, f"{pilot_name}__rosetta")
    if not os.path.isdir(base_dir):
        return out
    for i in range(1, 6):
        p = os.path.join(base_dir, f"{protein}_{chain}_{i:04d}.pdb")
        if os.path.exists(p):
            out[f"ros_rep{i}"] = p
    return out


# -----------------------------
# Core alignment + metrics
# -----------------------------
def align_and_superpose_by_tmalign(
    a: CachedPDB,
    b: CachedPDB,
    tmalign_bin: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[int], str, str]:
    pairs, aln1, aln2 = run_tmalign_pairs(tmalign_bin, a.pdb_path, b.pdb_path)

    pairs = [(i, j) for (i, j) in pairs if i < a.ca.shape[0] and j < b.ca.shape[0]]
    if len(pairs) < 5:
        raise ValueError(f"Too few aligned residue pairs after bounds filtering: {len(pairs)}")

    a_idx = [i for i, _ in pairs]
    b_idx = [j for _, j in pairs]

    ca_a = a.ca[a_idx]
    ca_b = b.ca[b_idx]

    # superpose A onto B
    R, t = kabsch_rt(ca_a, ca_b)
    ca_a_sup = apply_rt(ca_a, R, t)

    C_a = a.contact[np.ix_(a_idx, a_idx)]
    C_b = b.contact[np.ix_(b_idx, b_idx)]

    return ca_a_sup, ca_b, C_a, C_b, a_idx, b_idx, aln1, aln2


def compare_structures_global_tmalign(
    wt: CachedPDB,
    mut: CachedPDB,
    mut_pos0: Optional[int],
    local_win: int,
    sphere_radii: List[float],
    tmalign_bin: str,
) -> Dict[str, float]:
    wt_sup, mut_aln, C_wt, C_mut, wt_idx, mut_idx, aln1, aln2 = align_and_superpose_by_tmalign(
        wt, mut, tmalign_bin
    )

    n_gained, n_lost = contact_delta_counts(C_wt, C_mut)
    coverage = len(wt_idx) / float(min(wt.ca.shape[0], mut.ca.shape[0]))

    out: Dict[str, float] = {
        "n_aligned": int(len(wt_idx)),
        "coverage": float(coverage),
        "gaps_wt": float(aln1.count("-")),
        "gaps_mut": float(aln2.count("-")),
        "ca_rmsd": ca_rmsd(wt_sup, mut_aln),
        "ca_pair_mean_abs_diff": ca_pair_mean_abs_diff(wt_sup, mut_aln),
        "n_contact_gained": int(n_gained),
        "n_contact_lost": int(n_lost),
        "ca_rmsd_local": np.nan,
        "ca_rmsd_distal": np.nan,
    }

    def _rad_key(r: float) -> str:
        r_int = int(round(r))
        if abs(r - r_int) < 1e-6:
            return f"ca_rmsd_sphere_{r_int}A"
        return f"ca_rmsd_sphere_{str(r).replace('.','p')}A"

    for r in sphere_radii:
        out[_rad_key(r)] = np.nan

    # Multi-radius sphere RMSD about mutation site (WT indexing)
    if mut_pos0 is not None:
        for r in sphere_radii:
            out[_rad_key(r)] = sphere_rmsd_about_index(
                a_sup=wt_sup,
                b_aln=mut_aln,
                a_idx=wt_idx,
                center_pos0=mut_pos0,
                radius_A=float(r),
            )

    # Sequence-window RMSDs (kept)
    if mut_pos0 is not None:
        k = None
        for aligned_pos, wt_i in enumerate(wt_idx):
            if wt_i == mut_pos0:
                k = aligned_pos
                break
        if k is not None:
            idxs = np.arange(len(wt_idx))
            local_mask = np.abs(idxs - k) <= local_win
            distal_mask = ~local_mask
            if local_mask.sum() >= 3:
                out["ca_rmsd_local"] = ca_rmsd(wt_sup[local_mask], mut_aln[local_mask])
            if distal_mask.sum() >= 3:
                out["ca_rmsd_distal"] = ca_rmsd(wt_sup[distal_mask], mut_aln[distal_mask])

    return out


def compare_mutants_global_tmalign(
    a: CachedPDB,
    b: CachedPDB,
    tmalign_bin: str,
    mut_pos0: Optional[int] = None,
    sphere_radii: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Pairwise mutant-vs-mutant comparison using the SAME alignment + Kabsch method.
    UPDATED: optionally compute ca_rmsd_sphere_{R}A using mut_pos0 in A-indexing.
    """
    a_sup, b_aln, _, _, a_idx, b_idx, aln1, aln2 = align_and_superpose_by_tmalign(a, b, tmalign_bin)

    out: Dict[str, float] = {
        "n_aligned": int(len(a_idx)),
        "coverage": float(len(a_idx) / float(min(a.ca.shape[0], b.ca.shape[0]))),
        "gaps_a": float(aln1.count("-")),
        "gaps_b": float(aln2.count("-")),
        "ca_rmsd": ca_rmsd(a_sup, b_aln),
    }

    if sphere_radii is None:
        return out

    def _rad_key(r: float) -> str:
        r_int = int(round(r))
        if abs(r - r_int) < 1e-6:
            return f"ca_rmsd_sphere_{r_int}A"
        return f"ca_rmsd_sphere_{str(r).replace('.','p')}A"

    for r in sphere_radii:
        out[_rad_key(r)] = np.nan

    if mut_pos0 is not None:
        for r in sphere_radii:
            out[_rad_key(r)] = sphere_rmsd_about_index(
                a_sup=a_sup,
                b_aln=b_aln,
                a_idx=a_idx,          # A indexing for center lookup
                center_pos0=mut_pos0,
                radius_A=float(r),
            )

    return out


# -----------------------------
# Model results loading
# -----------------------------
def load_model_results(meltingtemp_results_fns_dct: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for model_name, files in meltingtemp_results_fns_dct.items():
        if isinstance(files, str):
            files = [files]
        for fn in files:
            if not os.path.exists(fn):
                print("Warning: missing:", fn)
                continue
            df = pd.read_csv(fn)
            if "name" not in df.columns:
                raise ValueError(f"{fn} missing 'name' column")

            if "model_score" in df.columns:
                pred = df["model_score"].astype(float)
            elif "pred" in df.columns:
                pred = df["pred"].astype(float)
            else:
                cand = [c for c in df.columns if c not in ("name", "true", "true_label")]
                if len(cand) == 0:
                    raise ValueError(f"{fn} could not infer prediction column")
                pred = df[cand[0]].astype(float)

            if "true_label" in df.columns:
                true = df["true_label"].astype(float)
            elif "true" in df.columns:
                true = df["true"].astype(float)
            else:
                raise ValueError(f"{fn} missing true labels ('true' or 'true_label')")

            for i in range(len(df)):
                rows.append({
                    "model": model_name,
                    "run_file": fn,
                    "name": str(df.loc[i, "name"]),
                    "pred": float(pred.iloc[i]),
                    "true": float(true.iloc[i]),
                })

    out = pd.DataFrame(rows)
    out["abs_err"] = (out["pred"] - out["true"]).abs()
    return out


def summarize_model_runs(df_model: pd.DataFrame) -> pd.DataFrame:
    g = df_model.groupby(["model", "name"], as_index=False)
    out = g.agg(
        true=("true", "first"),
        pred_mean=("pred", "mean"),
        pred_std=("pred", "std"),
        abs_err_mean=("abs_err", "mean"),
        abs_err_std=("abs_err", "std"),
        n_runs=("run_file", "nunique"),
    )
    return out


# -----------------------------
# Plotting helpers
# -----------------------------
def save_scatter(out_png: str, x, y, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(x, y, alpha=0.65, s=18)
    ok = np.isfinite(x) & np.isfinite(y)
    rho = np.nan
    if ok.sum() >= 10:
        rho, _ = spearmanr(x[ok], y[ok])
    plt.title(f"{title}\nSpearman ρ={rho:.3f} (n={ok.sum()})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_hist(out_png: str, vals, title: str, xlabel: str):
    plt.figure(figsize=(6.5, 4.8))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=50, edgecolor="black", linewidth=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_sphere_radii(arg: Optional[str]) -> List[float]:
    if arg is None:
        return []
    s = str(arg).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    radii = [float(p) for p in parts]
    radii = sorted(set(radii))
    return radii


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="/projects/ashehu/amoldwin/GeoStab/data/dTm/S4346/")
    ap.add_argument("--out_dir", default="analysis_outputs_graph_tmalign_global_strict_multiSphere")
    ap.add_argument("--contact_thresh", type=float, default=8.0)
    ap.add_argument("--cache_dir", default="/scratch/amoldwin/cache/")
    ap.add_argument("--rosetta_work_root", default="/scratch/amoldwin/datasets/PILOT_dTm_esmfold/cleaned_pdb/rosetta_work")
    ap.add_argument("--tmalign_bin", default=os.environ.get("TMALIGN_BIN", "TMalign"))
    ap.add_argument("--local_win", type=int, default=8)

    ap.add_argument("--sphere_radius", type=float, default=15.0, help="(legacy) single radius in Å if --sphere_radii not provided")
    ap.add_argument("--sphere_radii", type=str, default="8,10,12,15", help="comma-separated radii in Å, e.g. 8,10,12,15")

    args = ap.parse_args()

    if not (os.path.isfile(args.tmalign_bin) or shutil.which(args.tmalign_bin)):
        raise RuntimeError(
            f"TM-align not found: '{args.tmalign_bin}'. "
            f"Provide --tmalign_bin /path/to/TMalign or set TMALIGN_BIN."
        )

    sphere_radii = parse_sphere_radii(args.sphere_radii)
    if len(sphere_radii) == 0:
        sphere_radii = [float(args.sphere_radius)]

    out_dir = args.out_dir
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ---- model predictions mapping
    ablation_job_ids = ['5095391', '5095466', '5095467', '5095470', '5095471', '5095472', '5095473', '5095478', '5095479', '5095465']
    all_inputs_ids = ['5095603', '5095466', '5095467', '5095470', '5095471', '5095472', '5095473', '5095478', '5095479', '5095391']
    meltingtemp_results_fns_dct = {
        "MMGAT* (all inputs)": [
            "./geodtm_models_ablation/" + x + f"_geodtm_test_predictions_all_inputs_{i}.csv" for i, x in enumerate(all_inputs_ids)
        ],
        "MMGAT* (-atom mask)": [
            "./geodtm_models_ablation/" + x + f"_geodtm_test_predictions_no_atommask_{i}.csv" for i, x in enumerate(ablation_job_ids)
        ],
        "MMGAT* (-pair geometry)": [
            "./geodtm_models_ablation/" + x + f"_geodtm_test_predictions_no_pair_{i}.csv" for i, x in enumerate(ablation_job_ids)
        ],
        "MMGAT* (-ESM2 embeddings)": [
            "./geodtm_models_ablation/" + x + f"_geodtm_test_predictions_no_demb_{i}.csv" for i, x in enumerate(ablation_job_ids)
        ],
        "MMGAT* (-AA descriptors)": [
            "./geodtm_models_ablation/" + x + f"_geodtm_test_predictions_no_fixed_{i}.csv" for i, x in enumerate(ablation_job_ids)
        ],
        "MMGAT* (ESM2 only)": [
            f"./geodtm_models_ablation/5099249_geodtm_test_predictions_no_fixed_no_pair_no_atommask_no_pH_no_plddt_{i}.csv"
            for i in range(10)
        ],
        "MMGAT* (WT-duplicated structure)": [
            f"./geodtm_models_ablation/5142092_geodtm_test_predictions_dupWTstruct_{i}.csv"
            for i in range(10)
        ],
        "MMGAT* Rosetta Mut Struct": [
            f"./geodtm_models_ablation/5336582_geodtm_test_predictions_rosetta_{i}.csv"
            for i in range(10)
        ],
        "MMGAT* Rosetta (-ESM2)": [
            f"./geodtm_models_ablation/5336583_geodtm_test_predictions_no_demb_rosetta_{i}.csv"
            for i in range(10)
        ],
        "Baseline: ESM2 Masked Marginal": "./zero_shot_esm2_masked_S571.csv",
    }

    df_model_raw = load_model_results(meltingtemp_results_fns_dct)
    df_model_raw.to_csv(os.path.join(out_dir, "model_results_raw.csv"), index=False)

    df_model = summarize_model_runs(df_model_raw)
    df_model["abs_dTm"] = df_model["true"].abs()
    df_model.to_csv(os.path.join(out_dir, "model_results_agg.csv"), index=False)

    variants = sorted(df_model["name"].unique().tolist())

    # ---- geo_pilot_map
    train_map = pd.read_csv('../PILOT/dataset/dTm_train_mutpos_name_mapping.csv')
    test_map = pd.read_csv('../PILOT/dataset/dTm_test_mutpos_name_mapping.csv')

    pilot_names = list(train_map.apply(lambda row: f"{row['PDB']}_{row['chain']}_{row['wt_res']}{row['mut_pos']}{row['mut_res']}", axis=1))
    pilot_names += list(test_map.apply(lambda row: f"{row['PDB']}_{row['chain']}_{row['wt_res']}{row['mut_pos']}{row['mut_res']}", axis=1))

    geo_names = list(train_map['name']) + list(test_map['name'])
    geo_pilot_map = dict(zip(geo_names, pilot_names))

    # helper for consistent key naming
    def sphere_key(r: float) -> str:
        r_int = int(round(r))
        if abs(r - r_int) < 1e-6:
            return f"ca_rmsd_sphere_{r_int}A"
        return f"ca_rmsd_sphere_{str(r).replace('.','p')}A"

    sphere_cols = [sphere_key(r) for r in sphere_radii]

    # ---- compute structural metrics (STRICT)
    per_rep_rows = []
    agg_rows = []
    var_rows = []

    for v in tqdm(variants, desc="Variants"):
        vdir = find_variant_dir(args.features_dir, v)
        if not os.path.isdir(vdir):
            raise FileNotFoundError(f"Variant directory missing: {vdir}")
        if v not in geo_pilot_map:
            raise KeyError(f"Variant missing from geo_pilot_map: {v}")

        mutpos = int(geo_pilot_map[v].split('_')[2][1:-1])  # e.g. A123B -> 123
        mutpos0 = mutpos - 1

        wt_p = wt_pdb_path(args.features_dir, v)
        wt_cached = load_or_build_cache(wt_p, args.cache_dir, args.contact_thresh)

        esm_paths = esmfold_mut_paths(args.features_dir, v)
        ros_single = rosetta_mut_path(args.features_dir, v)
        ros_reps = rosetta_rep_paths(args.features_dir, v, geo_pilot_map=geo_pilot_map, rosetta_work_root=args.rosetta_work_root)

        # WT vs ESMFold (single + reps)
        for tag, p in esm_paths.items():
            m = load_or_build_cache(p, args.cache_dir, args.contact_thresh)
            met = compare_structures_global_tmalign(
                wt_cached, m,
                mut_pos0=mutpos0,
                local_win=args.local_win,
                sphere_radii=sphere_radii,
                tmalign_bin=args.tmalign_bin,
            )
            per_rep_rows.append({"name": v, "mut_source": "ESMFold", "rep": tag, "pdb": p, **met})

        # WT vs Rosetta (single)
        if ros_single:
            m = load_or_build_cache(ros_single, args.cache_dir, args.contact_thresh)
            met = compare_structures_global_tmalign(
                wt_cached, m,
                mut_pos0=mutpos0,
                local_win=args.local_win,
                sphere_radii=sphere_radii,
                tmalign_bin=args.tmalign_bin,
            )
            per_rep_rows.append({"name": v, "mut_source": "Rosetta", "rep": "ros_single", "pdb": ros_single, **met})

        # WT vs Rosetta (reps)
        for tag, p in ros_reps.items():
            m = load_or_build_cache(p, args.cache_dir, args.contact_thresh)
            met = compare_structures_global_tmalign(
                wt_cached, m,
                mut_pos0=mutpos0,
                local_win=args.local_win,
                sphere_radii=sphere_radii,
                tmalign_bin=args.tmalign_bin,
            )
            per_rep_rows.append({"name": v, "mut_source": "Rosetta", "rep": tag, "pdb": p, **met})

        # Aggregate per variant/source
        def agg_source(source: str):
            rows = [r for r in per_rep_rows if r["name"] == v and r["mut_source"] == source]
            if len(rows) == 0:
                return None
            df = pd.DataFrame(rows)
            out = {"name": v, "mut_source": source, "n_reps": int(len(df))}

            base_cols = [
                "n_aligned", "coverage", "gaps_wt", "gaps_mut",
                "ca_rmsd", "ca_rmsd_local", "ca_rmsd_distal",
                "ca_pair_mean_abs_diff",
                "n_contact_gained", "n_contact_lost",
            ]
            cols = base_cols + sphere_cols

            for col in cols:
                if col not in df.columns:
                    out[col + "_mean"] = np.nan
                    out[col + "_std"] = np.nan
                    continue
                out[col + "_mean"] = float(df[col].mean())
                out[col + "_std"] = float(df[col].std())
            return out

        a_esm = agg_source("ESMFold")
        a_ros = agg_source("Rosetta")
        if a_esm is not None:
            agg_rows.append(a_esm)
        if a_ros is not None:
            agg_rows.append(a_ros)

        # Intra-run variability: pairwise RMSD among replicate structures (UPDATED: includes sphere windows)
        def intra_variability(source: str):
            reps = [r for r in per_rep_rows if r["name"] == v and r["mut_source"] == source]
            rep_paths = []
            for r in reps:
                if source == "ESMFold" and str(r["rep"]).startswith("esm_rep"):
                    rep_paths.append(r["pdb"])
                if source == "Rosetta" and str(r["rep"]).startswith("ros_rep"):
                    rep_paths.append(r["pdb"])
            rep_paths = sorted(set(rep_paths))
            if len(rep_paths) < 2:
                return None

            cached = [load_or_build_cache(p, args.cache_dir, args.contact_thresh) for p in rep_paths]

            rmsds = []
            sphere_vals = {c: [] for c in sphere_cols}

            for i in range(len(cached)):
                for j in range(i + 1, len(cached)):
                    met = compare_mutants_global_tmalign(
                        cached[i], cached[j],
                        tmalign_bin=args.tmalign_bin,
                        mut_pos0=mutpos0,
                        sphere_radii=sphere_radii,
                    )
                    if np.isfinite(met["ca_rmsd"]):
                        rmsds.append(met["ca_rmsd"])
                    for c in sphere_cols:
                        if c in met and np.isfinite(met[c]):
                            sphere_vals[c].append(met[c])

            if len(rmsds) == 0:
                return None

            out = {
                "name": v,
                "mut_source": source,
                "rep_pair_rmsd_mean": float(np.mean(rmsds)),
                "rep_pair_rmsd_std": float(np.std(rmsds)),
                "rep_pair_rmsd_max": float(np.max(rmsds)),
                "n_rep_pairs": int(len(rmsds)),
            }

            # add windowed/sphere pairwise summaries
            for c in sphere_cols:
                vals = np.asarray(sphere_vals[c], dtype=float)
                if vals.size == 0:
                    out[f"rep_pair_rmsd_{c}_mean"] = np.nan
                    out[f"rep_pair_rmsd_{c}_std"] = np.nan
                    out[f"rep_pair_rmsd_{c}_max"] = np.nan
                else:
                    out[f"rep_pair_rmsd_{c}_mean"] = float(np.mean(vals))
                    out[f"rep_pair_rmsd_{c}_std"] = float(np.std(vals))
                    out[f"rep_pair_rmsd_{c}_max"] = float(np.max(vals))

            return out

        v_esm = intra_variability("ESMFold")
        v_ros = intra_variability("Rosetta")
        if v_esm:
            var_rows.append(v_esm)
        if v_ros:
            var_rows.append(v_ros)

    # Save CSVs
    df_per_rep = pd.DataFrame(per_rep_rows)
    df_agg = pd.DataFrame(agg_rows)
    df_var = pd.DataFrame(var_rows)

    df_per_rep.to_csv(os.path.join(out_dir, "struct_metrics_per_rep.csv"), index=False)
    df_agg.to_csv(os.path.join(out_dir, "struct_metrics_agg.csv"), index=False)
    df_var.to_csv(os.path.join(out_dir, "struct_metrics_variability.csv"), index=False)

    # Merge model + struct
    if len(df_agg) == 0:
        raise RuntimeError("No aggregated structural metrics computed (unexpected if per-rep succeeded).")

    df_merged = df_model.merge(df_agg, on="name", how="left")
    df_merged = df_merged.merge(
        df_var.rename(columns={"mut_source": "mut_source_var"}),
        left_on=["name", "mut_source"],
        right_on=["name", "mut_source_var"],
        how="left",
    )
    df_merged.to_csv(os.path.join(out_dir, "merged_model_struct.csv"), index=False)

    # Correlations + plots
    metrics = [
        "ca_rmsd_mean",
        *[f"{c}_mean" for c in sphere_cols],
        "ca_pair_mean_abs_diff_mean",
        "n_contact_gained_mean",
        "n_contact_lost_mean",
        "rep_pair_rmsd_mean",
        "rep_pair_rmsd_max",
        # NEW: replicate-pair sphere RMSDs (these live in df_var; after merge they may appear with NaNs if missing)
        *[f"rep_pair_rmsd_{c}_mean" for c in sphere_cols],
        *[f"rep_pair_rmsd_{c}_max" for c in sphere_cols],
    ]

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    summary_rows = []
    for (model, mut_source), g in df_merged.groupby(["model", "mut_source"]):
        for m in metrics:
            if m not in g.columns:
                continue
            x = g[m].to_numpy(dtype=float)
            y = g["abs_err_mean"].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(x[ok], y[ok])
            summary_rows.append({
                "model": model,
                "mut_source": mut_source,
                "metric": m,
                "n": int(ok.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            })

            if ok.sum() >= 10:
                out_png = os.path.join(
                    plots_dir,
                    f"scatter__{re.sub('[^A-Za-z0-9]+','_',model)}__{mut_source}__{m}.png"
                )
                save_scatter(
                    out_png,
                    x[ok],
                    y[ok],
                    title=f"{model} | {mut_source} | abs_err_mean vs {m}",
                    xlabel=m,
                    ylabel="abs_err_mean (°C)",
                )

    df_corr = pd.DataFrame(summary_rows).sort_values(["metric", "spearman_rho"])
    df_corr.to_csv(os.path.join(out_dir, "corr_summary.csv"), index=False)

    # Histograms (global + multi-sphere)
    if "ca_rmsd_mean" in df_merged.columns:
        save_hist(
            os.path.join(plots_dir, "hist__ca_rmsd_mean.png"),
            df_merged["ca_rmsd_mean"].to_numpy(dtype=float),
            "Distribution of WT–MUT CA RMSD (mean across reps)",
            "CA RMSD (Å)",
        )

    for c in sphere_cols:
        col = f"{c}_mean"
        if col in df_merged.columns:
            save_hist(
                os.path.join(plots_dir, f"hist__{col}.png"),
                df_merged[col].to_numpy(dtype=float),
                f"Distribution of WT–MUT {c.replace('ca_rmsd_sphere_','').replace('A','Å')}-sphere CA RMSD (mean across reps)",
                f"{c.replace('ca_rmsd_sphere_','').replace('A','Å')}-sphere RMSD (Å)",
            )

    print("Done.")
    print("Sphere radii (Å):", sphere_radii)
    print("Wrote:")
    print("  - struct_metrics_per_rep.csv")
    print("  - struct_metrics_agg.csv")
    print("  - struct_metrics_variability.csv")
    print("  - model_results_raw.csv / model_results_agg.csv")
    print("  - merged_model_struct.csv")
    print("  - corr_summary.csv")
    print("  - plots/*.png")


if __name__ == "__main__":
    main()
