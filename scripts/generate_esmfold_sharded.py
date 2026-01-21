import os
import time
import argparse
import pickle
import hashlib
import shutil
from pathlib import Path

import torch
import esm
from Bio import SeqIO
from typing import Optional, List, Tuple, Dict


# -------------------------
# Utilities
# -------------------------

def sanitize_sequence(seq: str) -> str:
    """
    Substitute non-standard amino acids:
    - 'U' (selenocysteine) -> 'C'
    - 'O' (pyrrolysine) -> 'K'
    """
    return seq.replace("U", "C").replace("O", "K")


def format_hms(seconds: float) -> str:
    """Convert seconds to H:MM:SS or M:SS."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:d}:{s:02d}"


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_if_missing(src: Path, dst: Path, overwrite: bool = False) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return True


def stable_key_hash(s: str) -> int:
    """
    Stable hash for sharding that won't change across processes.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def shard_by_key(items: List, array_ID: Optional[int], array_N: Optional[int], key_fn) -> List:
    """
    Assign items to shards by hashing a stable per-item key.
    This avoids "orphan jobs" when different shards see different filtered lists.
    """
    if array_ID is None or array_N is None:
        return items
    out = []
    for x in items:
        k = key_fn(x)
        if (stable_key_hash(k) % array_N) == array_ID:
            out.append(x)
    return out


def shard_by_index(items: List, array_ID: Optional[int], array_N: Optional[int]) -> List:
    """
    Kept for WT hashes (WT hash list is stable), but MUT uses shard_by_key.
    """
    if array_ID is None or array_N is None:
        return items
    return [x for i, x in enumerate(items) if i % array_N == array_ID]


def create_batched_job_dataset(jobs, max_tokens_per_batch: int):
    """
    jobs: list of (header, seq, pdb_path, pkl_path)
    """
    batch_headers = []
    batch_seqs = []
    batch_pdbs = []
    batch_pkls = []
    num_tokens = 0

    for header, seq, pdb_path, pkl_path in jobs:
        seq = sanitize_sequence(seq)
        seq_len = len(seq)

        if num_tokens > 0 and (num_tokens + seq_len > max_tokens_per_batch):
            yield batch_headers, batch_seqs, batch_pdbs, batch_pkls
            batch_headers, batch_seqs = [], []
            batch_pdbs, batch_pkls = [], []
            num_tokens = 0

        batch_headers.append(header)
        batch_seqs.append(seq)
        batch_pdbs.append(pdb_path)
        batch_pkls.append(pkl_path)
        num_tokens += seq_len

    if batch_headers:
        yield batch_headers, batch_seqs, batch_pdbs, batch_pkls


# -------------------------
# Job discovery
# -------------------------

def list_sample_dirs(data_parent_path: Path) -> List[Path]:
    # Iterate subfolders (no need to require mut_info.csv)
    sample_dirs = []
    for name in os.listdir(data_parent_path):
        p = data_parent_path / name
        if p.is_dir():
            sample_dirs.append(p)
    sample_dirs.sort(key=lambda x: str(x).lower())
    return sample_dirs


def build_mut_jobs(sample_dirs: List[Path]) -> List[Tuple[str, str, str, str]]:
    """
    FIXED: Build MUT jobs for every sample that has mut.fasta.
    DO NOT filter out jobs based on existing outputs here.
    Skipping happens later inside run_jobs_with_model(), which is safe under restarts.
    """
    jobs = []
    for sample_dir in sample_dirs:
        mut_dir = sample_dir / "mut_data"
        mut_fasta = mut_dir / "mut.fasta"
        mut_pdb = mut_dir / "mut_esmf.pdb"
        mut_pkl = mut_dir / "mut_esmf.pkl"

        if mut_fasta.exists():
            record = SeqIO.read(str(mut_fasta), "fasta")
            header = record.id
            seq = sanitize_sequence(str(record.seq))
            jobs.append((header, seq, str(mut_pdb), str(mut_pkl)))
    return jobs


def group_wt_dirs_by_fasta_hash(sample_dirs: List[Path]) -> Dict[str, List[Path]]:
    """
    WT de-dup groups by hashing wt_data/wt.fasta.
    Returns: {hash: [wt_data_dir1, wt_data_dir2, ...]}
    """
    groups: Dict[str, List[Path]] = {}
    for sample_dir in sample_dirs:
        wt_dir = sample_dir / "wt_data"
        wt_fasta = wt_dir / "wt.fasta"
        if not wt_fasta.exists():
            continue
        h = sha1_file(wt_fasta)
        groups.setdefault(h, []).append(wt_dir)

    # Stable ordering within groups
    for h in list(groups.keys()):
        groups[h].sort(key=lambda x: str(x).lower())

    return groups


def build_wt_representative_jobs(wt_groups: Dict[str, List[Path]]) -> List[Tuple[str, str, str, str, str]]:
    """
    Returns list of representative WT jobs:
      (hash, header, seq, rep_wt_pdb_path, rep_wt_pkl_path)

    We compute in the representative wt_dir = groups[h][0].
    """
    reps = []
    for h, wt_dirs in wt_groups.items():
        rep_wt_dir = wt_dirs[0]
        wt_fasta = rep_wt_dir / "wt.fasta"
        wt_pdb = rep_wt_dir / "wt_esmf.pdb"
        wt_pkl = rep_wt_dir / "wt_esmf.pkl"

        record = SeqIO.read(str(wt_fasta), "fasta")
        header = record.id
        seq = sanitize_sequence(str(record.seq))
        reps.append((h, header, seq, str(wt_pdb), str(wt_pkl)))
    return reps


# -------------------------
# Inference + writing
# -------------------------

def run_jobs_with_model(
    model,
    jobs: List[Tuple[str, str, str, str]],
    max_tokens_per_batch: int,
    num_recycles: Optional[int],
    start_time: float,
    label: str,
):
    """
    jobs: list of (header, seq, pdb_path, pkl_path)
    Writes pdb if missing; writes pkl if missing.
    Skips jobs that already have pkl at runtime (restart-safe).
    """
    total_jobs = len(jobs)
    if total_jobs == 0:
        print(f"No {label} jobs to run; skipping.", flush=True)
        return

    completed = 0

    for headers, sequences, pdb_paths, pkl_paths in create_batched_job_dataset(
        jobs, max_tokens_per_batch=max_tokens_per_batch
    ):
        batch_size = len(sequences)
        if batch_size == 0:
            continue

        # Filter already-complete pkls (safe under restarts)
        filtered = []
        for h, s, pdb, pkl in zip(headers, sequences, pdb_paths, pkl_paths):
            if os.path.exists(pkl):
                completed += 1
                continue
            filtered.append((h, s, pdb, pkl))
        if not filtered:
            continue

        headers, sequences, pdb_paths, pkl_paths = zip(*filtered)
        headers, sequences = list(headers), list(sequences)
        pdb_paths, pkl_paths = list(pdb_paths), list(pkl_paths)
        batch_size = len(sequences)

        try:
            output = model.infer(sequences, num_recycles=num_recycles)
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("CUDA out of memory"):
                print(
                    f"[WARN] CUDA OOM on {label} batch of size {batch_size} "
                    f"(max_tokens_per_batch={max_tokens_per_batch}). "
                    f"Consider lowering --max_tokens_per_batch.",
                    flush=True,
                )
                continue
            else:
                print(f"[ERROR] Unexpected RuntimeError on {label} batch {headers}: {e}", flush=True)
                continue

        output = {key: value.cpu() for key, value in output.items()}
        pdb_strings = model.output_to_pdb(output)
        plddts = output.get("plddt", None)
        mean_plddt = output.get("mean_plddt", None)
        ptm = output.get("ptm", None)

        for idx, (header, seq, pdb_str, pdb_path, pkl_path) in enumerate(
            zip(headers, sequences, pdb_strings, pdb_paths, pkl_paths)
        ):
            if not os.path.exists(pdb_path):
                with open(pdb_path, "w") as f:
                    f.write(pdb_str)
                print(f"Wrote {pdb_path}", flush=True)

            if plddts is not None and not os.path.exists(pkl_path):
                plddt_vec = plddts[idx].numpy()
                with open(pkl_path, "wb") as f:
                    pickle.dump({"plddt": plddt_vec}, f)
                print(f"Wrote {pkl_path}", flush=True)

            completed += 1
            elapsed = time.time() - start_time
            avg_per_job = elapsed / max(completed, 1)
            remaining = total_jobs - completed
            eta = remaining * avg_per_job
            pct = 100.0 * completed / total_jobs if total_jobs > 0 else 100.0
            mean_pl = mean_plddt[idx].item() if mean_plddt is not None else float("nan")
            ptm_val = ptm[idx].item() if ptm is not None else float("nan")

            print(
                f"[{label}] [{completed}/{total_jobs}] ({pct:5.1f}%) "
                f"| Elapsed: {format_hms(elapsed)} "
                f"| ETA: {format_hms(eta)} "
                f"| len={len(seq)} "
                f"| mean pLDDT={mean_pl:0.1f} pTM={ptm_val:0.3f} "
                f"| {header}",
                flush=True,
            )


# -------------------------
# WT propagation
# -------------------------

def propagate_wt_outputs(
    wt_groups: Dict[str, List[Path]],
    overwrite: bool = False,
):
    """
    Copy representative WT outputs into all wt_dirs in group.
    We propagate:
      - wt_esmf.pdb
      - wt_esmf.pkl
    """
    for _, wt_dirs in wt_groups.items():
        rep = wt_dirs[0]
        rep_pdb = rep / "wt_esmf.pdb"
        rep_pkl = rep / "wt_esmf.pkl"

        for dst in wt_dirs[1:]:
            dst_pdb = dst / "wt_esmf.pdb"
            dst_pkl = dst / "wt_esmf.pkl"

            if copy_if_missing(rep_pdb, dst_pdb, overwrite=overwrite):
                print(f"[WT] Propagated PDB {rep} -> {dst}", flush=True)
            if copy_if_missing(rep_pkl, dst_pkl, overwrite=overwrite):
                print(f"[WT] Propagated PKL {rep} -> {dst}", flush=True)


# -------------------------
# Main
# -------------------------

def main(
    data_parent: str,
    num_samples: Optional[int] = None,
    max_tokens_per_batch: int = 1024,
    num_recycles: Optional[int] = None,
    chunk_size: Optional[int] = None,
    cpu_only: bool = False,
    start_from_longest: bool = False,
    array_ID: Optional[int] = None,
    array_N: Optional[int] = None,
    overwrite_wt_propagation: bool = False,
):
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)

    if cpu_only or not torch.cuda.is_available():
        device = "cpu"
        model.cpu()
    else:
        device = "cuda"
        model.cuda()

    print(f"Loaded ESMFold model on {device}", flush=True)

    data_parent_path = Path(data_parent)
    sample_dirs = list_sample_dirs(data_parent_path)
    print(f"Found {len(sample_dirs)} sample dirs under {data_parent_path}", flush=True)

    # -------------------------
    # WT (dedup groups, shard by unique WT hashes)
    # WT hash list is stable, so index-based sharding is fine here.
    # -------------------------
    wt_groups = group_wt_dirs_by_fasta_hash(sample_dirs)
    wt_hashes_all = sorted(list(wt_groups.keys()))
    wt_hashes = shard_by_index(wt_hashes_all, array_ID, array_N)

    # Representatives for THIS SHARD only
    wt_rep_jobs = []
    for h in wt_hashes:
        rep_job = build_wt_representative_jobs({h: wt_groups[h]})[0]
        _, header, seq, pdb_path, pkl_path = rep_job
        if os.path.exists(pkl_path):
            continue
        wt_rep_jobs.append((header, seq, pdb_path, pkl_path))

    if start_from_longest:
        wt_rep_jobs.sort(key=lambda x: len(x[1]), reverse=True)
    else:
        wt_rep_jobs.sort(key=lambda x: len(x[1]))

    print(
        f"WT unique groups total={len(wt_groups)} | "
        f"this shard WT groups={len(wt_hashes)} | "
        f"this shard WT reps needing pkl={len(wt_rep_jobs)}",
        flush=True,
    )

    # -------------------------
    # MUT (FIXED: stable per-job sharding)
    # -------------------------
    mut_jobs_all = build_mut_jobs(sample_dirs)

    if num_samples is not None:
        mut_jobs_all = mut_jobs_all[:num_samples]

    if start_from_longest:
        mut_jobs_all.sort(key=lambda x: len(x[1]), reverse=True)
    else:
        mut_jobs_all.sort(key=lambda x: len(x[1]))

    total_mut_all = len(mut_jobs_all)

    # FIXED: shard by stable key (pkl_path), not by list index.
    mut_jobs = shard_by_key(
        mut_jobs_all,
        array_ID,
        array_N,
        key_fn=lambda job: job[3],  # pkl_path
    )

    print(
        f"MUT jobs total (after num_samples)={total_mut_all} | "
        f"this shard MUT jobs={len(mut_jobs)}",
        flush=True,
    )

    # -------------------------
    # Execute
    # -------------------------
    start_time = time.time()

    run_jobs_with_model(
        model=model,
        jobs=wt_rep_jobs,
        max_tokens_per_batch=max_tokens_per_batch,
        num_recycles=num_recycles,
        start_time=start_time,
        label="WT",
    )

    # Safe, idempotent propagation
    propagate_wt_outputs(wt_groups, overwrite=overwrite_wt_propagation)

    run_jobs_with_model(
        model=model,
        jobs=mut_jobs,
        max_tokens_per_batch=max_tokens_per_batch,
        num_recycles=num_recycles,
        start_time=start_time,
        label="MUT",
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_parent", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_tokens_per_batch", type=int, default=1024)
    parser.add_argument("--num_recycles", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--start_from_longest", action="store_true")

    # Array job parameters
    parser.add_argument("--ID", type=int, default=None, help="SLURM array job index (0-based)")
    parser.add_argument("--N", type=int, default=None, help="Total number of array jobs")

    # WT propagation behavior
    parser.add_argument("--overwrite_wt_propagation", action="store_true",
                        help="Overwrite existing wt_esmf.* files when propagating to duplicate wt dirs.")

    args = parser.parse_args()

    # Get SLURM_ARRAY_TASK_ID from environment if --ID not provided
    if args.ID is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if slurm_id is not None:
            args.ID = int(slurm_id)

    main(
        data_parent=args.data_parent,
        num_samples=args.num_samples,
        max_tokens_per_batch=args.max_tokens_per_batch,
        num_recycles=args.num_recycles,
        chunk_size=args.chunk_size,
        cpu_only=args.cpu_only,
        start_from_longest=args.start_from_longest,
        array_ID=args.ID,
        array_N=args.N,
        overwrite_wt_propagation=args.overwrite_wt_propagation,
    )
