import os
import sys
import argparse
import hashlib
import shutil
from typing import List, Dict, Tuple

# -------------------------
# Helpers
# -------------------------

def sha1_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def copy_if_missing(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Returns True if copied, False otherwise.
    """
    if not os.path.exists(src):
        return False
    if os.path.exists(dst) and not overwrite:
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True

def stable_sorted(paths: List[str]) -> List[str]:
    return sorted(paths, key=lambda x: x.lower())

def shard_by_index(items: List, shard_id: int, num_shards: int) -> List:
    return [x for i, x in enumerate(items) if (i % num_shards) == shard_id]

def add_feature_script_dir_to_path(feature_script_dir: str) -> None:
    """
    Make imports like `import generate_features.coordinate as coord` work
    even when this script is run from elsewhere.
    """
    # If feature_script_dir is "generate_features", it should be importable as a package.
    # If it's a parent folder containing generate_features/, also support that.
    feature_script_dir = os.path.abspath(feature_script_dir)

    if os.path.isdir(feature_script_dir):
        # If this is the package dir itself (contains __init__.py), add its parent.
        if os.path.exists(os.path.join(feature_script_dir, "__init__.py")):
            parent = os.path.dirname(feature_script_dir)
            if parent not in sys.path:
                sys.path.insert(0, parent)
        else:
            # Otherwise add it directly (maybe it contains generate_features/)
            if feature_script_dir not in sys.path:
                sys.path.insert(0, feature_script_dir)

# -------------------------
# Feature runners (same logic as your original, but param-driven)
# -------------------------

def collect_variant_fastas(sample_dirs: List[str], variant: str) -> Tuple[List[str], List[str]]:
    fasta_files = []
    parent_dirs = []
    for sample_dir in sample_dirs:
        vdir = os.path.join(sample_dir, variant)
        fasta = "wt.fasta" if variant == "wt_data" else "mut.fasta"
        fastapath = os.path.join(vdir, fasta)
        if os.path.exists(fastapath):
            fasta_files.append(fastapath)
            parent_dirs.append(vdir)
    return fasta_files, parent_dirs

def batch_run_esm2(fasta_files: List[str], parent_dirs: List[str], batch_size: int = 8) -> None:
    from generate_features.esm2_embedding import run_esm2_embedding

    missing = []
    for fastapath, vdir in zip(fasta_files, parent_dirs):
        outpath = os.path.join(vdir, "esm2.pt")
        if not os.path.exists(outpath):
            missing.append((fastapath, vdir))

    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        input_files = [f for f, _ in batch]
        out_dirs = [d for _, d in batch]
        print(f"Running ESM2 batch: {input_files}", flush=True)
        run_esm2_embedding(input_files, out_dirs)

def run_fixed_embedding(fasta_files: List[str], parent_dirs: List[str]) -> None:
    import generate_features.fixed_embedding as fe
    for fastapath, vdir in zip(fasta_files, parent_dirs):
        outpath = os.path.join(vdir, "fixed_embedding.pt")
        if not os.path.exists(outpath):
            print(f"Running fixed embedding for: {fastapath}", flush=True)
            fe.main.callback(fasta_file=fastapath, saved_folder=vdir)

def run_coordinate_and_pair_for_variant(vdirs: List[str], variant: str) -> None:
    import generate_features.coordinate as coord
    import generate_features.pair as pair

    pdb_name = "wt_esmf.pdb" if variant == "wt_data" else "mut_rosetta.pdb"#"mut_esmf.pdb"

    for vdir in vdirs:
        pdbpath = os.path.join(vdir, pdb_name)
        coord_path = os.path.join(vdir, "coordinate_rosetta.pt")
        pair_path = os.path.join(vdir, "pair_rosetta.pt")

        if not os.path.exists(pdbpath):
            continue

        if not os.path.exists(coord_path):
            print(f"Running coordinate for: {pdbpath}", flush=True)
            coord.main.callback(pdb_file=pdbpath, saved_folder=vdir, out_fn=coord_path)

        if os.path.exists(coord_path) and not os.path.exists(pair_path):
            print(f"Running pair for: {coord_path}", flush=True)
            pair.main.callback(coordinate_file=coord_path, saved_folder=vdir, out_fn=pair_path)

# -------------------------
# WT de-dup logic
# -------------------------

def group_wt_by_fasta_hash(sample_dirs: List[str]) -> Dict[str, List[str]]:
    """
    Groups wt_data directories by the hash of wt.fasta contents.
    Returns: {wt_hash: [wt_data_dir1, wt_data_dir2, ...]}
    """
    groups: Dict[str, List[str]] = {}
    for sample_dir in sample_dirs:
        wt_dir = os.path.join(sample_dir, "wt_data")
        wt_fa = os.path.join(wt_dir, "wt.fasta")
        if not os.path.exists(wt_fa):
            continue
        h = sha1_file(wt_fa)
        groups.setdefault(h, []).append(wt_dir)

    # stable ordering inside each group
    for h in list(groups.keys()):
        groups[h] = stable_sorted(groups[h])

    return groups

def propagate_wt_outputs(src_wt_dir: str, dst_wt_dirs: List[str], overwrite: bool = False) -> None:
    """
    Copy WT outputs computed in src_wt_dir into every dir in dst_wt_dirs.
    We only propagate WT artifacts:
      - esm2.pt
      - fixed_embedding.pt
      - coordinate.pt
      - pair.pt
    """
    artifacts = ["esm2.pt", "fixed_embedding.pt", "coordinate.pt", "pair.pt"]
    for dst in dst_wt_dirs:
        if dst == src_wt_dir:
            continue
        for a in artifacts:
            src = os.path.join(src_wt_dir, a)
            dstp = os.path.join(dst, a)
            copied = copy_if_missing(src, dstp, overwrite=overwrite)
            if copied:
                print(f"Propagated {a}: {src_wt_dir} -> {dst}", flush=True)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_script_dir", type=str, default="generate_features",
                    help="Path to feature scripts package dir (or parent dir containing it).")
    ap.add_argument("--data_parent_dir", type=str, required=True,
                    help="Parent directory containing sample folders.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total shards for array jobs.")
    ap.add_argument("--shard_id", type=int, default=None,
                    help="Shard index (0-based). If omitted, uses SLURM_ARRAY_TASK_ID if set, else 0.")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for ESM2 embedding calls.")
    ap.add_argument("--overwrite_propagation", action="store_true",
                    help="If set, propagation overwrites existing WT outputs in duplicate dirs.")
    args = ap.parse_args()

    shard_id = args.shard_id
    if shard_id is None:
        shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    num_shards = max(1, int(args.num_shards))
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"--shard_id must be in [0, {num_shards-1}] but got {shard_id}")

    add_feature_script_dir_to_path(args.feature_script_dir)

    data_parent_dir = os.path.abspath(args.data_parent_dir)
    if not os.path.isdir(data_parent_dir):
        raise FileNotFoundError(f"DATA_PARENT_DIR not found: {data_parent_dir}")

    # Identify sample folders (same criterion as your original)
    sample_dirs = []
    for name in os.listdir(data_parent_dir):
        p = os.path.join(data_parent_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "mut_info.csv")):
            sample_dirs.append(p)
    sample_dirs = stable_sorted(sample_dirs)

    print(f"[shard {shard_id}/{num_shards}] Found {len(sample_dirs)} sample dirs", flush=True)

    # -------------------------
    # WT: de-dup by wt.fasta hash, then shard by GROUP (not by sample)
    # -------------------------
    wt_groups = group_wt_by_fasta_hash(sample_dirs)
    wt_hashes = stable_sorted(list(wt_groups.keys()))
    my_wt_hashes = shard_by_index(wt_hashes, shard_id, num_shards)

    print(f"[shard {shard_id}/{num_shards}] WT unique groups: {len(wt_hashes)}; "
          f"this shard will compute: {len(my_wt_hashes)}", flush=True)

    for h in my_wt_hashes:
        wt_dirs = wt_groups[h]
        rep_wt_dir = wt_dirs[0]  # stable representative
        rep_wt_fa = os.path.join(rep_wt_dir, "wt.fasta")

        # Compute WT features ONCE (in representative dir)
        # ESM2 + fixed embedding
        batch_run_esm2([rep_wt_fa], [rep_wt_dir], batch_size=1)
        run_fixed_embedding([rep_wt_fa], [rep_wt_dir])

        # Coordinate + pair for WT (once)
        run_coordinate_and_pair_for_variant([rep_wt_dir], variant="wt_data")

        # Propagate results to all duplicates (including directories that may be in other shards)
        propagate_wt_outputs(rep_wt_dir, wt_dirs, overwrite=args.overwrite_propagation)

    # -------------------------
    # MUT: shard by sample_dirs and compute per-sample (no de-dup)
    # -------------------------
    my_sample_dirs = shard_by_index(sample_dirs, shard_id, num_shards)
    print(f"[shard {shard_id}/{num_shards}] MUT samples this shard: {len(my_sample_dirs)}", flush=True)

    mut_fastas, mut_dirs = collect_variant_fastas(my_sample_dirs, "mut_data")
    # if mut_fastas:
    #     batch_run_esm2(mut_fastas, mut_dirs, batch_size=args.batch_size)
    #     run_fixed_embedding(mut_fastas, mut_dirs)

    # Coordinate + pair for MUT per-sample
    mut_vdirs = [os.path.join(sd, "mut_data") for sd in my_sample_dirs]
    run_coordinate_and_pair_for_variant(mut_vdirs, variant="mut_data")

    print(f"[shard {shard_id}/{num_shards}] Done.", flush=True)

if __name__ == "__main__":
    main()
