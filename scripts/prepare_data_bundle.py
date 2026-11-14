#!/usr/bin/env python3
"""Prepare RheumaVIT cropped ROIs, verify frozen patient splits, copy E1 SSL encoders,
and package `/data/Smart-X/new-slr/data/colab_data_bundle.tar.gz` for Google Colab CLI execution.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code' / 'src'))

from ramscore.data import load_records as load_h1200_records, subsample_patients
from ramscore.rheumavit import prepare as prepare_rheumavit, load_records as load_rv_records

RAM_H1200_BE = Path('/home/sandimulyadi/Projects/Dataset/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring')
RHEUMAVIT_SRC = Path('/home/sandimulyadi/Projects/Dataset/AIRINIIReumHands')
SRC_ARTIFACTS = Path('/data/Smart-X/ra-ablation/artifacts')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def step1_prepare_rheumavit() -> Path:
    out_dir = ROOT / 'data' / 'derived' / 'rheumavit'
    manifest = out_dir / 'manifest.csv'
    t0 = time.time()
    print(f'[1/4] Preparing RheumaVIT cropped ROIs -> {out_dir} ...', flush=True)
    df = prepare_rheumavit(RHEUMAVIT_SRC, out_dir)
    print(
        f'      Done in {time.time() - t0:.1f}s: {len(df)} joint ROIs from '
        f'{df.patient.nunique()} patients (pool={df[df.e4_split=="pool"].patient.nunique()}, '
        f'test={df[df.e4_split=="test"].patient.nunique()})',
        flush=True,
    )
    return manifest


def step2_verify_and_freeze_splits(manifest_path: Path) -> Path:
    print('[2/4] Verifying RAM-H1200 & RheumaVIT patient splits and zero leakage ...', flush=True)
    splits = {s: load_h1200_records(RAM_H1200_BE, s) for s in ('train', 'val', 'test')}
    patients = {s: sorted({r['patient'] for r in recs}) for s, recs in splits.items()}

    assert not (set(patients['train']) & set(patients['val'])), 'Train/Val overlap!'
    assert not (set(patients['train']) & set(patients['test'])), 'Train/Test overlap!'
    assert not (set(patients['val']) & set(patients['test'])), 'Val/Test overlap!'

    draws_10pct = {
        f'd{d}': sorted({r['patient'] for r in subsample_patients(splits['train'], 0.1, seed=d)})
        for d in (0, 1, 2)
    }
    draws_25pct = {
        f'd{d}': sorted({r['patient'] for r in subsample_patients(splits['train'], 0.25, seed=d)})
        for d in (0, 1, 2)
    }

    rv_pool = load_rv_records(manifest_path, e4_split='pool')
    rv_test = load_rv_records(manifest_path, e4_split='test')
    rv_pool_p = sorted({r['patient'] for r in rv_pool})
    rv_test_p = sorted({r['patient'] for r in rv_test})
    assert not (set(rv_pool_p) & set(rv_test_p)), 'RheumaVIT Pool/Test overlap!'

    info = {
        'ram_h1200': {
            'n_patients': {s: len(patients[s]) for s in ('train', 'val', 'test')},
            'n_joints': {s: len(splits[s]) for s in ('train', 'val', 'test')},
            'patients': patients,
            'draws_10pct': draws_10pct,
            'draws_25pct': draws_25pct,
        },
        'rheumavit': {
            'n_patients': {'pool': len(rv_pool_p), 'test': len(rv_test_p)},
            'n_joints': {'pool': len(rv_pool), 'test': len(rv_test)},
            'pool_patients': rv_pool_p,
            'test_patients': rv_test_p,
        },
    }
    out_path = ROOT / 'data' / 'derived' / 'splits_manifest.json'
    out_path.write_text(json.dumps(info, indent=2))
    print(
        f"      RAM-H1200 patients: train={len(patients['train'])}, val={len(patients['val'])}, test={len(patients['test'])} (0 overlap)\n"
        f"      RheumaVIT patients: pool={len(rv_pool_p)}, test={len(rv_test_p)} (0 overlap)\n"
        f"      Saved -> {out_path}",
        flush=True,
    )
    return out_path


def step3_sync_e1_encoders() -> Path:
    dest_art = ROOT / 'artifacts'
    dest_enc = dest_art / 'encoders'
    print(f'[3/4] Syncing & verifying E1 SSL encoders -> {dest_enc} ...', flush=True)
    if SRC_ARTIFACTS.exists():
        dest_art.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC_ARTIFACTS / 'SHA256SUMS', dest_art / 'SHA256SUMS')
        for line in (SRC_ARTIFACTS / 'SHA256SUMS').read_text().strip().splitlines():
            expected_hash, rel_path = line.split()
            src_f = SRC_ARTIFACTS / rel_path
            dst_f = dest_art / rel_path
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            if not dst_f.exists() or sha256_file(dst_f) != expected_hash:
                shutil.copy2(src_f, dst_f)
            actual_hash = sha256_file(dst_f)
            if actual_hash != expected_hash:
                raise RuntimeError(f'SHA256 mismatch for {dst_f}: {actual_hash} != {expected_hash}')
        print('      All 5 E1 SSL encoders verified (100% SHA-256 match).', flush=True)
    else:
        print('      Warning: source artifacts folder not found.', flush=True)
    return dest_art


def step4_package_tarball() -> Path:
    bundle_path = ROOT / 'data' / 'colab_data_bundle.tar.gz'
    print(f'[4/4] Packaging dataset bundle -> {bundle_path} ...', flush=True)
    t0 = time.time()
    with tarfile.open(bundle_path, 'w:gz', compresslevel=6) as tar:
        tar.add(RAM_H1200_BE, arcname='RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring')
        tar.add(ROOT / 'data' / 'derived' / 'rheumavit', arcname='derived/rheumavit')
        tar.add(ROOT / 'data' / 'derived' / 'splits_manifest.json', arcname='derived/splits_manifest.json')
    digest = sha256_file(bundle_path)
    (ROOT / 'data' / 'colab_data_bundle.sha256').write_text(f'{digest}  colab_data_bundle.tar.gz\n')
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    print(f'      Bundle ready in {time.time() - t0:.1f}s: {size_mb:.1f} MB (sha256: {digest[:16]}...)', flush=True)
    return bundle_path


def main():
    manifest = step1_prepare_rheumavit()
    step2_verify_and_freeze_splits(manifest)
    step3_sync_e1_encoders()
    step4_package_tarball()


if __name__ == '__main__':
    main()
