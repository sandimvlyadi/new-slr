#!/usr/bin/env python3
"""Colab CLI Controller for Master's Thesis Experiments (RQ0-RQ6).

Automates:
  1. `status`:      check Colab compute unit balance and active sessions (`colab usage`, `colab sessions`)
  2. `sync-gdrive`: upload `data/colab_data_bundle.tar.gz` and `artifacts/encoders` to `gdrive:new-slr-experiments/`
  3. `bootstrap`:   allocate a Colab GPU VM (`colab new -s <session> --gpu <L4|T4|A100>`), install dependencies,
                    upload `ramscore` package + extract dataset bundle into `/content/data`
  4. `run-queue`:   run a batch of experiment arms (`rq0`, `rq1`, `rq2`, `rq3`, `rq4`, `rq5`, `rq6`) on the remote
                    Colab session, downloading each completed run's metrics/predictions/logits to local `runs/`
                    immediately after each run finishes, and optionally stopping the VM (`--auto-stop`) at the end.
  5. `stop`:        immediately release the Colab VM (`colab stop -s <session>`) so 0 compute units are wasted.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
COLAB_BIN = os.environ.get('COLAB_BIN', '/home/sandimulyadi/.local/bin/colab')
RCLONE_CONF = os.environ.get('RCLONE_CONF', '/home/sandimulyadi/.config/rclone/colab.conf')
GDRIVE_REMOTE = 'gdrive:new-slr-experiments'

# Complete arm definitions for RQ0, RQ2, RQ3, RQ4, RQ5
DESIGNS_6 = [
    ('0.1', 0, 42),
    ('0.1', 1, 42),
    ('0.1', 2, 42),
    ('1.0', 0, 42),
    ('1.0', 0, 1337),
    ('1.0', 0, 2026),
]

COMMON_FLAGS = [
    '--amp', '--n-boot', '0', '--epochs', '50', '--min-steps', '2000',
    '--batch-size', '32', '--lr', '1e-4', '--weight-decay', '1e-3',
    '--aug', 'base', '--seed-aug', '--monitor', 'qwk',
    '--data-root', '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
]

RQ0_ARMS = {
    'rq0-01': ('resnet34', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1']),
    'rq0-02': ('convnext_tiny', ['--model', 'convnext_tiny.fb_in1k', '--pretrained', '--head', 'softmax', '--logit-adjust', '1']),
}

RQ2_ARMS = {
    'rq2-01': ('resnet34_ce', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '0']),
    'rq2-02': ('resnet34_b1', ['--model', 'resnet34', '--pretrained', '--head', 'ordinal', '--class-weight', '--oversample', '1']),
    'rq2-03': ('resnet34_oversample', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--oversample', '1']),
    'rq2-04': ('resnet34_cweight', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--class-weight']),
    'rq2-05': ('resnet34_corn', ['--model', 'resnet34', '--pretrained', '--head', 'corn']),
    'rq2-06': ('resnet34_corn_la', ['--model', 'resnet34', '--pretrained', '--head', 'corn', '--logit-adjust', '1']),
    'rq2-08': ('resnet34_urot', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--undersample-zero', 'rotate']),
    'rq2-09': ('resnet34_focal', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--focal-gamma', '2.0']),
    'rq2-10': ('resnet34_focal_ov', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--focal-gamma', '2.0', '--oversample', '1']),
    'rq2-11': ('resnet34_fol', ['--model', 'resnet34', '--pretrained', '--head', 'ordinal', '--focal-gamma', '2.0']),
    'rq2-12': ('resnet34_cumlink', ['--model', 'resnet34', '--pretrained', '--head', 'cumulative_link']),
    'rq2-13': ('resnet34_softord', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--soft-label-sigma', '0.5']),
    'rq2-14': ('resnet34_ldam_drw', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--ldam-drw']),
    'rq2-15': ('resnet34_cbloss', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--effective-beta', '0.9999']),
    'rq2-16': ('resnet34_b2_cornreg', ['--model', 'resnet34', '--pretrained', '--head', 'softmax_corn', '--logit-adjust', '1', '--corn-reg-weight', '0.5']),
    'rq2-17': ('resnet34_joint_la', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--joint-logit-adjust', '1.0']),
}

RQ3_TRAIN_ARMS = {
    'rq3-04': ('resnet34_aug_strong', [
        '--amp', '--n-boot', '0', '--epochs', '50', '--min-steps', '2000',
        '--batch-size', '32', '--lr', '1e-4', '--weight-decay', '1e-3',
        '--aug', 'strong', '--seed-aug', '--monitor', 'qwk',
        '--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1',
        '--data-root', '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
    ]),
    'rq3-05': ('resnet34_aug_rvmatch', [
        '--amp', '--n-boot', '0', '--epochs', '50', '--min-steps', '2000',
        '--batch-size', '32', '--lr', '1e-4', '--weight-decay', '1e-3',
        '--aug', 'rheumavit_match', '--seed-aug', '--monitor', 'qwk',
        '--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1',
        '--data-root', '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
    ]),
}

RQ4_TRAIN_ARMS = {
    'rq4-01': ('resnet34_mcdrop', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--drop-rate', '0.2']),
}

RQ5_TRAIN_ARMS = {
    'rq5-01': ('resnet34_cl_relabel', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--label-overrides', '/content/runs/rq5/oof/cleanlab_relabel.csv']),
    'rq5-02': ('resnet34_cl_prune', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--label-overrides', '/content/runs/rq5/oof/cleanlab_prune.csv']),
    'rq5-03': ('resnet34_cl_downweight', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--label-overrides', '/content/runs/rq5/oof/cleanlab_downweight.csv']),
    'rq5-04': ('resnet34_ls01', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--label-smoothing', '0.1']),
    'rq5-05': ('resnet34_disagree_soft', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--disagreement-soft-label']),
    'rq5-06': ('resnet34_conf_split', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--label-overrides', '/content/runs/rq5/oof/conf_split.csv']),
    'rq5-07': ('resnet34_long_rel', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--label-overrides', '/content/runs/rq5/oof/longitudinal_reliability.csv']),
    'rq5-08': ('resnet34_gce', COMMON_FLAGS + ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--gce-q', '0.7']),
}


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f'$ {" ".join(cmd)}', flush=True)
    return subprocess.run(cmd, check=check)


def cmd_status(_args):
    run_cmd([COLAB_BIN, 'usage'], check=False)
    run_cmd([COLAB_BIN, 'sessions'], check=False)


def cmd_sync_gdrive(_args):
    bundle = ROOT / 'data' / 'colab_data_bundle.tar.gz'
    if not bundle.exists():
        raise SystemExit(f'{bundle} not found; run scripts/prepare_data_bundle.py first')
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, str(bundle), f'{GDRIVE_REMOTE}/data/'])
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, str(ROOT / 'data' / 'colab_data_bundle.sha256'), f'{GDRIVE_REMOTE}/data/'])
    if (ROOT / 'artifacts' / 'encoders').exists():
        run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, str(ROOT / 'artifacts'), f'{GDRIVE_REMOTE}/artifacts/'])
    print('Synced dataset bundle and E1 SSL encoders to Google Drive.', flush=True)


def cmd_bootstrap(args):
    s = args.session
    if args.new:
        run_cmd([COLAB_BIN, 'new', '-s', s, '--gpu', args.gpu])
    run_cmd([COLAB_BIN, 'install', '-s', s, 'timm', 'albumentations', 'opencv-python-headless', 'scikit-learn', 'scipy'])

    # Package ramscore code and upload to /content/code.tar.gz
    code_tar = Path(tempfile.gettempdir()) / 'ramscore_code.tar.gz'
    subprocess.run(['tar', '-czf', str(code_tar), '-C', str(ROOT), 'code'], check=True)
    run_cmd([COLAB_BIN, 'upload', '-s', s, str(code_tar), '/content/ramscore_code.tar.gz'])

    bundle = ROOT / 'data' / 'colab_data_bundle.tar.gz'
    run_cmd([COLAB_BIN, 'upload', '-s', s, str(bundle), '/content/colab_data_bundle.tar.gz'])

    setup_py = Path(tempfile.gettempdir()) / 'colab_setup_remote.py'
    setup_py.write_text(
        "import subprocess, pathlib, torch\n"
        "pathlib.Path('/content/data').mkdir(exist_ok=True)\n"
        "subprocess.run(['tar', '-xzf', '/content/ramscore_code.tar.gz', '-C', '/content'], check=True)\n"
        "subprocess.run(['tar', '-xzf', '/content/colab_data_bundle.tar.gz', '-C', '/content/data'], check=True)\n"
        "print('CUDA available:', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\n"
    )
    run_cmd([COLAB_BIN, 'exec', '-s', s, '-f', str(setup_py)])


def run_remote_train(session: str, stage: str, run_name: str, cli_args: list[str], keep_ckpt: bool = False):
    local_run_dir = ROOT / 'runs' / stage / run_name
    if (local_run_dir / 'metrics.json').exists():
        print(f'[SKIP] {stage}/{run_name} already exists locally.', flush=True)
        return

    remote_out = f'/content/runs/{stage}'
    full_argv = ['--name', run_name, '--out', remote_out] + cli_args
    runner_py = Path(tempfile.gettempdir()) / f'exec_{run_name}.py'
    runner_py.write_text(
        "import sys\n"
        "sys.path.insert(0, '/content/code/src')\n"
        "from ramscore.train import main\n"
        f"main({full_argv!r})\n"
    )
    t0 = time.time()
    run_cmd([COLAB_BIN, 'exec', '-s', session, '-f', str(runner_py)])
    dt = (time.time() - t0) / 60.0
    print(f'[DONE] {stage}/{run_name} finished on Colab in {dt:.1f} min; downloading artifacts ...', flush=True)

    local_run_dir.mkdir(parents=True, exist_ok=True)
    files = ['args.json', 'history.csv', 'metrics.json', 'test_predictions.csv', 'val_logits.npz', 'test_logits.npz']
    if keep_ckpt:
        files.append('model_best.pt')
    for fname in files:
        run_cmd([COLAB_BIN, 'download', '-s', session, f'{remote_out}/{run_name}/{fname}', str(local_run_dir / fname)], check=False)


def cmd_run_queue(args):
    s = args.session
    stage = args.stage
    try:
        if stage == 'rq0':
            for frac, draw, seed in DESIGNS_6:
                for arm, (tag, flags) in RQ0_ARMS.items():
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    run_remote_train(s, 'rq0', name, cli, keep_ckpt=(arm == 'rq0-01'))
        elif stage == 'rq2':
            selected = args.arms.split(',') if args.arms else list(RQ2_ARMS.keys())
            for frac, draw, seed in DESIGNS_6:
                for arm in selected:
                    if arm == 'rq2-07':
                        # Decoupled cRT (Kang et al., 2020): stage 2 from rq2-01 of the same design
                        base_name = f'rq2-01_resnet34_ce_b2_f{frac}_d{draw}_s{seed}'
                        name = f'rq2-07_resnet34_crt_b2_f{frac}_d{draw}_s{seed}'
                        cli = [
                            '--amp', '--n-boot', '0', '--epochs', '10', '--min-steps', '400',
                            '--batch-size', '32', '--lr', '1e-3', '--weight-decay', '1e-3',
                            '--aug', 'base', '--seed-aug', '--monitor', 'qwk',
                            '--data-root', '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
                            '--model', 'resnet34', '--init-from', f'/content/runs/rq2/{base_name}/model_best.pt',
                            '--freeze-backbone', '--head', 'softmax', '--oversample', '1',
                            '--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed),
                        ]
                        run_remote_train(s, 'rq2', name, cli, keep_ckpt=False)
                        continue
                    tag, flags = RQ2_ARMS[arm]
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = COMMON_FLAGS + flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    run_remote_train(s, 'rq2', name, cli, keep_ckpt=(arm == 'rq2-01'))
        elif stage == 'rq5-oof':
            runner_py = Path(tempfile.gettempdir()) / 'exec_rq5_oof.py'
            runner_py.write_text(
                "import sys\n"
                "sys.path.insert(0, '/content/code/src')\n"
                "from ramscore.rq5 import run_oof_and_overrides\n"
                "res = run_oof_and_overrides(\n"
                "    data_root='/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',\n"
                "    out_dir='/content/runs/rq5/oof',\n"
                ")\n"
                "print('OOF summary:', res)\n"
            )
            run_cmd([COLAB_BIN, 'exec', '-s', s, '-f', str(runner_py)])
            local_oof = ROOT / 'runs' / 'rq5' / 'oof'
            local_oof.mkdir(parents=True, exist_ok=True)
            for fname in [
                'oof_probs.npz', 'oof_summary.csv', 'summary.json',
                'cleanlab_relabel.csv', 'cleanlab_prune.csv', 'cleanlab_downweight.csv',
                'conf_split.csv', 'longitudinal_reliability.csv',
            ]:
                run_cmd([COLAB_BIN, 'download', '-s', s, f'/content/runs/rq5/oof/{fname}', str(local_oof / fname)], check=False)
        elif stage == 'rq5-mt':
            for frac in ('0.1', '0.25'):
                for draw in (0, 1, 2):
                    name = f'rq5-11_resnet34_meanteacher_b2_f{frac}_d{draw}_s42'
                    local_dir = ROOT / 'runs' / 'rq5' / name
                    if (local_dir / 'metrics.json').exists():
                        print(f'[SKIP] rq5/{name} already exists locally.', flush=True)
                        continue
                    runner_py = Path(tempfile.gettempdir()) / f'exec_{name}.py'
                    runner_py.write_text(
                        "import sys\n"
                        "sys.path.insert(0, '/content/code/src')\n"
                        "from ramscore.rq5 import train_mean_teacher\n"
                        f"train_mean_teacher(name={name!r}, out='/content/runs/rq5', "
                        "data_root='/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring', "
                        f"label_fraction={float(frac)}, fraction_seed={draw}, seed=42)\n"
                    )
                    run_cmd([COLAB_BIN, 'exec', '-s', s, '-f', str(runner_py)])
                    local_dir.mkdir(parents=True, exist_ok=True)
                    for fname in ['args.json', 'metrics.json', 'test_predictions.csv']:
                        run_cmd([COLAB_BIN, 'download', '-s', s, f'/content/runs/rq5/{name}/{fname}', str(local_dir / fname)], check=False)
        elif stage == 'rq3-train':
            for frac, draw, seed in DESIGNS_6:
                for arm, (tag, flags) in RQ3_TRAIN_ARMS.items():
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    run_remote_train(s, 'rq3', name, cli, keep_ckpt=True)
        elif stage == 'rq4-train':
            for frac, draw, seed in DESIGNS_6:
                for arm, (tag, flags) in RQ4_TRAIN_ARMS.items():
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    run_remote_train(s, 'rq4', name, cli, keep_ckpt=True)
        elif stage == 'rq5-train':
            selected = args.arms.split(',') if args.arms else list(RQ5_TRAIN_ARMS.keys())
            for frac, draw, seed in DESIGNS_6:
                for arm in selected:
                    tag, flags = RQ5_TRAIN_ARMS[arm]
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    run_remote_train(s, 'rq5', name, cli, keep_ckpt=False)
        else:
            raise SystemExit(f'Unknown stage {stage}')
    finally:
        if args.auto_stop:
            print(f'[AUTO-STOP] Releasing Colab session {s} ...', flush=True)
            run_cmd([COLAB_BIN, 'stop', '-s', s], check=False)


def cmd_stop(args):
    run_cmd([COLAB_BIN, 'stop', '-s', args.session])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    sub.add_parser('status', help='Show Colab compute balance and active sessions')
    sub.add_parser('sync-gdrive', help='Upload dataset bundle and E1 encoders to Google Drive via rclone')

    p_boot = sub.add_parser('bootstrap', help='Provision/prepare a Colab GPU session with code and dataset')
    p_boot.add_argument('-s', '--session', default='slr-gpu')
    p_boot.add_argument('--gpu', default='L4', choices=['T4', 'L4', 'A100'])
    p_boot.add_argument('--new', action='store_true', help='Allocate a new Colab VM first')

    p_run = sub.add_parser('run-queue', help='Execute an experiment queue on the remote Colab session')
    p_run.add_argument('-s', '--session', default='slr-gpu')
    p_run.add_argument('--stage', required=True, choices=['rq0', 'rq2', 'rq3-train', 'rq4-train', 'rq5-oof', 'rq5-train', 'rq5-mt'])
    p_run.add_argument('--arms', default='', help='Optional comma-separated subset of arms (e.g. rq2-01,rq2-05)')
    p_run.add_argument('--auto-stop', action='store_true', help='Automatically stop the Colab VM when done')

    p_stop = sub.add_parser('stop', help='Stop and release a Colab VM session')
    p_stop.add_argument('-s', '--session', default='slr-gpu')

    args = ap.parse_args(argv)
    if args.command == 'status':
        cmd_status(args)
    elif args.command == 'sync-gdrive':
        cmd_sync_gdrive(args)
    elif args.command == 'bootstrap':
        cmd_bootstrap(args)
    elif args.command == 'run-queue':
        cmd_run_queue(args)
    elif args.command == 'stop':
        cmd_stop(args)


if __name__ == '__main__':
    main()
