#!/usr/bin/env python3
"""Colab CLI Controller for Master's Thesis Experiments (RQ0-RQ5).

Automates:
  1. `status`:      check Colab compute unit balance, active sessions, and completed runs (`colab usage`, `colab sessions`)
  2. `sync-gdrive`: upload `data/colab_data_bundle.tar.gz` and `artifacts/encoders` to `gdrive:new-slr-experiments/`
  3. `bootstrap`:   allocate a Colab GPU VM (`colab new -s <session> --gpu <L4|T4|A100>`), install dependencies,
                    upload `ramscore` package + extract dataset bundle & E1 SSL encoders into `/content/`
  4. `run-queue`:   run a batch of experiment arms (`rq0`, `rq1`, `rq2`, `rq2-inter`, `rq3-train`, `rq3-eval`, `rq4-train`,
                    `rq5-oof`, `rq5-train`, `rq5-mt`, `step2-and-3`) on the remote Colab session, streaming live epoch logs
                    to `/data/Smart-X/new-slr/runs/colab_live.log`, backing up each finished run to
                    `gdrive:new-slr-experiments/runs/`, and syncing locally.
  5. `stop`:        immediately release the Colab VM (`colab stop -s <session>`) so 0 compute units are wasted.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
COLAB_BIN = os.environ.get('COLAB_BIN', '/home/sandimulyadi/.local/bin/colab')
RCLONE_CONF = os.environ.get('RCLONE_CONF', '/home/sandimulyadi/.config/rclone/colab.conf')
GDRIVE_REMOTE = 'gdrive:new-slr-experiments'
LIVE_LOG = ROOT / 'runs' / 'colab_live.log'


def get_hf_token() -> str:
    if os.environ.get('HF_TOKEN'):
        return os.environ['HF_TOKEN'].strip()
    env_file = ROOT / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith('HF_TOKEN='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    hf_cache = Path.home() / '.cache' / 'huggingface' / 'token'
    if hf_cache.exists():
        return hf_cache.read_text().strip()
    return ''


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

RQ1_MAIN_ARMS = {
    'rq1-01': ('resnet34_scratch', ['--model', 'resnet34', '--head', 'softmax', '--logit-adjust', '1']),
    'rq1-02': ('resnet34_dino', ['--model', 'resnet34', '--init-from', '/content/artifacts/encoders/e1_dino_resnet34/encoder.pt', '--head', 'softmax', '--logit-adjust', '1']),
    'rq1-03': ('resnet34_dinochrono', ['--model', 'resnet34', '--init-from', '/content/artifacts/encoders/e1_dinochrono_resnet34/encoder.pt', '--head', 'softmax', '--logit-adjust', '1']),
    'rq1-04': ('vits_imagenet', ['--model', 'vit_small_patch16_224', '--pretrained', '--layer-decay', '0.75', '--warmup', '0.1', '--head', 'softmax', '--logit-adjust', '1']),
    'rq1-05': ('vits_mae', ['--model', 'vit_small_patch16_224', '--init-from', '/content/artifacts/encoders/e1_mae_vits/encoder.pt', '--layer-decay', '0.75', '--warmup', '0.1', '--head', 'softmax', '--logit-adjust', '1']),
    'rq1-06': ('vits_dino', ['--model', 'vit_small_patch16_224', '--init-from', '/content/artifacts/encoders/e1_dino_vits/encoder.pt', '--layer-decay', '0.75', '--warmup', '0.1', '--head', 'softmax', '--logit-adjust', '1']),
    'rq1-12': ('resnet34_supcon', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--supcon-weight', '0.5', '--supcon-temp', '0.1']),
    'rq1-13': ('resnet34_supcon_ord', ['--model', 'resnet34', '--pretrained', '--head', 'softmax', '--logit-adjust', '1', '--supcon-weight', '0.5', '--supcon-temp', '0.1', '--supcon-alpha', '0.2']),
}

RQ1_ENCODER_CONFIGS = {
    'rq0-01': ('resnet34_imagenet', ['--model', 'resnet34', '--pretrained']),
    'rq1-01': ('resnet34_scratch', ['--model', 'resnet34']),
    'rq1-02': ('resnet34_dino', ['--model', 'resnet34', '--init-from', '/content/artifacts/encoders/e1_dino_resnet34/encoder.pt']),
    'rq1-03': ('resnet34_dinochrono', ['--model', 'resnet34', '--init-from', '/content/artifacts/encoders/e1_dinochrono_resnet34/encoder.pt']),
    'rq1-04': ('vits_imagenet', ['--model', 'vit_small_patch16_224', '--pretrained', '--layer-decay', '0.75', '--warmup', '0.1']),
    'rq1-05': ('vits_mae', ['--model', 'vit_small_patch16_224', '--init-from', '/content/artifacts/encoders/e1_mae_vits/encoder.pt', '--layer-decay', '0.75', '--warmup', '0.1']),
    'rq1-06': ('vits_dino', ['--model', 'vit_small_patch16_224', '--init-from', '/content/artifacts/encoders/e1_dino_vits/encoder.pt', '--layer-decay', '0.75', '--warmup', '0.1']),
    'rq1-12': ('resnet34_supcon', ['--model', 'resnet34', '--pretrained', '--supcon-weight', '0.5', '--supcon-temp', '0.1']),
    'rq1-13': ('resnet34_supcon_ord', ['--model', 'resnet34', '--pretrained', '--supcon-weight', '0.5', '--supcon-temp', '0.1', '--supcon-alpha', '0.2']),
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


def log_print(msg: str):
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LIVE_LOG, 'a') as f:
        f.write(line + '\n')


def _kill_pgrp(proc: subprocess.Popen):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_cmd(
    cmd: list[str],
    check: bool = True,
    retries: int = 4,
    tee_live: bool = False,
    idle_timeout: float = 90.0,
) -> tuple[int, str]:
    """Run a command with non-blocking output streaming and immediate kill on `Connection was lost`
    (preventing colab-cli background websocket threads from hanging the process)."""
    for attempt in range(1, retries + 1):
        log_print(f'$ {" ".join(cmd)}')
        LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        collected: list[str] = []
        lost_conn = False
        last_io = time.monotonic()

        with open(LIVE_LOG, 'a') as lf:
            while True:
                events = sel.select(timeout=2.0)
                if events:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    last_io = time.monotonic()
                    collected.append(line)
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if tee_live:
                        lf.write(line)
                        lf.flush()
                    if 'RuntimeError: Connection was lost.' in line or 'Session not found' in line:
                        lost_conn = True
                        time.sleep(0.2)
                        _kill_pgrp(proc)
                        break
                else:
                    if proc.poll() is not None:
                        break
                    if time.monotonic() - last_io > idle_timeout:
                        log_print(f'[TIMEOUT] No output for {idle_timeout:.0f}s; killing process ...')
                        lost_conn = True
                        _kill_pgrp(proc)
                        break

        sel.close()
        try:
            rc = proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            _kill_pgrp(proc)
            rc = 1

        if lost_conn and rc == 0:
            rc = 1
        out_str = ''.join(collected)
        if rc == 0 or not check:
            return rc, out_str
        if attempt < retries:
            log_print(f'[WARN] Command exited with {rc} (attempt {attempt}/{retries}); retrying in 4s ...')
            time.sleep(4)
    raise subprocess.CalledProcessError(rc, cmd)


def run_remote_background(
    session: str,
    job_name: str,
    python_code: str,
    poll_window: int = 25,
    check_session: bool = True,
):
    """Run `python_code` as a detached process on the Colab VM so that any websocket drops never interrupt
    GPU training, while keeping the Colab Jupyter kernel actively busy and streaming logs in real time."""
    if check_session:
        ensure_session(session)

    payload_local = Path(tempfile.gettempdir()) / f'payload_{job_name}.py'
    payload_remote = f'/content/payload_{job_name}.py'
    log_remote = f'/content/payload_{job_name}.log'
    pid_remote = f'/content/payload_{job_name}.pid'
    rc_remote = f'/content/payload_{job_name}.rc'

    wrapped_code = (
        "import os, pathlib, sys, traceback\n"
        f"rc_path = pathlib.Path({rc_remote!r})\n"
        "try:\n"
        + "\n".join("    " + ln for ln in python_code.splitlines())
        + "\n    rc_path.write_text('0')\n"
        "except SystemExit as e:\n"
        "    rc_path.write_text(str(e.code if isinstance(e.code, int) else 1))\n"
        "    raise\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    rc_path.write_text('1')\n"
        "    sys.exit(1)\n"
    )
    payload_local.write_text(wrapped_code)
    run_cmd([COLAB_BIN, 'upload', '-s', session, str(payload_local), payload_remote], retries=5, idle_timeout=60.0)
    time.sleep(2)

    # Poll script: starts the detached worker if not already running, then tails its log for `poll_window` seconds
    # inside the Jupyter kernel (keeping Colab status = BUSY and streaming live logs back to local).
    poller_local = Path(tempfile.gettempdir()) / f'poll_{job_name}.py'
    offset_remote = f'/content/payload_{job_name}.offset'

    poller_local.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        f"payload = pathlib.Path({payload_remote!r})\n"
        f"log_f = pathlib.Path({log_remote!r})\n"
        f"pid_f = pathlib.Path({pid_remote!r})\n"
        f"rc_f = pathlib.Path({rc_remote!r})\n"
        f"off_f = pathlib.Path({offset_remote!r})\n"
        "def is_running():\n"
        "    if not pid_f.exists():\n"
        "        return False\n"
        "    try:\n"
        "        pid = int(pid_f.read_text().strip())\n"
        "        os.kill(pid, 0)\n"
        "        cmdline = pathlib.Path(f'/proc/{pid}/cmdline').read_text()\n"
        f"        return 'payload_{job_name}' in cmdline\n"
        "    except Exception:\n"
        "        return False\n"
        "if not is_running() and not rc_f.exists():\n"
        "    log_f.write_text('')\n"
        "    off_f.write_text('0')\n"
        "    lf = open(log_f, 'a', buffering=1)\n"
        "    p = subprocess.Popen([sys.executable, '-u', str(payload)], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)\n"
        "    pid_f.write_text(str(p.pid))\n"
        "    print(f'[worker] Started background GPU process PID={p.pid} for " + job_name + "', flush=True)\n"
        "offset = int(off_f.read_text().strip()) if off_f.exists() else 0\n"
        f"for _ in range({poll_window}):\n"
        "    if log_f.exists():\n"
        "        with open(log_f, 'r', errors='replace') as f:\n"
        "            f.seek(offset)\n"
        "            chunk = f.read()\n"
        "            if chunk:\n"
        "                sys.stdout.write(chunk)\n"
        "                sys.stdout.flush()\n"
        "                offset = f.tell()\n"
        "                off_f.write_text(str(offset))\n"
        "    if rc_f.exists() and not is_running():\n"
        "        if log_f.exists():\n"
        "            with open(log_f, 'r', errors='replace') as f:\n"
        "                f.seek(offset)\n"
        "                chunk = f.read()\n"
        "                if chunk:\n"
        "                    sys.stdout.write(chunk)\n"
        "                    sys.stdout.flush()\n"
        "                    offset = f.tell()\n"
        "                    off_f.write_text(str(offset))\n"
        "        rc = rc_f.read_text().strip()\n"
        "        print(f'__JOB_DONE_RC_{rc}__', flush=True)\n"
        "        break\n"
        "    time.sleep(1)\n"
    )

    # Clear any stale rc file before starting a new job
    clear_local = Path(tempfile.gettempdir()) / f'clear_{job_name}.py'
    clear_local.write_text(
        "import pathlib\n"
        f"for p in ({rc_remote!r}, {offset_remote!r}):\n"
        "    pathlib.Path(p).unlink(missing_ok=True)\n"
    )
    run_cmd([COLAB_BIN, 'exec', '-s', session, '--timeout', '30', '-f', str(clear_local)], retries=5, idle_timeout=35.0)

    consec_not_found = 0
    while True:
        rc, out = run_cmd(
            [COLAB_BIN, 'exec', '-s', session, '--timeout', str(poll_window + 25), '-f', str(poller_local)],
            check=False,
            retries=1,
            tee_live=True,
            idle_timeout=float(poll_window + 20),
        )
        if '__JOB_DONE_RC_0__' in out:
            log_print(f'[REMOTE-OK] Job {job_name} finished with exit code 0.')
            return 0
        if '__JOB_DONE_RC_' in out:
            log_print(f'[REMOTE-ERR] Job {job_name} finished with non-zero exit code.')
            return 1
        if 'Session not found' in out:
            consec_not_found += 1
            if consec_not_found >= 2 and check_session:
                _BOOTSTRAPPED_SESSIONS.discard(session)
                ensure_session(session)
                run_cmd([COLAB_BIN, 'upload', '-s', session, str(payload_local), payload_remote], retries=5, idle_timeout=60.0)
                consec_not_found = 0
        else:
            consec_not_found = 0
        if rc != 0:
            time.sleep(3)


def cmd_status(_args):
    run_cmd([COLAB_BIN, 'usage'], check=False)
    run_cmd([COLAB_BIN, 'sessions'], check=False)
    run_cmd(['rclone', 'lsd', '--config', RCLONE_CONF, GDRIVE_REMOTE], check=False)
    run_cmd(['rclone', 'lsd', '--config', RCLONE_CONF, f'{GDRIVE_REMOTE}/runs'], check=False)
    run_cmd(['rclone', 'lsd', '--config', RCLONE_CONF, f'{GDRIVE_REMOTE}/artifacts/encoders'], check=False)
    runs_root = ROOT / 'runs'
    if runs_root.exists():
        done = sorted(p.parent.relative_to(runs_root) for p in runs_root.rglob('metrics.json'))
        print(f'\nCompleted local runs ({len(done)}):')
        for d in done:
            print(f'  - {d}')


def cmd_sync_gdrive(_args):
    bundle = ROOT / 'data' / 'colab_data_bundle.tar.gz'
    if not bundle.exists():
        raise SystemExit(f'{bundle} not found; run scripts/prepare_data_bundle.py first')
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, str(bundle), f'{GDRIVE_REMOTE}/data/'])
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, str(ROOT / 'data' / 'colab_data_bundle.sha256'), f'{GDRIVE_REMOTE}/data/'])
    if (ROOT / 'artifacts' / 'encoders').exists():
        run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, str(ROOT / 'artifacts'), f'{GDRIVE_REMOTE}/artifacts/'])
    log_print('Synced dataset bundle and E1 SSL encoders to Google Drive.')


def cmd_bootstrap(args):
    s = args.session
    if args.new:
        run_cmd([COLAB_BIN, 'new', '-s', s, '--gpu', args.gpu], retries=3, idle_timeout=90.0)
    run_cmd([COLAB_BIN, 'install', '-s', s, 'timm', 'albumentations', 'opencv-python-headless', 'scikit-learn', 'scipy'], retries=4, idle_timeout=90.0)

    code_tar = Path(tempfile.gettempdir()) / 'ramscore_code.tar.gz'
    subprocess.run(
        ['tar', '-czf', str(code_tar), '--exclude=.venv', '--exclude=__pycache__', '-C', str(ROOT), 'code/src', 'code/pyproject.toml'],
        check=True,
    )
    run_cmd([COLAB_BIN, 'upload', '-s', s, str(code_tar), '/content/ramscore_code.tar.gz'], retries=4, idle_timeout=60.0)
    time.sleep(2)
    run_cmd([COLAB_BIN, 'upload', '-s', s, RCLONE_CONF, '/content/colab.conf'], retries=4, idle_timeout=60.0)
    time.sleep(2)

    hf_tok = get_hf_token()
    setup_code = (
        "import os, shutil, subprocess, pathlib, torch\n"
        + (f"os.environ['HF_TOKEN'] = {hf_tok!r}\n" if hf_tok else "")
        + "print('[setup] Checking rclone...', flush=True)\n"
        "if shutil.which('rclone') is None:\n"
        "    subprocess.run(['apt-get', 'update', '-qq'], check=True)\n"
        "    subprocess.run(['apt-get', 'install', '-y', '-qq', 'rclone'], check=True)\n"
        "print('[setup] Extracting ramscore_code.tar.gz...', flush=True)\n"
        "pathlib.Path('/content/data').mkdir(exist_ok=True)\n"
        "subprocess.run(['tar', '-xzf', '/content/ramscore_code.tar.gz', '-C', '/content'], check=True)\n"
        "if not pathlib.Path('/content/colab_data_bundle.tar.gz').exists():\n"
        "    print('[setup] Pulling colab_data_bundle.tar.gz from Google Drive...', flush=True)\n"
        "    subprocess.run([\n"
        "        'rclone', 'copy', '--config', '/content/colab.conf',\n"
        f"        '{GDRIVE_REMOTE}/data/colab_data_bundle.tar.gz', '/content/'\n"
        "    ], check=True)\n"
        "be_json = pathlib.Path('/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring/train/_annotation_be_scores.json')\n"
        "if not be_json.exists():\n"
        "    print('[setup] Extracting colab_data_bundle.tar.gz to /content/data...', flush=True)\n"
        "    subprocess.run(['tar', '-xzf', '/content/colab_data_bundle.tar.gz', '-C', '/content/data'], check=True)\n"
        "if not pathlib.Path('/content/artifacts/encoders/e1_dino_resnet34/encoder.pt').exists():\n"
        "    print('[setup] Pulling E1 SSL encoders from Google Drive...', flush=True)\n"
        "    pathlib.Path('/content/artifacts/encoders').mkdir(parents=True, exist_ok=True)\n"
        "    subprocess.run([\n"
        "        'rclone', 'copy', '--config', '/content/colab.conf',\n"
        f"        '{GDRIVE_REMOTE}/artifacts/encoders', '/content/artifacts/encoders/'\n"
        "    ], check=True)\n"
        "assert be_json.exists(), f'Missing {be_json} after extraction!'\n"
        "print('[setup] Ready! CUDA available:', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', flush=True)\n"
    )
    run_remote_background(s, 'setup_env', setup_code, poll_window=20, check_session=False)


_BOOTSTRAPPED_SESSIONS: set[str] = set()


def ensure_session(session: str, gpu: str = 'L4'):
    sess_res = subprocess.run([COLAB_BIN, 'sessions'], capture_output=True, text=True)
    combined_sess = (sess_res.stdout + sess_res.stderr).lower()
    if 'no active sessions' in combined_sess or session.lower() not in combined_sess:
        log_print(f'[BOOT] Session {session} is not active; provisioning and bootstrapping fresh {gpu} VM ...')
        _BOOTSTRAPPED_SESSIONS.add(session)
        cmd_bootstrap(argparse.Namespace(session=session, gpu=gpu, new=True))
    elif session not in _BOOTSTRAPPED_SESSIONS:
        log_print(f'[BOOT] Session {session} is active on Colab; ensuring code & data bundle are bootstrapped ...')
        _BOOTSTRAPPED_SESSIONS.add(session)
        cmd_bootstrap(argparse.Namespace(session=session, gpu=gpu, new=False))


def run_batch_jobs(session: str, stage: str, jobs: list[tuple[str, list[str], bool]]):
    """Execute a list of (run_name, cli_args, keep_ckpt) jobs inside a detached remote Python worker,
    automatically evaluating zero-shot RheumaVIT transfer before syncing to Google Drive."""
    local_stage_dir = ROOT / 'runs' / stage
    local_stage_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (run_name, cli_args, keep_ckpt)
        for run_name, cli_args, keep_ckpt in jobs
        if not (local_stage_dir / run_name / 'metrics.json').exists()
    ]
    if not pending:
        log_print(f'All {len(jobs)} jobs in stage {stage} are already completed!')
        return

    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, '--exclude', 'model_best.pt', f'{GDRIVE_REMOTE}/runs/{stage}', str(local_stage_dir)], check=False)

    hf_tok = get_hf_token()
    remote_out = f'/content/runs/{stage}'

    for attempt in range(1, 6):
        still_pending = [
            (run_name, cli_args, keep_ckpt)
            for run_name, cli_args, keep_ckpt in pending
            if not (local_stage_dir / run_name / 'metrics.json').exists()
        ]
        if not still_pending:
            log_print(f'[SUCCESS] All {len(pending)} runs in stage {stage} completed and synced to {local_stage_dir}!')
            return

        code = (
            "import os, pathlib, subprocess, sys, time, torch\n"
            + (f"os.environ['HF_TOKEN'] = {hf_tok!r}\n" if hf_tok else "")
            + "sys.path.insert(0, '/content/code/src')\n"
            "from ramscore.train import main\n"
            "from ramscore.rq3 import eval_direct\n"
            "be_json = pathlib.Path('/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring/train/_annotation_be_scores.json')\n"
            "if not be_json.exists():\n"
            "    print('[batch] Extracting /content/colab_data_bundle.tar.gz to /content/data ...', flush=True)\n"
            "    subprocess.run(['tar', '-xzf', '/content/colab_data_bundle.tar.gz', '-C', '/content/data'], check=True)\n"
            "rv_manifest = pathlib.Path('/content/data/derived/rheumavit/manifest.csv')\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            f"jobs = {still_pending!r}\n"
            "for idx, (run_name, cli_args, keep_ckpt) in enumerate(jobs, 1):\n"
            f"    out_dir = pathlib.Path('{remote_out}') / run_name\n"
            "    if (out_dir / 'metrics.json').exists():\n"
            "        print(f'[{idx}/{len(jobs)}] [SKIP-REMOTE] {run_name} already finished on VM.', flush=True)\n"
            "        continue\n"
            "    if '--init-from' in cli_args:\n"
            "        init_p = pathlib.Path(cli_args[cli_args.index('--init-from') + 1])\n"
            "        if not init_p.exists():\n"
            "            init_p.parent.mkdir(parents=True, exist_ok=True)\n"
            "            if str(init_p).startswith('/content/artifacts/'):\n"
            "                rel = init_p.relative_to('/content/artifacts')\n"
            "                subprocess.run(['rclone', 'copyto', '--config', '/content/colab.conf', "
            f"f'{GDRIVE_REMOTE}/artifacts/{{rel}}', str(init_p)], check=True)\n"
            "            else:\n"
            "                rel = init_p.relative_to('/content/runs')\n"
            "                subprocess.run(['rclone', 'copyto', '--config', '/content/colab.conf', "
            f"f'{GDRIVE_REMOTE}/runs/{{rel}}', str(init_p)], check=True)\n"
            "    if '--label-overrides' in cli_args:\n"
            "        ov_p = pathlib.Path(cli_args[cli_args.index('--label-overrides') + 1])\n"
            "        if not ov_p.exists():\n"
            "            ov_p.parent.mkdir(parents=True, exist_ok=True)\n"
            "            subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', "
            f"'{GDRIVE_REMOTE}/runs/rq5/oof', str(ov_p.parent)], check=True)\n"
            "    t0 = time.time()\n"
            "    print(f'\\n=== [{idx}/{len(jobs)}] STARTING {run_name} ===', flush=True)\n"
            f"    main(['--name', run_name, '--out', '{remote_out}'] + cli_args)\n"
            "    if rv_manifest.exists() and (out_dir / 'model_best.pt').exists():\n"
            "        try:\n"
            "            eval_direct(out_dir, rv_manifest, out_dir / 'rheumavit', device)\n"
            "        except Exception as e:\n"
            "            print(f'[WARN] RheumaVIT direct eval failed for {run_name}: {e}', flush=True)\n"
            "    dt = (time.time() - t0) / 60.0\n"
            "    print(f'=== [{idx}/{len(jobs)}] FINISHED {run_name} in {dt:.1f} min; syncing to Google Drive ===', flush=True)\n"
            "    rclone_cmd = ['rclone', 'copy', '--config', '/content/colab.conf']\n"
            "    if not keep_ckpt:\n"
            "        rclone_cmd += ['--exclude', 'model_best.pt']\n"
            f"    rclone_cmd += [str(out_dir), f'{GDRIVE_REMOTE}/runs/{stage}/{{run_name}}']\n"
            "    subprocess.run(rclone_cmd, check=True)\n"
            "    if not keep_ckpt and (out_dir / 'model_best.pt').exists():\n"
            "        (out_dir / 'model_best.pt').unlink(missing_ok=True)\n"
        )
        run_remote_background(session, f'batch_{stage}', code, poll_window=25)
        run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, '--exclude', 'model_best.pt', f'{GDRIVE_REMOTE}/runs/{stage}', str(local_stage_dir)], check=False)
        still_missing = [name for name, _, _ in pending if not (local_stage_dir / name / 'metrics.json').exists()]
        if not still_missing:
            log_print(f'[SUCCESS] All {len(pending)} runs in stage {stage} completed and synced to {local_stage_dir}!')
            return
        log_print(f'[RESUME] {len(still_missing)} runs remaining in stage {stage} (attempt {attempt}/5); retrying in 5s ...')
        time.sleep(5)


def select_best_arm(sub_dir: str, candidates: list[str]) -> str:
    """Pick the arm with the highest mean validation BACC4 across f1.0 and f0.1 (tie-broken by validation QWK)."""
    best_arm = candidates[0]
    best_key = (-1.0, -1.0)
    for arm in candidates:
        base = ROOT / 'runs' / ('rq0' if arm.startswith('rq0') else sub_dir)
        ms = [json.loads(p.read_text()) for p in sorted(base.glob(f'{arm}_*/metrics.json')) if p.exists()]
        if not ms:
            continue
        bacc = statistics.mean([m['val']['bacc4'] for m in ms])
        qwk = statistics.mean([m['val']['qwk'] for m in ms])
        if (round(bacc, 4), round(qwk, 4)) > best_key:
            best_key = (round(bacc, 4), round(qwk, 4))
            best_arm = arm
    log_print(f'[SELECT] Selected best arm from {candidates}: {best_arm} (val_bacc4={best_key[0]:.4f}, val_qwk={best_key[1]:.4f})')
    return best_arm


def run_stage_rq5_mt(session: str):
    """Run all remaining Mean-Teacher semi-supervised jobs (RQ5-11) inside a detached remote worker."""
    local_rq5 = ROOT / 'runs' / 'rq5'
    local_rq5.mkdir(parents=True, exist_ok=True)
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, '--exclude', 'model_best.pt', f'{GDRIVE_REMOTE}/runs/rq5', str(local_rq5)], check=False)

    targets = [(frac, draw) for frac in ('0.1', '0.25') for draw in (0, 1, 2)]
    pending = [
        (frac, draw, f'rq5-11_resnet34_meanteacher_b2_f{frac}_d{draw}_s42')
        for frac, draw in targets
        if not (local_rq5 / f'rq5-11_resnet34_meanteacher_b2_f{frac}_d{draw}_s42' / 'metrics.json').exists()
    ]
    if not pending:
        log_print('All 6 rq5-11 Mean-Teacher runs are already completed locally!')
        return

    hf_tok = get_hf_token()
    code = (
        "import os, pathlib, subprocess, sys, time, torch\n"
        + (f"os.environ['HF_TOKEN'] = {hf_tok!r}\n" if hf_tok else "")
        + "sys.path.insert(0, '/content/code/src')\n"
        "from ramscore.rq5 import train_mean_teacher\n"
        "from ramscore.rq3 import eval_direct\n"
        "rv_manifest = pathlib.Path('/content/data/derived/rheumavit/manifest.csv')\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        f"pending = {pending!r}\n"
        "for idx, (frac, draw, name) in enumerate(pending, 1):\n"
        "    out_dir = pathlib.Path('/content/runs/rq5') / name\n"
        "    if (out_dir / 'metrics.json').exists():\n"
        "        print(f'[{idx}/{len(pending)}] [SKIP-REMOTE] {name} already done on VM.', flush=True)\n"
        "        continue\n"
        "    t0 = time.time()\n"
        "    print(f'\\n=== [{idx}/{len(pending)}] STARTING {name} ===', flush=True)\n"
        "    train_mean_teacher(\n"
        "        name=name, out='/content/runs/rq5',\n"
        "        data_root='/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',\n"
        "        label_fraction=float(frac), fraction_seed=int(draw), seed=42,\n"
        "    )\n"
        "    if rv_manifest.exists() and (out_dir / 'model_best.pt').exists():\n"
        "        try:\n"
        "            eval_direct(out_dir, rv_manifest, out_dir / 'rheumavit', device)\n"
        "        except Exception as e:\n"
        "            print(f'[WARN] RheumaVIT direct eval failed for {name}: {e}', flush=True)\n"
        "    dt = (time.time() - t0) / 60.0\n"
        "    print(f'=== [{idx}/{len(pending)}] FINISHED {name} in {dt:.1f} min; syncing to Google Drive ===', flush=True)\n"
        "    subprocess.run([\n"
        "        'rclone', 'copy', '--config', '/content/colab.conf',\n"
        f"        str(out_dir), f'{GDRIVE_REMOTE}/runs/rq5/{{name}}'\n"
        "    ], check=True)\n"
    )
    run_remote_background(session, 'rq5_mt', code, poll_window=25)
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, '--exclude', 'model_best.pt', f'{GDRIVE_REMOTE}/runs/rq5', str(local_rq5)], check=False)
    log_print('[SUCCESS] All 6 Mean-Teacher runs (rq5-11) completed and synced!')


def run_stage_rq3_eval(session: str):
    """Execute RQ3 external validation on RheumaVIT (direct transfer, AdaBN, HistMatch, Few-Shot 5 & 20 patients)
    and RQ2-P post-hoc k-NN & 4-view TTA on saved checkpoints."""
    local_rq3_eval = ROOT / 'runs' / 'rq3_eval'
    local_rq2_ph = ROOT / 'runs' / 'rq2_posthoc'
    if (local_rq3_eval / 'done.json').exists() and (local_rq2_ph / 'done.json').exists():
        log_print('[SKIP] rq3-eval & rq2-posthoc already completed locally.')
        return

    hf_tok = get_hf_token()
    code = (
        "import json, os, pathlib, subprocess, sys, time, torch\n"
        + (f"os.environ['HF_TOKEN'] = {hf_tok!r}\n" if hf_tok else "")
        + "sys.path.insert(0, '/content/code/src')\n"
        "from ramscore.rq2 import run_posthoc_knn_and_tta\n"
        "from ramscore.rq3 import eval_direct, eval_adabn, eval_histmatch, eval_fewshot\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "rv_manifest = pathlib.Path('/content/data/derived/rheumavit/manifest.csv')\n"
        "ram_root = '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring'\n"
        "designs = [('0.1', 0, 42), ('0.1', 1, 42), ('0.1', 2, 42), ('1.0', 0, 42), ('1.0', 0, 1337), ('1.0', 0, 2026)]\n"
        "ckpt_groups = [\n"
        "    ('rq0', 'rq0-01_resnet34_b2'),\n"
        "    ('rq2', 'rq2-01_resnet34_ce_b2'),\n"
        "    ('rq3', 'rq3-04_resnet34_aug_strong_b2'),\n"
        "    ('rq3', 'rq3-05_resnet34_aug_rvmatch_b2'),\n"
        "    ('rq4', 'rq4-01_resnet34_mcdrop_b2'),\n"
        "]\n"
        "def ensure_ckpt(stage, run_name):\n"
        "    rdir = pathlib.Path('/content/runs') / stage / run_name\n"
        "    if not (rdir / 'model_best.pt').exists() or not (rdir / 'test_predictions.csv').exists():\n"
        "        rdir.mkdir(parents=True, exist_ok=True)\n"
        "        subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', "
        f"f'{GDRIVE_REMOTE}/runs/{{stage}}/{{run_name}}', str(rdir)], check=True)\n"
        "    return rdir\n"
        "\n"
        "# 1. RQ2-P k-NN & 4-view TTA on RQ0-01 (6 runs)\n"
        "ph_root = pathlib.Path('/content/runs/rq2_posthoc')\n"
        "for frac, draw, seed in designs:\n"
        "    rname = f'rq0-01_resnet34_b2_f{frac}_d{draw}_s{seed}'\n"
        "    knn_dir = ph_root / f'rq2-pb_knn_b2_f{frac}_d{draw}_s{seed}'\n"
        "    tta_dir = ph_root / f'rq2-pc_tta_b2_f{frac}_d{draw}_s{seed}'\n"
        "    if not (knn_dir / 'metrics.json').exists() or not (tta_dir / 'metrics.json').exists():\n"
        "        print(f'[RQ2-P] Running k-NN & TTA for {rname} ...', flush=True)\n"
        "        rdir = ensure_ckpt('rq0', rname)\n"
        "        run_posthoc_knn_and_tta(rdir, knn_dir, tta_dir, data_root=ram_root)\n"
        "(ph_root / 'done.json').write_text(json.dumps({'status': 'ok'}))\n"
        f"subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', str(ph_root), '{GDRIVE_REMOTE}/runs/rq2_posthoc'], check=True)\n"
        "\n"
        "# 2. RQ3-01 Direct Transfer on RheumaVIT for all saved checkpoint arms\n"
        "ev_root = pathlib.Path('/content/runs/rq3_eval')\n"
        "for stage, pref in ckpt_groups:\n"
        "    for frac, draw, seed in designs:\n"
        "        rname = f'{pref}_f{frac}_d{draw}_s{seed}'\n"
        "        dest = ev_root / 'direct' / rname\n"
        "        if not (dest / 'metrics.json').exists():\n"
        "            print(f'[RQ3-01 Direct] Evaluating {rname} on RheumaVIT ...', flush=True)\n"
        "            rdir = ensure_ckpt(stage, rname)\n"
        "            eval_direct(rdir, rv_manifest, dest, device)\n"
        "        if pref == 'rq0-01_resnet34_b2':\n"
        "            adabn_dest = ev_root / 'adabn' / f'rq3-02_adabn_b2_f{frac}_d{draw}_s{seed}'\n"
        "            if not (adabn_dest / 'metrics.json').exists():\n"
        "                print(f'[RQ3-02 AdaBN] Evaluating {rname} ...', flush=True)\n"
        "                rdir = ensure_ckpt(stage, rname)\n"
        "                eval_adabn(rdir, rv_manifest, adabn_dest, device)\n"
        "            hm_dest = ev_root / 'histmatch' / f'rq3-03_histmatch_b2_f{frac}_d{draw}_s{seed}'\n"
        "            if not (hm_dest / 'metrics.json').exists():\n"
        "                print(f'[RQ3-03 HistMatch] Evaluating {rname} ...', flush=True)\n"
        "                rdir = ensure_ckpt(stage, rname)\n"
        "                eval_histmatch(rdir, rv_manifest, hm_dest, device, ram_root=ram_root)\n"
        "for frac in ('0.1', '0.25'):\n"
        "    for draw in (0, 1, 2):\n"
        "        rname = f'rq5-11_resnet34_meanteacher_b2_f{frac}_d{draw}_s42'\n"
        "        dest = ev_root / 'direct' / rname\n"
        "        if not (dest / 'metrics.json').exists():\n"
        "            print(f'[RQ3-01 Direct] Evaluating {rname} on RheumaVIT ...', flush=True)\n"
        "            rdir = ensure_ckpt('rq5', rname)\n"
        "            eval_direct(rdir, rv_manifest, dest, device)\n"
        "\n"
        "# 3. RQ3-06 Few-Shot Adaptation (5 & 20 patients x 3 draws) on anchor RQ0-01\n"
        "anchor_dir = ensure_ckpt('rq0', 'rq0-01_resnet34_b2_f1.0_d0_s42')\n"
        "for kp in (5, 20):\n"
        "    for d in (0, 1, 2):\n"
        "        fs_dest = ev_root / 'fewshot' / f'rq3-06_fewshot_k{kp}_d{d}_s42'\n"
        "        if not (fs_dest / 'metrics.json').exists():\n"
        "            print(f'[RQ3-06 Few-Shot] k={kp} patients, draw={d} ...', flush=True)\n"
        "            eval_fewshot(anchor_dir, rv_manifest, k_patients=kp, draw=d, out_dir=fs_dest, device=device)\n"
        "\n"
        "(ev_root / 'done.json').write_text(json.dumps({'status': 'ok'}))\n"
        f"subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', str(ev_root), '{GDRIVE_REMOTE}/runs/rq3_eval'], check=True)\n"
    )
    run_remote_background(session, 'rq3_eval', code, poll_window=25)
    local_rq2_ph.mkdir(parents=True, exist_ok=True)
    local_rq3_eval.mkdir(parents=True, exist_ok=True)
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, f'{GDRIVE_REMOTE}/runs/rq2_posthoc', str(local_rq2_ph)], check=False)
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, f'{GDRIVE_REMOTE}/runs/rq3_eval', str(local_rq3_eval)], check=False)
    log_print('[SUCCESS] Synced rq2_posthoc and rq3_eval to local runs/!')


def run_stage_rq1_normality(session: str):
    """Run RQ1-14 (SS-FewSOME normality model on Grade 0) and RQ1-15 (Normality Gate 0 vs >=1)."""
    local_rq1 = ROOT / 'runs' / 'rq1'
    local_rq1.mkdir(parents=True, exist_ok=True)
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, '--exclude', 'model_best.pt', f'{GDRIVE_REMOTE}/runs/rq1', str(local_rq1)], check=False)

    pending = []
    for frac, draw, seed in DESIGNS_6:
        n14 = f'rq1-14_resnet34_normality_b2_f{frac}_d{draw}_s{seed}'
        n15 = f'rq1-15_resnet34_normgate_b2_f{frac}_d{draw}_s{seed}'
        if not (local_rq1 / n14 / 'metrics.json').exists() or not (local_rq1 / n15 / 'metrics.json').exists():
            pending.append((frac, draw, seed, n14, n15))
    if not pending:
        log_print('All RQ1-14/15 normality & gate runs are already completed locally!')
        return

    hf_tok = get_hf_token()
    code = (
        "import os, pathlib, subprocess, sys, time\n"
        + (f"os.environ['HF_TOKEN'] = {hf_tok!r}\n" if hf_tok else "")
        + "sys.path.insert(0, '/content/code/src')\n"
        "from ramscore.normality import main as norm_main\n"
        "ram_root = '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring'\n"
        f"pending = {pending!r}\n"
        "for idx, (frac, draw, seed, n14, n15) in enumerate(pending, 1):\n"
        "    d14 = pathlib.Path('/content/runs/rq1') / n14\n"
        "    d15 = pathlib.Path('/content/runs/rq1') / n15\n"
        "    ref_dir = pathlib.Path(f'/content/runs/rq0/rq0-01_resnet34_b2_f{frac}_d{draw}_s{seed}')\n"
        "    if not (ref_dir / 'val_logits.npz').exists():\n"
        "        ref_dir.mkdir(parents=True, exist_ok=True)\n"
        "        subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', "
        f"f'{GDRIVE_REMOTE}/runs/rq0/{{ref_dir.name}}', str(ref_dir)], check=True)\n"
        "    if not (d14 / 'metrics.json').exists():\n"
        "        print(f'\\n=== [{idx}/{len(pending)}] STARTING {n14} ===', flush=True)\n"
        "        norm_main(['train', '--name', n14, '--out', '/content/runs/rq1', '--data-root', ram_root, "
        "'--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed), '--amp', '--workers', '2'])\n"
        "        subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', '--exclude', 'normality.pt', "
        f"str(d14), f'{GDRIVE_REMOTE}/runs/rq1/{{n14}}'], check=True)\n"
        "    if not (d15 / 'metrics.json').exists():\n"
        "        print(f'=== [{idx}/{len(pending)}] STARTING {n15} ===', flush=True)\n"
        "        norm_main(['gate', '--name', n15, '--out', '/content/runs/rq1', '--data-root', ram_root, "
        "'--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed), "
        "'--normality', str(d14), '--classifier', str(ref_dir), '--workers', '2'])\n"
        "        subprocess.run(['rclone', 'copy', '--config', '/content/colab.conf', "
        f"str(d15), f'{GDRIVE_REMOTE}/runs/rq1/{{n15}}'], check=True)\n"
    )
    run_remote_background(session, 'rq1_normality', code, poll_window=25)
    run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, '--exclude', 'model_best.pt', f'{GDRIVE_REMOTE}/runs/rq1', str(local_rq1)], check=False)


def run_stage_rq1(session: str, selected_arms: str = ''):
    """Execute RQ1 arms: Scratch, DINO, ChronoCon, ViT-S, MAE, SupCon, SupCon-Ord, Normality/Gate, and LP/LP-FT."""
    arms = selected_arms.split(',') if selected_arms else list(RQ1_MAIN_ARMS.keys())
    main_jobs = []
    for frac, draw, seed in DESIGNS_6:
        for arm in arms:
            if arm not in RQ1_MAIN_ARMS:
                continue
            tag, flags = RQ1_MAIN_ARMS[arm]
            name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
            cli = COMMON_FLAGS + flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
            main_jobs.append((name, cli, False))
    if main_jobs:
        run_batch_jobs(session, 'rq1', main_jobs)

    if not selected_arms or 'rq1-14' in arms or 'rq1-15' in arms:
        run_stage_rq1_normality(session)

    if not selected_arms or any(a.startswith('rq1-17') for a in arms):
        best_ssl = select_best_arm('rq1', ['rq1-02', 'rq1-03', 'rq1-05', 'rq1-06'])
        ssl_tag, ssl_flags = RQ1_ENCODER_CONFIGS[best_ssl]
        ssl_init = [ssl_flags[ssl_flags.index('--init-from') + 1]] if '--init-from' in ssl_flags else []
        ssl_model = ssl_flags[ssl_flags.index('--model') + 1]

        lp_flags_base = [
            '--amp', '--n-boot', '0', '--epochs', '20', '--min-steps', '800',
            '--batch-size', '32', '--lr', '1e-3', '--weight-decay', '1e-3',
            '--aug', 'base', '--seed-aug', '--monitor', 'qwk',
            '--data-root', '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
            '--freeze-backbone', '--head', 'softmax', '--logit-adjust', '1',
        ]
        lpft_flags_base = [
            '--amp', '--n-boot', '0', '--epochs', '40', '--min-steps', '1600',
            '--batch-size', '32', '--lr', '1e-4', '--weight-decay', '1e-3',
            '--aug', 'base', '--seed-aug', '--monitor', 'qwk',
            '--data-root', '/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',
            '--keep-head', '--head', 'softmax', '--logit-adjust', '1',
        ]

        lp_jobs = []
        lpft_jobs = []
        for frac, draw, seed in DESIGNS_6:
            n17a = f'rq1-17a_resnet34_lp_b2_f{frac}_d{draw}_s{seed}'
            c17a = lp_flags_base + ['--model', 'resnet34', '--pretrained', '--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
            lp_jobs.append((n17a, c17a, True))

            n17b = f'rq1-17b_resnet34_lpft_b2_f{frac}_d{draw}_s{seed}'
            c17b = lpft_flags_base + ['--model', 'resnet34', '--init-from', f'/content/runs/rq1/{n17a}/model_best.pt', '--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
            lpft_jobs.append((n17b, c17b, False))

            n17c = f'rq1-17c_{ssl_tag}_lp_b2_f{frac}_d{draw}_s{seed}'
            c17c = lp_flags_base + ['--model', ssl_model, '--init-from', ssl_init[0], '--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
            lp_jobs.append((n17c, c17c, True))

            n17d = f'rq1-17d_{ssl_tag}_lpft_b2_f{frac}_d{draw}_s{seed}'
            extra_llrd = ['--layer-decay', '0.75', '--warmup', '0.1'] if 'vit' in ssl_model else []
            c17d = lpft_flags_base + ['--model', ssl_model, '--init-from', f'/content/runs/rq1/{n17c}/model_best.pt'] + extra_llrd + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
            lpft_jobs.append((n17d, c17d, False))

        run_batch_jobs(session, 'rq1', lp_jobs)
        run_batch_jobs(session, 'rq1', lpft_jobs)


def run_stage_rq2_inter(session: str):
    """Run RQ1 x RQ2 interaction: top 2 heads/losses from RQ2 (rq2-15 cbloss & rq2-16 b2_cornreg) on best RQ1 encoder."""
    best_enc_arm = select_best_arm('rq1', ['rq1-02', 'rq1-03', 'rq1-04', 'rq1-05', 'rq1-06', 'rq1-12', 'rq1-13'])
    enc_tag, enc_flags = RQ1_ENCODER_CONFIGS[best_enc_arm]

    top2_rq2_heads = [
        ('rq2-18a', f'{enc_tag}_cbloss', ['--head', 'softmax', '--effective-beta', '0.9999']),
        ('rq2-18b', f'{enc_tag}_b2_cornreg', ['--head', 'softmax_corn', '--logit-adjust', '1', '--corn-reg-weight', '0.5']),
    ]
    jobs = []
    for frac, draw, seed in DESIGNS_6:
        for code, tag, hflags in top2_rq2_heads:
            name = f'{code}_{tag}_b2_f{frac}_d{draw}_s{seed}'
            cli = COMMON_FLAGS + enc_flags + hflags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
            jobs.append((name, cli, False))
    run_batch_jobs(session, 'rq2', jobs)


def cmd_run_queue(args):
    s = args.session
    stage = args.stage
    try:
        if stage == 'rq0':
            jobs = []
            for frac, draw, seed in DESIGNS_6:
                for arm, (tag, flags) in RQ0_ARMS.items():
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = COMMON_FLAGS + flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    jobs.append((name, cli, arm == 'rq0-01'))
            run_batch_jobs(s, 'rq0', jobs)
        elif stage == 'rq1':
            run_stage_rq1(s, args.arms)
        elif stage == 'rq2':
            selected = args.arms.split(',') if args.arms else list(RQ2_ARMS.keys())
            jobs = []
            for frac, draw, seed in DESIGNS_6:
                for arm in selected:
                    if arm == 'rq2-07':
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
                        jobs.append((name, cli, False))
                        continue
                    tag, flags = RQ2_ARMS[arm]
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = COMMON_FLAGS + flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    jobs.append((name, cli, arm == 'rq2-01'))
            run_batch_jobs(s, 'rq2', jobs)
        elif stage == 'rq2-inter':
            run_stage_rq2_inter(s)
        elif stage == 'rq5-oof':
            local_oof = ROOT / 'runs' / 'rq5' / 'oof'
            if (local_oof / 'summary.json').exists():
                log_print('[SKIP] rq5/oof already exists locally.')
            else:
                hf_tok = get_hf_token()
                code = (
                    "import os, subprocess, sys\n"
                    + (f"os.environ['HF_TOKEN'] = {hf_tok!r}\n" if hf_tok else "")
                    + "sys.path.insert(0, '/content/code/src')\n"
                    "from ramscore.rq5 import run_oof_and_overrides\n"
                    "res = run_oof_and_overrides(\n"
                    "    data_root='/content/data/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring',\n"
                    "    out_dir='/content/runs/rq5/oof',\n"
                    ")\n"
                    "print('OOF summary:', res, flush=True)\n"
                    "subprocess.run([\n"
                    "    'rclone', 'copy', '--config', '/content/colab.conf',\n"
                    f"    '/content/runs/rq5/oof', '{GDRIVE_REMOTE}/runs/rq5/oof'\n"
                    "], check=True)\n"
                )
                run_remote_background(s, 'rq5_oof', code, poll_window=25)
                local_oof.mkdir(parents=True, exist_ok=True)
                run_cmd(['rclone', 'copy', '--config', RCLONE_CONF, f'{GDRIVE_REMOTE}/runs/rq5/oof', str(local_oof)], check=False)
        elif stage == 'rq5-mt':
            run_stage_rq5_mt(s)
        elif stage == 'rq3-train':
            jobs = []
            for frac, draw, seed in DESIGNS_6:
                for arm, (tag, flags) in RQ3_TRAIN_ARMS.items():
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    jobs.append((name, cli, True))
            run_batch_jobs(s, 'rq3', jobs)
        elif stage == 'rq3-eval':
            run_stage_rq3_eval(s)
        elif stage == 'rq4-train':
            jobs = []
            for frac, draw, seed in DESIGNS_6:
                for arm, (tag, flags) in RQ4_TRAIN_ARMS.items():
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    jobs.append((name, cli, True))
            run_batch_jobs(s, 'rq4', jobs)
        elif stage == 'rq5-train':
            selected = args.arms.split(',') if args.arms else list(RQ5_TRAIN_ARMS.keys())
            jobs = []
            for frac, draw, seed in DESIGNS_6:
                for arm in selected:
                    tag, flags = RQ5_TRAIN_ARMS[arm]
                    name = f'{arm}_{tag}_b2_f{frac}_d{draw}_s{seed}'
                    cli = flags + ['--label-fraction', frac, '--fraction-seed', str(draw), '--seed', str(seed)]
                    jobs.append((name, cli, False))
            run_batch_jobs(s, 'rq5', jobs)
        elif stage in ('all-remaining', 'step2-and-3'):
            for sub_stage, sub_arms in [
                ('rq5-mt', ''),
                ('rq3-eval', ''),
                ('rq1', ''),
                ('rq2-inter', ''),
            ]:
                log_print(f'=== RUNNING SUB-STAGE: {sub_stage} ({sub_arms or "all"}) ===')
                sub_ns = argparse.Namespace(session=s, stage=sub_stage, arms=sub_arms, auto_stop=False)
                cmd_run_queue(sub_ns)
        else:
            raise SystemExit(f'Unknown stage {stage}')
    finally:
        if args.auto_stop:
            log_print(f'[AUTO-STOP] Releasing Colab session {s} ...')
            run_cmd([COLAB_BIN, 'stop', '-s', s], check=False)


def cmd_stop(args):
    run_cmd([COLAB_BIN, 'stop', '-s', args.session])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    sub.add_parser('status', help='Show Colab compute balance, active sessions, and completed runs')
    sub.add_parser('sync-gdrive', help='Upload dataset bundle and E1 encoders to Google Drive via rclone')

    p_boot = sub.add_parser('bootstrap', help='Provision/prepare a Colab GPU session with code and dataset')
    p_boot.add_argument('-s', '--session', default='slr-gpu')
    p_boot.add_argument('--gpu', default='L4', choices=['T4', 'L4', 'A100'])
    p_boot.add_argument('--new', action='store_true', help='Allocate a new Colab VM first')

    p_run = sub.add_parser('run-queue', help='Execute an experiment queue on the remote Colab session')
    p_run.add_argument('-s', '--session', default='slr-gpu')
    p_run.add_argument(
        '--stage',
        required=True,
        choices=['rq0', 'rq1', 'rq2', 'rq2-inter', 'rq3-train', 'rq3-eval', 'rq4-train', 'rq5-oof', 'rq5-train', 'rq5-mt', 'step2-and-3', 'all-remaining'],
    )
    p_run.add_argument('--arms', default='', help='Optional comma-separated subset of arms (e.g. rq1-01,rq1-02)')
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
