from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = "efla-trainer"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_APP_DIR = "/app/efla-trainer"
DATA_VOLUME_NAME = "efla-data"
CHECKPOINT_VOLUME_NAME = "efla-checkpoints"
DATA_MOUNT_DIR = "/data"
CHECKPOINT_MOUNT_DIR = "/checkpoints"
TRAIN_BINARY = f"{REMOTE_APP_DIR}/zig-out/bin/efla-train"
STEP_DIR_RE = re.compile(r"^step_(\d+)$")

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "bash",
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "libnccl-dev",
        "libnccl2",
        "pkg-config",
        "xz-utils",
    )
    .pip_install("modal", "pyyaml")
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=REMOTE_APP_DIR,
        copy=True,
        ignore=[
            ".git",
            ".github",
            ".venv",
            ".idea",
            ".vscode",
            "__pycache__",
            "*.pyc",
            "zig-cache",
            "zig-out",
        ],
    )
    .run_commands(
        "bash -lc 'set -euo pipefail; cd /tmp; curl -fsSL https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz -o zig.tar.xz; tar -xJf zig.tar.xz; mv zig-linux-x86_64-0.13.0 /opt/zig; ln -sf /opt/zig/zig /usr/local/bin/zig'",
        f"bash -lc 'set -euo pipefail; cd {shlex.quote(REMOTE_APP_DIR)}; zig build -Doptimize=ReleaseFast'",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App(APP_NAME, image=image)


def _normalize_volume_subpath(value: str, mount_prefix: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if normalized == "":
        return ""
    if normalized == mount_prefix:
        return ""
    prefix = f"{mount_prefix}/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    normalized = normalized.lstrip("/")
    if normalized == ".":
        return ""
    return normalized


def _remote_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(REMOTE_APP_DIR) / path_str


def _remote_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    normalized = _normalize_volume_subpath(path_str, DATA_MOUNT_DIR)
    return Path(DATA_MOUNT_DIR) / normalized


def _checkpoint_root() -> Path:
    return Path(CHECKPOINT_MOUNT_DIR)


def _checkpoint_sort_key(path: Path) -> tuple[int, int, int, str]:
    match = STEP_DIR_RE.fullmatch(path.name)
    if match:
        return (0, int(match.group(1)), 0, path.name)
    stat = path.stat()
    return (1, int(stat.st_mtime_ns), int(stat.st_ctime_ns), path.name)


def _latest_checkpoint_path(root: Path | None = None) -> Path | None:
    checkpoint_root = root or _checkpoint_root()
    latest_link = checkpoint_root / "latest"
    if latest_link.exists():
        return latest_link
    if not checkpoint_root.exists():
        return None
    candidates = [path for path in checkpoint_root.iterdir() if path.is_dir() or path.is_symlink()]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_sort_key)


def _remote_checkpoint_path(path_str: str) -> Path:
    raw = path_str.strip()
    if raw == "":
        raw = "latest"
    absolute = Path(raw)
    if absolute.is_absolute():
        candidate = absolute
    else:
        normalized = _normalize_volume_subpath(raw, CHECKPOINT_MOUNT_DIR)
        if normalized in ("", "latest"):
            latest = _latest_checkpoint_path()
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {CHECKPOINT_MOUNT_DIR}")
            return latest
        candidate = Path(CHECKPOINT_MOUNT_DIR) / normalized
    if candidate.exists():
        return candidate
    normalized_candidate = _normalize_volume_subpath(raw, CHECKPOINT_MOUNT_DIR)
    if normalized_candidate == "latest":
        latest = _latest_checkpoint_path()
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found under {CHECKPOINT_MOUNT_DIR}")
        return latest
    raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")


def _require_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _completed_process_output(result: subprocess.CompletedProcess[str]) -> str:
    pieces: list[str] = []
    if result.stdout:
        pieces.append(result.stdout.strip())
    if result.stderr:
        pieces.append(result.stderr.strip())
    return "\n".join(piece for piece in pieces if piece)


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {"cwd": cwd, "env": env, "text": True}
    if capture_output:
        kwargs["capture_output"] = True
    return subprocess.run(cmd, check=False, **kwargs)


def _run_subprocess_checked(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = _run_subprocess(cmd, cwd=cwd, env=env, capture_output=capture_output)
    if result.returncode != 0:
        detail = _completed_process_output(result)
        message = f"Command failed with exit code {result.returncode}: {shlex.join(cmd)}"
        if detail:
            message = f"{message}\n{detail}"
        raise RuntimeError(message)
    return result


def _resolve_cli_binary() -> str:
    cli = shutil.which("modal")
    if cli is None:
        raise RuntimeError("The modal CLI executable was not found in PATH")
    return cli


def _local_modal_volume_put(volume_name: str, local_path: Path, remote_path: str) -> None:
    cli = _resolve_cli_binary()
    cmd = [cli, "volume", "put", volume_name, str(local_path)]
    if remote_path:
        cmd.append(remote_path)
    subprocess.run(cmd, check=True)


def _local_modal_volume_get(volume_name: str, remote_path: str, local_destination: Path) -> None:
    cli = _resolve_cli_binary()
    cmd = [cli, "volume", "get", volume_name, remote_path, str(local_destination), "--force"]
    subprocess.run(cmd, check=True)


def _local_upload_data(local_path: str, remote_name: str) -> None:
    source = Path(local_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Local path does not exist: {source}")
    target = remote_name.strip().replace("\\", "/")
    if target == "":
        target = source.name + ("/" if source.is_dir() else "")
    target = _normalize_volume_subpath(target, DATA_MOUNT_DIR)
    if source.is_dir() and target != "" and not target.endswith("/"):
        target = f"{target}/"
    _local_modal_volume_put(DATA_VOLUME_NAME, source, target)


def _local_download_checkpoint(checkpoint_name: str, local_path: str) -> None:
    remote_path = checkpoint_name.strip()
    if remote_path == "":
        remote_path = "latest"
    remote_path = _normalize_volume_subpath(remote_path, CHECKPOINT_MOUNT_DIR)
    if remote_path == "":
        remote_path = "latest"
    destination = Path(local_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _local_modal_volume_get(CHECKPOINT_VOLUME_NAME, remote_path, destination)


@app.cls(
    gpu="B200:8",
    volumes={
        DATA_MOUNT_DIR: data_volume,
        CHECKPOINT_MOUNT_DIR: checkpoint_volume,
    },
    timeout=86400,
    retries=0,
)
class EflaTrainer:
    @modal.enter()
    def setup(self) -> None:
        result = _run_subprocess_checked(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
        )
        gpu_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(gpu_lines) != 8:
            raise RuntimeError(f"Expected 8 GPUs, found {len(gpu_lines)}\n{result.stdout}")
        self.gpu_info = gpu_lines
        self.runtime_env = os.environ.copy()
        self.runtime_env.setdefault("NCCL_DEBUG", "INFO")
        self.runtime_env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        self.runtime_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        self.runtime_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(len(gpu_lines)))

    @modal.method()
    def train(self, config_path: str, resume_from: str | None = None, data_path: str = "train.bin") -> dict[str, object]:
        data_volume.reload()
        checkpoint_volume.reload()
        config = _require_existing_path(_remote_project_path(config_path), "Config path")
        data_file = _require_existing_path(_remote_data_path(data_path), "Training data path")
        cmd = [
            TRAIN_BINARY,
            "train",
            "--config",
            str(config),
            "--data",
            str(data_file),
            "--checkpoint-dir",
            CHECKPOINT_MOUNT_DIR,
        ]
        resume_path: Path | None = None
        if resume_from:
            resume_path = _remote_checkpoint_path(resume_from)
            cmd.extend(["--resume", str(resume_path)])
        result = _run_subprocess(cmd, cwd=REMOTE_APP_DIR, env=self.runtime_env, capture_output=False)
        checkpoint_volume.commit()
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {result.returncode}: {shlex.join(cmd)}")
        latest = _latest_checkpoint_path()
        return {
            "ok": True,
            "config": str(config),
            "data": str(data_file),
            "resume_from": str(resume_path) if resume_path is not None else None,
            "latest_checkpoint": str(latest) if latest is not None else None,
        }

    @modal.method()
    def evaluate(self, checkpoint_path: str, data_path: str) -> dict[str, object]:
        data_volume.reload()
        checkpoint_volume.reload()
        checkpoint = _require_existing_path(_remote_checkpoint_path(checkpoint_path), "Checkpoint path")
        data_file = _require_existing_path(_remote_data_path(data_path), "Evaluation data path")
        result = _run_subprocess_checked(
            [
                TRAIN_BINARY,
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(data_file),
            ],
            cwd=REMOTE_APP_DIR,
            env=self.runtime_env,
            capture_output=True,
        )
        return {
            "ok": True,
            "checkpoint": str(checkpoint),
            "data": str(data_file),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    @modal.method()
    def generate(self, checkpoint_path: str, prompt: str, max_tokens: int = 256) -> str:
        checkpoint_volume.reload()
        checkpoint = _require_existing_path(_remote_checkpoint_path(checkpoint_path), "Checkpoint path")
        result = _run_subprocess_checked(
            [
                TRAIN_BINARY,
                "generate",
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                prompt,
                "--max-tokens",
                str(max_tokens),
            ],
            cwd=REMOTE_APP_DIR,
            env=self.runtime_env,
            capture_output=True,
        )
        return result.stdout

    @modal.method()
    def status(self) -> dict[str, object]:
        checkpoint_volume.reload()
        gpu_result = _run_subprocess_checked(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
        )
        checkpoint_root = _checkpoint_root()
        checkpoint_entries: list[str] = []
        if checkpoint_root.exists():
            checkpoint_entries = sorted(
                [entry.name for entry in checkpoint_root.iterdir() if entry.is_dir() or entry.is_symlink()],
                key=lambda name: _checkpoint_sort_key(checkpoint_root / name),
            )
        latest = _latest_checkpoint_path(checkpoint_root)
        return {
            "gpu_status": [line.strip() for line in gpu_result.stdout.splitlines() if line.strip()],
            "gpu_count": len([line for line in gpu_result.stdout.splitlines() if line.strip()]),
            "latest_checkpoint": str(latest) if latest is not None else None,
            "checkpoint_count": len(checkpoint_entries),
            "checkpoints": checkpoint_entries,
        }


@app.function(gpu="B200", timeout=3600)
def smoke_test(config_path: str = "configs/smoke.yaml") -> bool:
    config = _require_existing_path(_remote_project_path(config_path), "Smoke test config path")
    _run_subprocess_checked(
        [TRAIN_BINARY, "smoke-test", "--config", str(config)],
        cwd=REMOTE_APP_DIR,
        capture_output=False,
    )
    return True


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/train.yaml",
    resume: str = "",
    data: str = "train.bin",
    checkpoint: str = "latest",
    prompt: str = "Hello, world!",
    max_tokens: int = 256,
    local_path: str = "",
    remote_name: str = "",
    output_path: str = "",
) -> None:
    if action == "upload_data":
        if local_path == "":
            raise SystemExit("local_path is required for action=upload_data")
        _local_upload_data(local_path, remote_name)
        return
    if action == "download_checkpoint":
        destination = output_path.strip()
        if destination == "":
            destination = Path(checkpoint if checkpoint.strip() else "latest").name or "latest"
        _local_download_checkpoint(checkpoint, destination)
        return
    if action == "smoke_test":
        smoke_config = config if config != "configs/train.yaml" else "configs/smoke.yaml"
        print(smoke_test.remote(smoke_config))
        return
    trainer = EflaTrainer()
    if action == "train":
        result = trainer.train.remote(config, resume or None, data)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if action == "evaluate":
        eval_data = data if data != "train.bin" else "eval.bin"
        result = trainer.evaluate.remote(checkpoint, eval_data)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if action == "generate":
        output = trainer.generate.remote(checkpoint, prompt, max_tokens)
        print(output, end="" if output.endswith("\n") else "\n")
        return
    if action == "status":
        result = trainer.status.remote()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    raise SystemExit(
        "Unknown action. Valid actions: train, evaluate, generate, status, smoke_test, upload_data, download_checkpoint"
    )
