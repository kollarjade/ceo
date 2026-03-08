"""
EFLA Trainer - Modal Cloud Launcher
Train 1T parameter models on 8×B200 GPUs
"""

import modal

# Define the Modal app
app = modal.App("efla-trainer")

# Define the container image with CUDA 12.8 and build artifacts
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "libnccl2",
        "libnccl-dev",
    )
    .run_commands(
        # Install Zig
        "curl -L https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ",
        "mv zig-linux-x86_64-0.13.0 /opt/zig",
        "ln -s /opt/zig/zig /usr/local/bin/zig",
    )
    .pip_install(
        "modal",
        "pyyaml",
    )
    .copy_local_dir("/home/z/my-project/efla-trainer", "/app/efla-trainer")
    .run_commands(
        # Build the project
        "cd /app/efla-trainer && zig build -Doptimize=ReleaseFast",
    )
)

# Define volumes for data and checkpoints
data_volume = modal.Volume.from_name("efla-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("efla-checkpoints", create_if_missing=True)

# Define the 8×B200 GPU cluster
@app.cls(
    image=image,
    gpu="B200",
    gpu_count=8,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=86400,  # 24 hours
    retries=0,
)
class EflaTrainer:
    @modal.enter()
    def setup(self):
        """Setup runs once per container startup."""
        import subprocess
        import os

        # Verify GPU setup
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
            capture_output=True,
            text=True
        )
        print(f"GPU Setup:\n{result.stdout}")

        # Set environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        os.environ["NCCL_DEBUG"] = "INFO"

    @modal.method()
    def train(self, config_path: str, resume_from: str | None = None):
        """Run distributed training on 8×B200."""
        import subprocess
        import os

        # Build command
        cmd = [
            "/app/efla-trainer/zig-out/bin/efla-train",
            "train",
            "--config", f"/app/efla-trainer/{config_path}",
            "--data", "/data/train.bin",
            "--checkpoint-dir", "/checkpoints",
        ]

        if resume_from:
            cmd.extend(["--resume", resume_from])

        print(f"Running: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            env=os.environ,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Training failed with code {result.returncode}")

        # Commit checkpoint volume
        checkpoint_volume.commit()

    @modal.method()
    def evaluate(self, checkpoint_path: str, data_path: str):
        """Run evaluation."""
        import subprocess

        cmd = [
            "/app/efla-trainer/zig-out/bin/efla-train",
            "evaluate",
            "--checkpoint", checkpoint_path,
            "--data", data_path,
        ]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed")

    @modal.method()
    def generate(self, checkpoint_path: str, prompt: str, max_tokens: int = 256):
        """Generate text."""
        import subprocess

        cmd = [
            "/app/efla-trainer/zig-out/bin/efla-train",
            "generate",
            "--checkpoint", checkpoint_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Generation failed")

        return result.stdout

    @modal.method()
    def status(self):
        """Get training status."""
        import subprocess

        # Check GPU status
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv"],
            capture_output=True,
            text=True
        )

        # Check latest checkpoint
        import os
        checkpoint_dir = "/checkpoints"
        checkpoints = []
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith("step_")])

        return {
            "gpu_status": gpu_result.stdout,
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
            "checkpoint_count": len(checkpoints),
        }


@app.local_entrypoint()
def main(
    config: str = "configs/train.yaml",
    resume: str | None = None,
    action: str = "train",
):
    """Main entry point for Modal training."""

    trainer = EflaTrainer()

    if action == "train":
        print(f"Starting training with config: {config}")
        if resume:
            print(f"Resuming from: {resume}")
        trainer.train.remote(config, resume)
        print("Training complete!")

    elif action == "evaluate":
        print("Running evaluation...")
        trainer.evaluate.remote("/checkpoints/latest", "/data/eval.bin")

    elif action == "generate":
        print("Generating text...")
        output = trainer.generate.remote("/checkpoints/latest", "Hello, world!")
        print(output)

    elif action == "status":
        status = trainer.status.remote()
        print(f"Status: {status}")

    else:
        print(f"Unknown action: {action}")
        print("Options: train, evaluate, generate, status")


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
)
def smoke_test():
    """Run smoke test on a single B200."""
    import subprocess

    result = subprocess.run(
        ["/app/efla-trainer/zig-out/bin/efla-train", "smoke-test", "--config", "configs/smoke.yaml"],
        capture_output=False,
        text=True,
    )

    return result.returncode == 0


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": data_volume},
)
def upload_data(local_path: str, remote_name: str):
    """Upload training data to Modal volume."""
    import subprocess

    # Copy local file to volume
    subprocess.run(["cp", local_path, f"/data/{remote_name}"], check=True)

    # Commit volume
    data_volume.commit()

    print(f"Uploaded {local_path} to {remote_name}")


@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
)
def download_checkpoint(checkpoint_name: str, local_path: str):
    """Download checkpoint from Modal volume."""
    import subprocess

    # Copy from volume to local
    subprocess.run(
        ["cp", "-r", f"/checkpoints/{checkpoint_name}", local_path],
        check=True
    )

    print(f"Downloaded {checkpoint_name} to {local_path}")
