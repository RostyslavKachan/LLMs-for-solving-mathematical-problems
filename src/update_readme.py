"""
Author: Rostyslav Kachan (xkacha02)
Year: 2026
Description: Writes per-checkpoint validation accuracy logs to a 
             README.md file inside the fine-tuning output directory.
"""

from pathlib import Path
from datetime import datetime


def update_readme(
    readme_path: Path,
    step: int,
    accuracy: float,
    correct: int,
    total: int,
    epoch: float,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    checkpoint_name = f"checkpoint-{step}"

    with open(readme_path, "a") as f:
        f.write(
            f"| {checkpoint_name} | {step} | "
            f"{accuracy * 100:.2f}% | {correct}/{total} | {epoch:.1f} | "
            f"{timestamp} |\n"
        )


def init_readme(readme_path: Path, output_dir: str):
    if not readme_path.exists():
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(readme_path, "w") as f:
            f.write(f"# Experiment: {output_dir}\n\n")

    with open(readme_path, "r") as f:
        content = f.read()

    if "## Real Generation Accuracy During Training" not in content:
        with open(readme_path, "a") as f:
            f.write("\n## Real Generation Accuracy During Training\n\n")
            f.write("| Checkpoint | Step | Accuracy | Correct/Total | Epoch | Date |\n")
            f.write("|------------|------|----------|---------------|-------|------|\n")
