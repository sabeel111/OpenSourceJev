"""Helper script to create a Hugging Face Model repo and upload OpenSourceJev weights + docs.

Usage:
    python scripts/upload_to_hf.py --repo-id <your-hf-username>/OpenSourceJev-Qwen3-1.7B --token <hf_token>
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Upload OpenSourceJev to Hugging Face")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g., 'your-username/OpenSourceJev-Qwen3-1.7B'",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--model-path",
        default="models/Qwen3-1.7B-Q8_0.gguf",
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[!] 'huggingface_hub' not found. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    if not args.token:
        print("[!] No Hugging Face token provided.")
        print("    Pass --token <hf_token> or set the HF_TOKEN environment variable.")
        print("    Get your token at: https://huggingface.co/settings/tokens (Write access)")
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"[*] Ensuring repository exists: {args.repo_id}...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"[+] Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as exc:
        print(f"[!] Error creating repository: {exc}")
        sys.exit(1)

    # 1. Upload Model Card (README.md)
    readme_source = Path("docs/HUGGINGFACE_MODEL_CARD.md")
    if readme_source.exists():
        print(f"[*] Uploading README.md from {readme_source}...")
        api.upload_file(
            path_or_fileobj=str(readme_source),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add OpenSourceJev model card and benchmark documentation",
        )
        print("[+] README.md uploaded.")

    # 2. Upload GGUF Model weights
    model_file = Path(args.model_path)
    if model_file.exists():
        size_gb = model_file.stat().st_size / (1024 ** 3)
        print(f"[*] Uploading GGUF model weights: {model_file} ({size_gb:.2f} GB)...")
        print("    (This may take a few minutes depending on your internet upload speed)")
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo="Qwen3-1.7B-Q8_0.gguf",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add Qwen3-1.7B-Q8_0 calibrated GGUF weights",
        )
        print("[+] Model weights uploaded successfully.")
    else:
        print(f"[!] Model file not found at {model_file}. Skipping GGUF upload.")

    # 3. Upload Benchmark Summaries
    for tier in ["original", "easy", "hard"]:
        summary_path = Path(f"benchmark_results/strict_{tier}/summary.json")
        if summary_path.exists():
            print(f"[*] Uploading benchmark summary for strict_{tier}...")
            api.upload_file(
                path_or_fileobj=str(summary_path),
                path_in_repo=f"benchmarks/jevbench_{tier}_summary.json",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Add JevBench {tier} tier official summary",
            )

    print("\n" + "=" * 70)
    print(f"[✓] Published successfully!")
    print(f"    View your Hugging Face repo at: https://huggingface.co/{args.repo_id}")
    print("=" * 70)

if __name__ == "__main__":
    main()
