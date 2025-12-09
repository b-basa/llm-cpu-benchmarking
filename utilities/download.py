from pathlib import Path
from huggingface_hub import hf_hub_download


def download_all(dry_run: bool = False, local_dir: Path | None = None):
    hf_hub_download(repo_id="ggml-org/gemma-3-4b-it-GGUF", filename="gemma-3-4b-it-Q4_K_M.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/gemma-3-4b-it-GGUF", filename="mmproj-model-f16.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", filename="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF", filename="mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen3-1.7B-GGUF", filename="Qwen3-1.7B-Q4_K_M.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen3-1.7B-GGUF", filename="Qwen3-1.7B-Q8_0.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen3-4B-GGUF", filename="Qwen3-4B-Q4_K_M.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen3-4B-GGUF", filename="Qwen3-4B-Q8_0.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/Qwen3-8B-GGUF", filename="Qwen3-8B-Q4_K_M.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/SmolVLM2-2.2B-Instruct-GGUF", filename="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf", dry_run=dry_run, local_dir=local_dir)
    hf_hub_download(repo_id="ggml-org/SmolVLM2-2.2B-Instruct-GGUF", filename="mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf", dry_run=dry_run, local_dir=local_dir)


if __name__ == "__main__":
    download_all(dry_run=True)
