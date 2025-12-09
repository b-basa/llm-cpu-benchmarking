from pathlib import Path
import subprocess
from utilities.common import (
    MODELS_FOLDER,
    LOGS_FOLDER,
    BENCH_EXECUTABLE,
    LLAMA_CPP_FOLDER,
    ensure_llama_cpp_folder_exists,
    ensure_logs_folder_exists,
)
from utilities.download import download_all
from datetime import datetime


def main(dry_run: bool = False, local_dir: Path = Path(MODELS_FOLDER)):
    ensure_llama_cpp_folder_exists()
    ensure_logs_folder_exists()
    download_all(dry_run=dry_run, local_dir=local_dir)
    run_benchmarks()


def run_benchmarks():
    benchmark_llms(
        model_file_names=[
            "Qwen3-1.7B-Q4_K_M.gguf",
            "Qwen3-1.7B-Q8_0.gguf",
            "gemma-3-4b-it-Q4_K_M.gguf",
            "Qwen3-4B-Q4_K_M.gguf",
            "Qwen3-4B-Q8_0.gguf",
            "Qwen3-8B-Q4_K_M.gguf",
        ],
        prompt_tokens=[16, 64, 128, 256, 1024],
        generated_tokens=[16, 128, 512, 1024],
        cores_to_use=[1, 4, 10],
    )


def benchmark_llms(
    model_file_names: list[str],
    prompt_tokens: list[int],
    generated_tokens: list[int],
    cores_to_use: list[int],
    local_dir: Path = Path(MODELS_FOLDER),
):
    cmd = get_cmd_arguments(
        model_file_names,
        prompt_tokens,
        generated_tokens,
        cores_to_use=cores_to_use,
        local_dir=local_dir,
    )
    run_and_monitor_benchmark(
        cmd, Path(LOGS_FOLDER) / f"benchmark-{get_datetime()}.csv"
    )


def test_vlms():
    # ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M
    # mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf

    # ggml-org/gemma-3-4b-it-GGUF:Q4_K_M
    # mmproj-model-f16.gguf

    # ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M
    # mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
    raise NotImplementedError()


def get_model_path_by_name(model_name: str, local_dir=Path(MODELS_FOLDER)) -> Path:
    model_path = local_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found in {local_dir}")
    return model_path


def get_base_benchmark_cmd() -> list[str]:
    return [f"{Path(LLAMA_CPP_FOLDER) / BENCH_EXECUTABLE}"]


def get_models_args(
    model_file_names: list[str], local_dir=Path(MODELS_FOLDER)
) -> list[str]:
    model_args = []
    for model_file_name in model_file_names:
        model_path = get_model_path_by_name(model_file_name, local_dir)
        model_args.append("-m")
        model_args.append(str(model_path))
    return model_args


def get_multithreading_args(cores_to_use: list[int]) -> list[str]:
    cores_str = ",".join(str(core) for core in cores_to_use)
    return [
        "--threads",
        cores_str,
    ]


def get_context_size_args(
    prompt_tokens: list[int], generated_tokens: list[int]
) -> list[str]:
    prompt_size_str = ",".join(str(size) for size in prompt_tokens)
    generation_size_str = ",".join(str(size) for size in generated_tokens)
    return [
        "--n-prompt",
        prompt_size_str,
        "--n-gen",
        generation_size_str,
    ]


def get_output_type_args() -> list[str]:
    return [
        "--output",
        "csv",
    ]


def get_other_args() -> list[str]:
    return [
        # "--progress",
        "--prio",
        "3",
        "--repetitions",
        "4",
    ]


def get_cmd_arguments(
    model_file_names: list[str],
    prompt_tokens: list[int],
    generated_tokens: list[int],
    cores_to_use: list[int],
    local_dir=Path(MODELS_FOLDER),
) -> list[str]:
    base_cmd = get_base_benchmark_cmd()
    model_args = get_models_args(model_file_names, local_dir)
    multithreading_args = get_multithreading_args(cores_to_use)
    context_size_args = get_context_size_args(prompt_tokens, generated_tokens)
    output_type_args = get_output_type_args()
    progress_args = get_other_args()
    return (
        base_cmd
        + model_args
        + multithreading_args
        + context_size_args
        + output_type_args
        + progress_args
    )


def run_and_monitor_benchmark(arguments: list[str], log_path: Path):
    print(f"Running command: {' '.join(arguments)}")
    print(f"Logging to: {log_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    proc = subprocess.Popen(
        args=arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    monitor_and_log_benchmark(proc, log_path)
    proc.wait()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def monitor_and_log_benchmark(proc, log_path: Path):
    with open(log_path, mode="a", encoding="utf-8") as f:
        for line in proc.stdout:
            decoded_line: str = line.decode("utf-8").strip().lower()
            print(decoded_line)
            print(decoded_line, file=f)


def get_datetime():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main(dry_run=False)
