from pathlib import Path

LLAMA_CPP_FOLDER = "llama-b7083-bin-win-cpu-x64"
BATCHED_BENCHMARK_EXECUTABLE = "llama-batched-bench.exe"
BENCH_EXECUTABLE = "llama-bench.exe"
MODELS_FOLDER = "models"
LOGS_FOLDER = "logs"


def ensure_llama_cpp_folder_exists():
    llama_path = Path(LLAMA_CPP_FOLDER)
    if not llama_path.exists() or not llama_path.is_dir():
        raise FileNotFoundError(
            f"""Llama.cpp folder not found at path: {llama_path.absolute()}
            \nPlease ensure llama.cpp is downloaded and placed in the correct directory.
            \nYou can download it from: https://github.com/ggml-org/llama.cpp/releases""",
        )


def ensure_logs_folder_exists():
    logs_path = Path(LOGS_FOLDER)
    logs_path.mkdir(parents=True, exist_ok=True)


def monitor_and_log_benchmark(proc, log_path: Path):
    logs = []
    for line in proc.stdout:
        decoded_line: str = line.decode("utf-8").strip().lower()
        print(decoded_line)
        logs.append(decoded_line)

    with open(log_path, mode="w", encoding="utf-8") as f:
        for log_line in logs:
            print(log_line, file=f)
