from datetime import datetime
import multiprocessing
from pathlib import Path
from utilities.common import LLAMA_CPP_FOLDER, BATCHED_BENCHMARK_EXECUTABLE, BENCH_EXECUTABLE, MODELS_FOLDER, LOGS_FOLDER
from utilities.common import ensure_logs_folder_exists, monitor_and_log_benchmark
import subprocess

def benchmark_single():
    # Get models list
    models_dir = Path(MODELS_FOLDER)
    model_paths = [str(p.absolute()) for p in models_dir.glob("*.gguf")]

    # Set up log path
    ensure_logs_folder_exists()
    log_path = Path(LOGS_FOLDER) / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Prepare process
    cmd = [f"{Path(LLAMA_CPP_FOLDER) / BENCH_EXECUTABLE}"]

    model_args = []
    for path in model_paths:
        model_args.append("-m")
        model_args.append(path)

    cores_to_use = max(1, multiprocessing.cpu_count() - 2)
    if cores_to_use > 1:
        thread_arg_str = f"1,{cores_to_use}"
    else:
        thread_arg_str = "1"

    multithreading_args = ["-t", thread_arg_str]
    benchmark_args = ["--n-prompt", "128,512", "--n-gen", "128,512"]

    cmd.extend(model_args + multithreading_args + benchmark_args)

    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        args=cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Monitor and log benchmark
    monitor_and_log_benchmark(proc, log_path)
    proc.wait()

def benchmark_batched():
    # Get models list
    models_dir = Path(MODELS_FOLDER)
    model_paths = [str(p.absolute()) for p in models_dir.glob("*.gguf")]

    # Set up log path
    ensure_logs_folder_exists()
    log_path = Path(LOGS_FOLDER) / f"batched_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Prepare process
    cmd = [f"{Path(LLAMA_CPP_FOLDER) / BATCHED_BENCHMARK_EXECUTABLE}"]

    model_args = ["-m", model_paths[0]]  # Can only benchmark one model at a time
    multithreading_args = ["-t", str(multiprocessing.cpu_count() - 1)]
    benchmark_args = ["--ctx-size", "4096", "--batch-size", "2048", "--ubatch-size", "512", "-npp", "128,512", "-ntg", "128,512", "-npl", "1"]

    cmd.extend(model_args + multithreading_args + benchmark_args)

    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        args=cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Monitor and log benchmark
    monitor_and_log_benchmark(proc, log_path)
    proc.wait()

if __name__ == "__main__":
    benchmark_single()
    benchmark_batched()