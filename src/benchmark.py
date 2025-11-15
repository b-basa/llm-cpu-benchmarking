import subprocess
import threading
from glob import glob
from pathlib import Path

from src.config import BenchmarkConfig
from src.constants import BENCH_EXECUTABLE
from src.utilities import find_model_paths


def monitor_and_log_benchmark(proc):
    for line in proc.stdout:
        decoded_line = line.decode("utf-8").strip().lower()
        print(decoded_line)


def run_benchmarks(config: BenchmarkConfig, env: dict) -> None:
    model_paths = find_model_paths(config)
    model_paths_str = [str(p.absolute()) for p in model_paths]
    model_path_args = []
    for path in model_paths_str:
        model_path_args.append("-m")
        model_path_args.append(path)

    benchmark_args = [f"{config.llama_cpp_folder / BENCH_EXECUTABLE}"]
    benchmark_args.extend(model_path_args)

    print(f"Benchmarking {len(model_paths)} models")
    p = subprocess.Popen(
        args=benchmark_args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Start monitoring in a separate thread
    thread = threading.Thread(target=monitor_and_log_benchmark, args=(p,))
    thread_resources = threading.Thread(target=monitor_resources, args=(p,))
    thread.start()
    thread_resources.start()

    # Wait for the process to finish
    p.wait()
    thread.join()
    thread_resources.join()


def monitor_resources(proc):
    return

    # TODO: CPU is not monitored correctly
    # TODO: Check max ram usage only
    p = psutil.Process(proc.pid)
    while proc.poll() is None:  # While process is running
        try:
            cpu = p.cpu_percent(interval=1)  # % CPU usage over 1 second
            mem = p.memory_info().rss / (1024 * 1024)  # Memory in MB
            print(f"CPU: {cpu:.2f}% | Memory: {mem:.2f} MB")
        except psutil.NoSuchProcess:
            break
