import multiprocessing
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from src.config import BenchmarkConfig
from src.constants import BENCH_EXECUTABLE
from src.utilities import find_model_paths


def monitor_and_log_benchmark(proc):
    log_path = Path(datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
    with open(log_path, mode="w", encoding="utf-8") as f:
        for line in proc.stdout:
            decoded_line = line.decode("utf-8").strip().lower()
            print(decoded_line)
            print(decoded_line, file=f)


def run_benchmarks_from_config(config: BenchmarkConfig, env: dict) -> None:
    model_paths = find_model_paths(config)
    model_paths_str = [str(p.absolute()) for p in model_paths]
    model_path_args = []
    for path in model_paths_str:
        model_path_args.append("-m")
        model_path_args.append(path)

    thread_amount = multiprocessing.cpu_count()
    multithreading_args = ["-t", str(thread_amount)]

    benchmark_args = [f"{config.llama_cpp_folder / BENCH_EXECUTABLE}"]
    benchmark_args.extend(model_path_args)
    benchmark_args.extend(multithreading_args)

    print(f"Benchmarking {len(model_paths)} models")
    print(f"{benchmark_args=}")
    p = subprocess.Popen(
        args=benchmark_args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    thread = threading.Thread(target=monitor_and_log_benchmark, args=(p,))
    thread.start()

    p.wait()
    thread.join()
