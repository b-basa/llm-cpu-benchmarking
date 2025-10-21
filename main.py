from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import psutil
from glob import glob
import pprint

CACHE_FOLDER_DEFAULT = "models"
SERVER_EXECUTABLE = "llama-server.exe"
BENCH_EXECUTABLE = "llama-bench.exe"


@dataclass
class BenchmarkConfig:
    model_names: list[str]
    cache_folder: Path
    llama_cpp_folder: Path
    test_pg: bool


def as_benchmark_config(d: dict):
    cache_folder = d.get("cache_folder", None)
    if not cache_folder:
        cache_folder = os.getenv("LLAMA_CACHE")
    if not cache_folder:
        cache_folder = CACHE_FOLDER_DEFAULT

    llamacpp_folder = d.get("llamacpp_folder", None)
    if not llamacpp_folder:
        print("Config must contain 'llamacpp_folder'.")
        print("That folder should contain 'llama-server' and 'llama-bench'.")
        sys.exit()

    return BenchmarkConfig(
        model_names=d["model_names"],
        cache_folder=Path(cache_folder),
        llama_cpp_folder=Path(llamacpp_folder),
        test_pg=d.get("test_pg", True),
    )


def read_config(path: Path) -> BenchmarkConfig:
    j = dict()
    with open(path, mode="r", encoding="utf-8") as f:
        contents = f.read()
        j = json.loads(contents, object_hook=as_benchmark_config)
    return j  # TODO might return dict...


def monitor_and_kill_download(proc):
    for line in proc.stdout:
        decoded_line = line.decode("utf-8").strip().lower()
        # print(decoded_line)
        if "listening on" in decoded_line:
            print("Model downloaded. Terminating process...")
            proc.terminate()
            break


def monitor_and_log_benchmark(proc):
    for line in proc.stdout:
        decoded_line = line.decode("utf-8").strip().lower()
        print(decoded_line)


def download_models(config: BenchmarkConfig, env: dict) -> None:
    for model_name in config.model_names:
        print(f"Downloading {model_name}")
        #  Run server to download model, slight hack
        p = subprocess.Popen(
            args=[
                f"{config.llama_cpp_folder / SERVER_EXECUTABLE}",
                "-hf",
                f"{model_name}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Start monitoring in a separate thread
        thread = threading.Thread(target=monitor_and_kill_download, args=(p,))
        thread.start()

        # Wait for the process to finish
        p.wait()
        thread.join()


def run_benchmarks(config: BenchmarkConfig, env: dict) -> None:
    model_paths = find_model_paths(config, env)
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


def find_model_paths(config: BenchmarkConfig, env: dict):
    names_to_match = set()
    for model_name in config.model_names:
        name_to_match = "_".join(
            model_name.lower().rsplit(":")[0].split("/", maxsplit=1)
        )
        names_to_match.add(name_to_match)

    cache_folder = env["LLAMA_CACHE"]
    gguf_files = set(glob("*.gguf", root_dir=cache_folder))
    common = set()
    for file in gguf_files:
        file_path = Path(cache_folder) / file
        if any([file.lower().startswith(m) for m in names_to_match]):
            common.add(file_path)

    if len(names_to_match) != len(common):
        print(f"Some models were not found in {cache_folder}")
        print(f"Searched -> {names_to_match}")
        print(f"Found    -> {[c.name for c in common]}")
    return common


if __name__ == "__main__":
    config = read_config(Path("config.json"))
    pprint.pprint(config)

    env = os.environ.copy()
    env["LLAMA_CACHE"] = str(config.cache_folder.absolute())

    download_models(config=config, env=env)
    run_benchmarks(config=config, env=env)
