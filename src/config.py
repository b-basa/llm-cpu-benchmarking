import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.constants import CACHE_FOLDER


@dataclass
class BenchmarkConfig:
    model_names: list[str]
    cache_folder: Path
    llama_cpp_folder: Path
    test_pg: bool


def read_config(path: Path) -> BenchmarkConfig:
    j = dict()
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            contents = f.read()
            j = json.loads(contents, object_hook=as_benchmark_config)
    except Exception as ex:
        print("Error while processing config")
        print(ex)
        sys.exit()

    return j


def as_benchmark_config(d: dict):
    cache_folder = d.get("cache_folder", CACHE_FOLDER)
    llamacpp_folder = d.get("llamacpp_folder", None)

    if not llamacpp_folder:
        print("Config must contain 'llamacpp_folder'.")
        print("That folder should contain 'llama-server' and 'llama-bench'.")
        print("See -> https://github.com/ggml-org/llama.cpp/releases")
        sys.exit()

    return BenchmarkConfig(
        model_names=d["model_names"],
        cache_folder=Path(cache_folder),
        llama_cpp_folder=Path(llamacpp_folder),
        test_pg=d.get("test_pg", True),
    )


if __name__ == "__main__":
    config_path = Path("config.json")
    config = read_config(config_path)
    print(config)
