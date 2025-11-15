import os
import pprint
from pathlib import Path

from src.benchmark import run_benchmarks
from src.config import read_config
from src.download import download_models_from_config

if __name__ == "__main__":
    config = read_config(Path("config.json"))
    pprint.pprint(config)

    env = os.environ.copy()
    if not env.get("LLAMA_CACHE"):
        env["LLAMA_CACHE"] = str(config.cache_folder.absolute())

    download_models_from_config(config=config, env=env)
    run_benchmarks(config=config, env=env)
