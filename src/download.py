import subprocess
import threading

from src.config import BenchmarkConfig
from src.constants import SERVER_EXECUTABLE
from src.utilities import get_cached_models, huggingface_path_to_file_name


def download_models_from_config(config: BenchmarkConfig, env: dict) -> None:
    cached_models = [name.lower() for name in get_cached_models()]
    for model_name in config.model_names:
        if any(
            [
                cached.startswith(huggingface_path_to_file_name(model_name))
                for cached in cached_models
            ]
        ):
            print(f"{model_name} was already downloaded.")
        else:
            download_model(config, env, model_name)


def download_model(config: BenchmarkConfig, env: dict, model_name: str):
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


def monitor_and_kill_download(proc):
    for line in proc.stdout:
        decoded_line = line.decode("utf-8").strip().lower()
        # print(decoded_line)
        if "listening on" in decoded_line:
            print("OK")
            proc.terminate()
            break
