import subprocess
import threading

from src.config import BenchmarkConfig
from src.constants import SERVER_EXECUTABLE


def download_models_from_config(config: BenchmarkConfig, env: dict) -> None:
    for model_name in config.model_names:
        download_model(config, env, model_name)


def download_model(config: BenchmarkConfig, env: dict, model_name: str):
    print(f"Validating download {model_name}")
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
