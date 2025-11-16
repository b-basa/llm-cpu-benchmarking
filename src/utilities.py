from glob import glob
from pathlib import Path

from src.config import BenchmarkConfig
from src.constants import CACHE_FOLDER


def find_model_paths(config: BenchmarkConfig):
    file_names_to_find = {
        huggingface_path_to_file_name(model_name) for model_name in config.model_names
    }

    cache_folder = Path(CACHE_FOLDER)
    files_in_cache = get_cached_models(cache_folder)

    files_matched: list[Path] = []
    for name_to_find in file_names_to_find:
        file_matches = [
            Path(CACHE_FOLDER) / file_name
            for file_name in files_in_cache
            if file_name.lower().startswith(name_to_find)
        ]
        if file_matches:
            files_matched.append(file_matches[0])

    if len(file_names_to_find) != len(files_matched):
        print(f"Some models were not found under {cache_folder}")
        print(f"Searched -> {file_names_to_find}")
        print(f"Found    -> {[c.name for c in files_matched]}")
        print(f"Make sure the models exist on huggingface")
        print(f"Make sure only .gguf files are used")
    return files_matched


def get_cached_models(cache_folder: Path = Path(CACHE_FOLDER)):
    return set(glob("*.gguf", root_dir=cache_folder))


def huggingface_path_to_local_path(model_name: str):
    file_name = huggingface_path_to_file_name(model_name)
    file_path = Path(CACHE_FOLDER) / file_name / ".gguf"
    return file_path


def huggingface_path_to_file_name(model_name: str):
    return "_".join(model_name.lower().rsplit(":")[0].split("/", maxsplit=1))
