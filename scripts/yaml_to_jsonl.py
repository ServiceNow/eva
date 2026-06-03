import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", nargs="+", type=Path, help="YAML file paths")
    file_paths: list[Path] = parser.parse_args().file_paths

    for file_path in file_paths:
        new_file_path = file_path.with_suffix(".jsonl")
        with open(file_path) as file_in, open(new_file_path, "w") as file_out:
            lines = yaml.safe_load_all(file_in)
            file_out.writelines(json.dumps(line, ensure_ascii=False, separators=(", ", ": ")) + "\n" for line in lines)


if __name__ == "__main__":
    main()
