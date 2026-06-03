import argparse
import json
from pathlib import Path

import yaml


def _str_representer(dumper: yaml.SafeDumper, data: str):
    stripped = "\n".join(line.rstrip() for line in data.split("\n"))
    if stripped != data:
        print(f"While dumping YAML, {len(data) - len(stripped)} space(s) were stripped at the end of line(s).")
    return dumper.represent_scalar("tag:yaml.org,2002:str", stripped, style="|" if "\n" in data else None)


yaml.SafeDumper.add_representer(str, _str_representer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", nargs="+", type=Path, help="JSONL file paths")
    file_paths: list[Path] = parser.parse_args().file_paths

    for file_path in file_paths:
        new_file_path = file_path.with_suffix(".yaml")
        with open(file_path) as file_in, open(new_file_path, "w") as file_out:
            lines = (json.loads(line) for line in file_in)
            yaml.safe_dump_all(lines, file_out, width=float("inf"), allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
