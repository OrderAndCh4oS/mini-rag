import html
import json
import os
import re
from hashlib import md5
from typing import Union


def create_file_if_not_exists(file_path, default_content=""):
    if not os.path.exists(file_path):
        write_file(file_path, default_content)


def write_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


def read_file(file_path):
    f = open(file_path, "r")
    return f.read()


def get_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path, data):
    with open(file_path, "w") as f:
        return json.dump(data, f, indent=4)


def get_docs(root_dir):
    text_files = []
    for path, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.txt', '.md', '.mdx', '.yaml', '.yml', '.tex', '.rst')):
                text_files.append(os.path.join(path, filename))
    return text_files


def make_hash(text, prefix=""):
    return prefix + md5(text.encode()).hexdigest()


def add_to_json(file_path, key, value):
    with open(file_path, "r+") as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


def remove_from_json(file_path, key):
    with open(file_path, "r+") as f:
        data = json.load(f)
        if key not in data:
            return
        del data[key]
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(text: str) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    assert isinstance(text, str)

    result = html.unescape(text.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers."""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def extract_json_from_text(content: str):
    json_str = re.search(r"{.*}", content, re.DOTALL)
    if json_str is not None:
        json_str = json_str.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None