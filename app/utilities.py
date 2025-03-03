import json
import os
from hashlib import md5


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
