import json


def load_json(file_path_and_name):
    with open(file_path_and_name, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path_and_name):
    with open(file_path_and_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
