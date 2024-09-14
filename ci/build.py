import yaml
import sys
import zipfile
import os.path

def logln(msg):
    print(msg, file=sys.stderr)


def read_compute_config_file(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Builder:
    def __init__(self, meta: dict):
        name = meta.get("name", "")
        if not name:
            raise ValueError("missing name in meta")
        input_dir = meta.get("dir", "")
        if not input_dir:
            raise ValueError("missing dir in meta")
        
        self._in = input_dir
        self._meta = meta
        self._name = name

    def build(self, output_dir: str):
        meta_yaml = yaml.dump(self._meta, default_flow_style=False)

        files = []
        for root, _, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_in_zip = os.path.relpath(file_path, dir)
                if file_in_zip == "README.md":
                    continue # skip README.md; it is drived from the notebook.
                files.append(file_in_zip)
        
        output_zip = os.path.join(output_dir, self._name, "files.zip")
        with zipfile.ZipFile(output_zip, "w") as z:
            z.writestr(".meta/ray-app.yaml", meta_yaml)

            for file in files:
                z.write(os.path.join(self._in, file), file)


def build_template(tmpl: dict, output_dir: str) -> None:
    name = tmpl.get("name")
    assert name is not None

    logln(f"Building {name}...")
    
    meta = dict(tmpl)
    del meta["compute_config"]
    del meta["dir"]

    meta["compute_config"] = dict()

    for cloud, config_file in tmpl["compute_config"].items():
        config = read_compute_config_file(config_file)
        meta["compute_config"][cloud] = config

    builder = Builder(meta)
    builder.build(output_dir)


def build_all(build_file = "BUILD.yaml", output_dir = "_build") -> None:
    with open(build_file, "r") as f:
        build = yaml.safe_load(f)
    
    assert isinstance(build, list)

    os.makedirs(output_dir, exist_ok=True)
    for item in build:
        build_template(item, output_dir)


if __name__ == "__main__":
    build_all()
