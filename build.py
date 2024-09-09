import yaml
import sys
import zipfile
import os.path

def logln(msg):
    print(msg, file=sys.stderr)


def read_compute_config_file(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

    meta_yaml = yaml.dump(meta, default_flow_style=False)
    output_zip = os.path.join(output_dir, f"{name}.zip")

    dir = tmpl.get("dir")
    if os.path.exists(output_zip) and os.path.isfile(output_zip):
        os.remove(output_zip)

    with zipfile.ZipFile(output_zip, "w") as z:
        logln(f"[{name}] adding .meta/app.yaml")
        z.writestr(".meta/app.yaml", meta_yaml)

        logln(f"[{name}] walking files in {dir}")

        if dir:
            if not os.path.isdir(dir):
                raise FileNotFoundError(f"directory {dir} not found")

            for root, _, files in os.walk(dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_in_zip = os.path.relpath(file_path, dir)
                    logln(f"[{name}] adding {file_in_zip}")
                    z.write(file_path, file_in_zip)

def build_all(build_file = "BUILD.yaml", output_dir = "_build") -> None:
    with open(build_file, "r") as f:
        build = yaml.safe_load(f)
    
    assert isinstance(build, list)

    os.makedirs(output_dir, exist_ok=True)
    for item in build:
        build_template(item, output_dir)


if __name__ == "__main__":
    build_all()