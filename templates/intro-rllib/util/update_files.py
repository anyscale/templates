import ray
import requests


RAY_VERSION = ray.__version__


def update_file(file_name, url, tags_map):
    # Send a GET request
    response = requests.get(url)

    def get_segment_lines(start_tag):
        segment_lines = []
        in_segment = False
        end_tag = tags_map[start_tag]
        if file_name == "atari_ppo.py":
            _file_name = "original_atari.py"
        else:
            _file_name = "original_custom_gym_env"
        with open(_file_name, 'r') as file:
            for line in file:
        # for line in response.iter_lines():
            # line = str(line)
                if start_tag in line:
                    in_segment = True
                if in_segment:
                    segment_lines.append(line)
                if end_tag in line:
                    in_segment = False
            
        return segment_lines

    lines = []

    with open(file_name, 'r') as file:
        end_tag = None
        for line in file:
            if line in tags_map:
                lines.extend(get_segment_lines(line))
                end_tag = tags_map.get(line, None)
            if end_tag:
                if end_tag in line:
                    end_tag = None
                continue
            else:
                lines.append(line)
            
    with open(FILE, 'w') as file:
        for line in lines:
            file.write(line)

    print(f"File {FILE} created with {len(lines)} lines.")

TAGS_MAP = {
        f"# ws-template-imports-start\n": f"# ws-template-imports-end\n",
        f"# ws-template-code-start\n": f"# ws-template-code-end\n"
    }

# Update atari_ppo.py:
update_file(
    file_name="atari_ppo.py",
    url=f"https://raw.githubusercontent.com/ray-project/ray/refs/heads/releases/{RAY_VERSION}/rllib/tuned_examples/ppo/atari_ppo.py",
    tags_map=TAGS_MAP
    )

# Update custom_gym_env.py:
update_file(
    file_name="../custom_gym_env/atari_ppo.py",
    url=f"https://raw.githubusercontent.com/ray-project/ray/refs/heads/releases/{RAY_VERSION}/rllib/examples/envs/custom_gym_env.py",
    tags_map=TAGS_MAP
    )