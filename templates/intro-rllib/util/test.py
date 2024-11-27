import ray
import requests

FILE = 'atari_ppo.py'
TAGS_MAP = {
        f"# ws-template-imports-start\n": f"# ws-template-imports-end\n",
        f"# ws-template-code-start\n": f"# ws-template-code-end\n"
    }

RAY_VERSION=ray.__version__
URL = f"https://raw.githubusercontent.com/ray-projectl/ray/refs/heads/releases/{RAY_VERSION}/rllib/tuned_examples/ppo/atari_ppo.py"

# Send a GET request
response = requests.get(URL)

def get_segment_lines(start_tag):
    segment_lines = []
    in_segment = False
    end_tag = TAGS_MAP[start_tag]
    with open("original_atari.py", 'r') as file:
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

with open(FILE, 'r') as file:
    end_tag = None
    for line in file:
        if line in TAGS_MAP:
            lines.extend(get_segment_lines(line))
            end_tag = TAGS_MAP.get(line, None)
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