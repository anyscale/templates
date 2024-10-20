import dspy
from src import LOCAL_API_PARAMETERS, MODEL_PARAMETERS
import os

os.environ["DSP_CACHEBOOL"] = "false"
os.environ["DSP_CACHEDIR"] = "/home/ray/default/dspy/temp_cache"

if not os.path.exists(os.environ["DSP_CACHEDIR"]):
    os.makedirs(os.environ["DSP_CACHEDIR"])
# clear cache
for file in os.listdir(os.environ["DSP_CACHEDIR"]):
    os.remove(os.path.join(os.environ["DSP_CACHEDIR"], file))

LOCAL_API_PARAMETERS = {
  "api_base": "http://localhost:8000/v1",
  "api_key": "fake-key-doesnt-matter"
}
lm = dspy.LM("openai/meta-llama/Meta-Llama-3.1-70B-Instruct", **LOCAL_API_PARAMETERS, cache=False)

predictor = dspy.ChainOfThought("question -> answer")

metric = lambda x,y,z=None: 1
num_threads = 100
display_progress = True
max_errors = 10000
questions = [dspy.Example(question=f"What is 1 + {i+1000}?").with_inputs("question") for i in range(num_threads)]

common_kwargs = dict(metric=metric, num_threads=num_threads, display_progress=display_progress, max_errors=max_errors, return_outputs=True)
evaluate = dspy.Evaluate(devset=questions, **common_kwargs)
with dspy.context(lm=lm):
    x = evaluate(predictor)
    
print([x[1][i][1]["answer"] for i in range(len(x[1]))])