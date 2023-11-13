from optimum.neuron import NeuronModelForCausalLM
from transformers import AutoTokenizer

from ray import serve
from fastapi import FastAPI
from fastapi.responses import Response
import time
import copy
import ray 

app = FastAPI()

@serve.deployment(num_replicas=2, ray_actor_options={"resources": {"neuron_cores": 2}})
@serve.ingress(app)
class Llama:
    def __init__(self):
        self.model = NeuronModelForCausalLM.from_pretrained('aws-neuron/Llama-2-7b-hf-neuron-budget')
        self.tokenizer = AutoTokenizer.from_pretrained("aws-neuron/Llama-2-7b-hf-neuron-budget")
    
    @app.get("/")
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs,
                                     max_new_tokens=128,
                                     do_sample=True,
                                     temperature=0.9,
                                     top_k=50,
                                     top_p=0.9)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    

lldep = Llama.bind()
