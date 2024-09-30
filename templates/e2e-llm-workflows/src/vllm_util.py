from src.utils import get_lora_path
from vllm import LLM
from vllm.lora.request import LoRARequest

class LLMPredictor:
    def __init__(self, fine_tuned_model, sampling_params):
        self.llm = LLM(model=fine_tuned_model.base_model_id, enable_lora=True)
        self.sampling_params = sampling_params
        self.lora_path = get_lora_path(fine_tuned_model)

    def __call__(self, batch):
        if not self.lora_path:
            outputs = self.llm.generate(
                prompts=batch['inputs'],
                sampling_params=self.sampling_params)
        else:
            outputs = self.llm.generate(
                prompts=batch['inputs'],
                sampling_params=self.sampling_params,
                lora_request=LoRARequest('lora_adapter', 1, self.lora_path))
        inputs = []
        generated_outputs = []
        for output in outputs:
            inputs.append(output.prompt)
            generated_outputs.append(' '.join([o.text for o in output.outputs]))
        return {
            'prompt': inputs,
            'expected_output': batch['outputs'],
            'generated_text': generated_outputs,
        }
