import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoAWQForCausalLM.from_pretrained(
              model_id,
              torch_dtype=torch.float16,
              low_cpu_mem_usage=True,
              device_map="auto",
            )

    def infer(self, inputs):
        prompt_user = inputs["prompt"]
        prompt = [{"role": "user", "content": prompt_user}]
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        outputs = self.model.generate(**inputs, do_sample=True, max_new_tokens=256)
        generated_text = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return {'generated_result': generated_text}

    def finalize(self):
        pass
