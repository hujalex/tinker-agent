import os
import re
import warnings
import tinker
import torch
import asyncio
from datasets import load_dataset
from tinker import TensorData
from tinker import types
from tinker_cookbook.renderers import get_renderer, get_text_content
from dotenv import load_dotenv

load_dotenv()
TINKER_API_KEY = os.getenv("TINKER_API_KEY")

class Model:
    ds = load_dataset("xw27/scibench")
    training_data = ds['train']
    question_suffix = " Provide a numerical answer without units and - in front if the answer is negative, written inside \\boxed{}."
    fewshot_prefix = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {
            "role": "assistant",
            "content": (
                "Let's spell the word out and number all the letters: "
                "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
                "We have r's at positions 3, 8, and 9. \\boxed{3}"
            ),
        },
    ]

    def __init__(self, model, renderer):
        self.base_model = model
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=model, rank=32
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self.renderer = get_renderer("llama3", self.tokenizer)
        self.sampling_params = tinker.SamplingParams(
            max_tokens=256,
            stop=self.renderer.get_stop_sequences(), 
        )
        self.adam_params = tinker.AdamParams(learning_rate=4e-5, beta1=0.9, beta2=0.95)


    def extract_boxed(self, text: str) -> str | None:
        match = re.findall(r"\\boxed\{([^}]+)\}", text)
        if match:
            return match[-1].strip()
        return None

    def grade_answer(self, response: str, ground_truth: str) -> float:
        answer = self.extract_boxed(response)
        if answer is None:
            return 0.0
        answer = answer.replace(",", "").strip()
        ground_truth = ground_truth.replace(",", "").strip()
        return 1.0 if answer == ground_truth else 0.0
    
    def extract_scibench_answer(self, text: str) -> str:
        match = re.search(r"####\s*(.+)", text)
        ans = match.group(1) if match else text
        ans = ans.replace(",", "").strip()
        ans = re.sub(r"^\+\s*(?=\d)", "", ans)
        return ans
    
    async def train(self, n_steps=10, batch_size=16, group_size=8):
        metrics_history = []

        for step in range(n_steps):
            batch_start = step * batch_size
            batch_end = batch_start + batch_size  
            batch_rows = Model.training_data.select(range(batch_start, batch_end))
            self.save_state()
            sample_results, prompts = await self.generate_sample_results(batch_rows, group_size)

            datums: list[tinker.Datum] = []
            rewards: list[float] = []
            n_degenerate = 0

            for sample_result, prompt, answer_text in zip(
                sample_results, prompts, batch_rows["answer_number"]
            ):
                ground_truth = self.extract_scibench_answer(answer_text)  

                rewards_group = []
                tokens_group = []
                logprobs_group = []

                for sequence in sample_result.sequences:
                    tokens_group.append(sequence.tokens)
                    logprobs_group.append(sequence.logprobs)
                    parsed_message, _ = self.renderer.parse_response(sequence.tokens) 
                    content = get_text_content(parsed_message)
                    reward = self.grade_answer(content, ground_truth) 
                    rewards_group.append(reward) 

                mean_reward = sum(rewards_group) / len(rewards_group)
                advantages_group = [r - mean_reward for r in rewards_group]
                rewards.append(mean_reward) 

                if all(a == 0.0 for a in advantages_group):
                    n_degenerate += 1
                    continue

                ob_len = prompt.length - 1
                for tokens, logprobs, advantage in zip(tokens_group, logprobs_group, advantages_group):
                    model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                    datums.append(self.build_datum(model_input, tokens, logprobs, advantage, ob_len))  

            self.update(datums)  

            mean_reward = sum(rewards) / len(rewards)
            frac_degenerate = n_degenerate / len(rewards)
            metrics_history.append(
                {"step": step, "reward": mean_reward, "frac_degenerate": frac_degenerate}
            )
            print(
                f"Step {step:2d} | reward: {mean_reward:.3f} | "
                f"degenerate: {frac_degenerate:.0%} | datums: {len(datums)}" 
            )
    
    def save_state(self):
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()
            
    async def generate_sample_results(self, batch_rows, group_size):
        prompts: list[tinker.ModelInput] = []
        coros = []
        for question in batch_rows["problem_text"]:
            convo = [*self.fewshot_prefix, {"role": "user", "content": question + self.question_suffix}] 
            prompt = self.renderer.build_generation_prompt(convo)
            coros.append(  
                self.sampling_client.sample_async( 
                    prompt=prompt, num_samples=group_size,
                    sampling_params=self.sampling_params 
                )
            )
            prompts.append(prompt)

        sample_results = await asyncio.gather(*coros)
        return sample_results, prompts

    def build_datum(self, model_input, tokens, logprobs, advantage, ob_len):
        target_tokens = [0] * ob_len + list(tokens)
        padded_logprobs = [0.0] * ob_len + list(logprobs)
        padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens":
                    TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs":
                    TensorData.from_torch(torch.tensor(padded_logprobs)),
                "advantages":
                    TensorData.from_torch(torch.tensor(padded_advantages)),
            },
        )
        return datum

    def update(self, datums):
        if len(datums) > 0:
            fwd_bwd_future = self.training_client.forward_backward(datums, loss_fn="importance_sampling")
            optim_future = self.training_client.optim_step(self.adam_params) 
            fwd_bwd_future.result()
            optim_future.result()
            
    def prompt(self, question):
        result = self.sampling_client.sample(
            prompt=question, num_samples=1, sampling_params=self.sampling_params
        ).result()
        print(self.tokenizer.decode(result.sequences[0].tokens))


if __name__ == "__main__":
    model = Model("meta-llama/Llama-3.1-8B", "llama3")
    asyncio.run(model.train())
    model.prompt("A 10 kg body moving at sticks to a 15 kg body at rest. What is the final velocity?")
    
