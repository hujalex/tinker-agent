import os
import tinker
from tinker import types
from dotenv import load_dotenv

load_dotenv()

TINKER_API_KEY = str(os.getenv("TINKER_API_KEY"))
MODEL = "Nemotron-3-Nano-30B-A3B"
LORA_RANK = 32
LEARNING_RATE = 1e-4


class Model:
    """Wraps Tinker training and sampling clients for LoRA fine-tuning and inference."""

    def __init__(self):
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=MODEL, rank=LORA_RANK
        )
        self.sampling_client = self.service_client.create_sampling_client(
            base_model=MODEL
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self.prompt: types.ModelInput | None = None
        self.params: types.SamplingParams | None = None

    def make_datum(
        self,
        input_tokens: list[int],
        target_tokens: list[int],
        weights: list[float],
    ) -> types.Datum:
        return types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
        )

    async def sample(self) -> None:
        self.prompt = types.ModelInput.from_ints(
            self.tokenizer.encode("The capital of France is")
        )
        self.params = types.SamplingParams(max_tokens=50, temperature=0.7, stop=["\n"])

        result = await self.sampling_client.sample_async(
            prompt=self.prompt, num_samples=1, sampling_params=self.params
        )
        print(self.tokenizer.decode(result.samples[0].tokens))

        result = await self.sampling_client.sample_async(
            prompt=self.prompt, num_samples=8, sampling_params=self.params
        )
        for seq in result.samples:
            print(self.tokenizer.decode(seq.tokens))

    async def compute_log_probs(self) -> None:
        result = await self.sampling_client.sample_async(
            prompt=self.prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
        )
        print(result.prompt_logprobs)  # [None, -9.5, -1.6, ...]

        logprobs = await self.sampling_client.compute_logprobs_async(self.prompt)

        result = await self.sampling_client.sample_async(
            prompt=self.prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=5,
        )
        print(result.topk_prompt_logprobs)  # [None, [(token_id, logprob), ...], ...]

    async def forward_backward(self, data: list[types.Datum]) -> None:
        fwdbwd_future = await self.training_client.forward_backward_async(
            data=data, loss_fn="cross_entropy"
        )
        fwdbwd_result = await fwdbwd_future.result_async()
        print(f"Loss: {fwdbwd_result.loss}")

        # RL losses
        # fwdbwd_future = await self.training_client.forward_backward_async(data, "importance_sampling")
        # fwdbwd_future = await self.training_client.forward_backward_async(data, "ppo")
        # fwdbwd_future = await self.training_client.forward_backward_async(data, "cispo")
        # fwdbwd_future = await self.training_client.forward_backward_async(data, "dro")

        # Custom loss
        # fwdbwd_future = await self.training_client.forward_backward_custom_async(data, my_loss_fn)

    async def optimize(self) -> None:
        optim_future = await self.training_client.optim_step_async(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )
        await optim_future.result_async()

    async def save_and_load(self) -> None:
        # Save weights → get a sampling client for evaluation
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()

        # Save full state (weights + optimizer) for resuming
        self.training_client.save_state(name="step-100")

        # Resume from weights only
        self.training_client = await self.service_client.create_training_client_from_state_async(
            path="tinker://run-id/sampler_weights/checkpoint-1"
        )

        # Resume with optimizer state
        self.training_client = await self.service_client.create_training_client_from_state_with_optimizer_async(
            path="tinker://run-id/weights/step-100"
        )


if __name__ == "__main__":
    model = Model()
