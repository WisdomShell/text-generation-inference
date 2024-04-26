import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_shell_modeling import (
    FlashShellForCausalLM,
    ShellConfig,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)


class FlashShell(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        use_medusa: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        
        config = ShellConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        config.num_key_value_heads = config.num_query_groups
        config.hidden_act = config.activation_function
        config.intermediate_size = config.n_embd
        # config.max_position_embeddings = config.n_positions
        config.quantize = quantize
        config.transpose = config.architectures[0].startswith("GPT2")
        config.device = device
        config.use_medusa = use_medusa

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group, aliases={"transformer.wte.weight": ["lm_head.weight"]})
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id)
        model = FlashShellForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashShell, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
