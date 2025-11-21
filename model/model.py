import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig
import torch.nn.init as init
import torch.nn.functional as F
import math

"""
GPT风格实现
"""



class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob  # 是否对选中的top_k的概率进行归一化
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hiddenstate):
        B, S, H = hiddenstate.shape
        hiddenstate = rearrange(hiddenstate, "b,s,h -> (b,s),h")
        logits = F.linear(hiddenstate, self.weight)
        if self.scoring_func:
            scroes = F.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weights, topk_index = torch.topk(
            scroes, k=self.top_k, dim=-1, sorted=False
        )

        if self.top_k > 1 and self.norm_topk_prob:
            norm = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / norm

        if self.training and self.alpha > 0.0:
            scores_aux = scroes
            topk_index_for_aux =topk_index
            # seq 维度的aux loss
            if self.seq_aux:
                ce = torch.zeros([B, self.n_routed_experts], device=hiddenstate.device)
                topk_index_for_aux = rearrange(
                    topk_index, "(b s) k -> b (sk)", b=B, k=self.top_k
                )
                ce.scatter_add_(
                    1,
                    topk_index,
                    torch.ones([B, S * self.top_k], device=hiddenstate.device),
                ).div_(S * self.top_k / self.n_routed_experts)

                scores_aux = rearrange(
                    scores_aux, "(b s) e->b s e", b=B, e=self.n_routed_experts
                )
                ci = scores_aux.mean(1)

                aux_loss = (ce * ci).sum(1).mean() * self.alpha * self.n_routed_experts

            else:
                mask_ce = F.one_hot(
                    topk_index_for_aux.view(-1), num_classes=self.n_routed_experts
                )

                ce = mask_ce.float().mean(0)
                ci = scores_aux.mean(0)
                aux_loss = (ce * ci).sum(0) * self.alpha * self.n_routed_experts

        else:
            aux_loss = 0

        return topk_index, topk_weights, aux_loss
    

class MoEFeedForward(nn.Module):

