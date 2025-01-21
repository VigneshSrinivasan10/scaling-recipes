import torch 
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

class FlowModel(nn.Module):

    def forward(self, X, time):
        raise NotImplementedError()

class ZeroToOneTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.register_buffer('freqs', torch.arange(1, dim // 2 + 1) * torch.pi)

    def forward(self, t):
        emb = self.freqs * t[..., None]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FlowMLP(FlowModel):
    def __init__(self, n_features=2, width=10, n_blocks=5, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super().__init__()

        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult

        self.n_blocks = n_blocks
        self.time_embedding_size = width - n_features
        
        self.time_embedding = ZeroToOneTimeEmbedding(self.time_embedding_size)
        blocks = []
        for _ in range(self.n_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(width, width, bias=False),
                nn.SiLU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(width, n_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self, base_std=0.02) -> None:
        # init all weights with fan_in / 1024 * base_std
        for n, p in self.named_parameters():
            skip_list = ["final"]
            if not any([s in n for s in skip_list]):
                fan_in = p.shape[1]
                p.data.normal_(mean=0.0, std=base_std * (fan_in) ** -0.5)
        # init final layer with zeros
        nn.init.zeros_(self.final.weight)

    def forward(self, X, time=None):
        if time is None:
            time = torch.rand(X.shape[0], device=X.device)
        X = torch.cat([X, self.time_embedding(time)], axis=1)
        for block in self.blocks:
            X = X + block(X)
        X = self.final(X)
        return X
    
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        no_decay_name_list = ["bias", "norm"]

        optimizer_grouped_parameters = []
        final_optimizer_settings = {}

        param_groups = defaultdict(
            lambda: {"params": [], "weight_decay": None, "lr": None}
        )

        for n, p in self.named_parameters():
            if p.requires_grad:

                # Define learning rate for specific types of params
                if any(ndnl in n for ndnl in no_decay_name_list):
                    lr_value = learning_rate * 0.1
                    per_layer_weight_decay_value = 0.0
                else:
                    hidden_dim = p.shape[-1]
                    lr_value = learning_rate * (32 / hidden_dim)
                    per_layer_weight_decay_value = (
                        weight_decay * hidden_dim / 1024
                    )  # weight decay 0.1 (SP: 1024)

                # in the case of embedding layer, we use higher lr.
                if "time_embedding" in n:
                    lr_value = learning_rate * 0.3
                    per_layer_weight_decay_value = 0.0

                group_key = (lr_value, per_layer_weight_decay_value)
                param_groups[group_key]["params"].append(p)
                param_groups[group_key]["weight_decay"] = per_layer_weight_decay_value
                param_groups[group_key]["lr"] = lr_value

                final_optimizer_settings[n] = {
                    "lr": lr_value,
                    "wd": per_layer_weight_decay_value,
                    "shape": str(list(p.shape)),
                }

        optimizer_grouped_parameters = [v for v in param_groups.values()]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=betas)

        return optimizer, final_optimizer_settings

