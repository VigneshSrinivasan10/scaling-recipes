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

class LeNet(FlowModel):
    def __init__(self, n_classes=10, width=120, n_blocks=5, nonlin=F.relu, output_mult=1.0, input_mult=1.0, parametrization="mup"):
        super().__init__()
        
        self.nonlin = nonlin

        # Not used for anything
        self.n_blocks = n_blocks
        self.input_mult = input_mult 
        self.output_mult = output_mult

        self.parametrization = parametrization

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, n_classes)

        if self.parametrization == "mup":
            self.reset_parameters_mup()
        elif self.parametrization == "sp":
            self.reset_parameters_sp()
        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")

    def reset_parameters_mup(self, base_std=0.02) -> None:
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'conv' in n:
                    fan_in = p.shape[1] * p.shape[2] * p.shape[3]
                else:
                    fan_in = p.shape[1]
                p.data.normal_(mean=0.0, std=base_std * (fan_in) ** -0.5)
        nn.init.zeros_(self.fc3.weight)

    def reset_parameters_sp(self) -> None:
        for n, p in self.named_parameters():
            if 'weight' in n:
                nn.init.xavier_normal_(p.data)
        nn.init.zeros_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
                    if 'conv' in n:
                        hidden_dim = p.shape[0]  # output channels for conv layers
                    else:
                        hidden_dim = p.shape[-1]
                    lr_value = learning_rate * (32 / hidden_dim)
                    per_layer_weight_decay_value = weight_decay * hidden_dim / 1024

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



class MLP(FlowModel):
    def __init__(self, n_classes=10, width=32, n_blocks=5, nonlin=F.relu, output_mult=1.0, input_mult=1.0, parametrization="mup"):
        super().__init__()

        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.parametrization = parametrization

        self.n_blocks = n_blocks
        
        blocks = []
        for i in range(self.n_blocks):
            if i == 0:
                blocks.append(nn.Sequential(
                    nn.Linear(784, width, bias=False),
                    nn.SiLU(),
                ))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(width, width, bias=False),
                    nn.SiLU(),
                ))
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Linear(width, n_classes, bias=False)
        if self.parametrization == "mup":
            self.reset_parameters_mup()
        elif self.parametrization == "sp":
            self.reset_parameters_sp()
        else:
            raise ValueError(f"Invalid parametrization: {self.parametrization}")

    def reset_parameters_mup(self, base_std=0.02) -> None:
        # init all weights with fan_in / 1024 * base_std
        for n, p in self.named_parameters():
            skip_list = ["final"]
            if not any([s in n for s in skip_list]):
                fan_in = p.shape[1]
                p.data.normal_(mean=0.0, std=base_std * (fan_in) ** -0.5)
        # init final layer with zeros
        nn.init.zeros_(self.final.weight)

    def reset_parameters_sp(self) -> None:
        for n, p in self.named_parameters():
            nn.init.xavier_normal_(p.data)
        nn.init.zeros_(self.final.weight)

    def forward(self, X):
        X = self.blocks[0](X) 
        X = X.view(X.shape[0], -1)
        for block in self.blocks[1:]:
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

