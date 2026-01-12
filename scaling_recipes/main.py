import hydra
from omegaconf import DictConfig, OmegaConf

import torch 
from tqdm import tqdm

from scaling_recipes.util import linear_decay_lr, warmup_cooldown_lr

def compute_validation_loss(model, val_data, loss_fn):
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    # Compute validation accuracy
    correct = 0
    total = 0
    val_loss = 0.0
    for x_val, y_val in val_data:
        if torch.cuda.is_available():
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        outputs = model(x_val)
        _, predicted = torch.max(outputs.data, 1)
        total += y_val.size(0)
        correct += (predicted == y_val).sum().item()
        val_loss += torch.nn.functional.cross_entropy(outputs, y_val).item()
    val_accuracy = 100 * correct / total
    val_loss /= len(val_data)
    model.train()
    return val_loss, val_accuracy

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def train(cfg: DictConfig) -> None:
    import math
    import gc

    model = hydra.utils.instantiate(cfg.model, width=cfg.model.width, parametrization=cfg.model.parametrization)

    if cfg.model.parametrization == "mup":
        optimizer, optimizer_settings = model.configure_optimizers(
            weight_decay=cfg.trainer.optimizer.weight_decay,
            learning_rate=cfg.trainer.optimizer.lr,
            betas=(1 - (cfg.data.size/5e5) * (1 - 0.9), 1 - (cfg.data.size/5e5) * (1 - 0.95)),
        )
    elif cfg.model.parametrization == "sp":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
        optimizer_settings = {}
    else:
        raise ValueError(f"Invalid parametrization: {cfg.model.parametrization}")

    # Learning rate scheduler
    if cfg.trainer.optimizer.lr_scheduler == "linear_decay":
        get_lr = lambda step: linear_decay_lr(step, cfg.trainer.max_epochs, cfg.trainer.optimizer.lr)
    else:  # warmup_cooldown
        warmup_iters = cfg.trainer.max_epochs // 10
        warmdown_iters = cfg.trainer.max_epochs - warmup_iters
        get_lr = lambda step: warmup_cooldown_lr(
            step, cfg.trainer.max_epochs, cfg.trainer.optimizer.lr, warmup_iters, warmdown_iters
        )
                    
    logger = hydra.utils.instantiate(cfg.logger)

    dataset = hydra.utils.instantiate(cfg.data, size = cfg.data.size)            
    loss_fn = hydra.utils.instantiate(cfg.trainer.loss)
    train_data = dataset.create(type="train")
    val_data = dataset.create(type="val")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in tqdm(range(cfg.trainer.max_epochs), desc="Training"):
        
        for it, (x, y) in enumerate(train_data):
            model.zero_grad()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            loss = loss_fn(model, x, y)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.optimizer.gradient_clip)

            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()
        
        if epoch % cfg.logger.log_interval_epochs == 0:
            logger.log_train_loss(epoch+1, loss.item())
            # Compute validation accuracy
            val_loss, val_accuracy = compute_validation_loss(model, val_data, loss_fn)
            
            print(f"Validation: Width: {cfg.model.width} | Epoch: {epoch} | Loss: {val_loss} | Accuracy: {val_accuracy}")
            logger.log_val_loss(epoch+1, val_loss, val_accuracy)
            
        # Switch back to training mode
        if math.isnan(loss.item()):
            break   
    
    # Save the model and optimizer state
    logger.save_model(model, optimizer)
    print(cfg.model.width, 2**cfg.trainer.optimizer.lr, loss.item())
    # Free up memory after each iteration
    del model  # Delete the model
    torch.cuda.empty_cache()  # Free up GPU memory
    gc.collect()  # Free up CPU memory


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def evaluate(cfg: DictConfig) -> torch.Tensor:
    model = hydra.utils.instantiate(cfg.model)
    logger = hydra.utils.instantiate(cfg.logger)
    dataset = hydra.utils.instantiate(cfg.data, size = cfg.data.size)    
    loss_fn = hydra.utils.instantiate(cfg.trainer.loss)
    
    model = logger.load_model(model)
    model.eval()

    test_data = dataset.create(type="test")

    # Evaluate the model on the test dataset
    loss, accuracy = compute_validation_loss(model, test_data, loss_fn)
    print(f"Test: Width: {cfg.model.width} | Loss: {loss} | Accuracy: {accuracy}")
    logger.log_val_loss("Test", loss, accuracy)

    return loss, accuracy

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def train_and_evaluate(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train(cfg)
    evaluate(cfg)



@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_train(cfg: DictConfig) -> None:
    # Initialize variables from the config
    
    import numpy as np
    for width in cfg.sweep.widths: #, 64, 128, 256, 512]:
        for log2lr in np.linspace(cfg.sweep.lr_range[0], cfg.sweep.lr_range[1], cfg.sweep.lr_intervals):
            cfg.trainer.optimizer.lr = float(f"{2**log2lr:.5f}")
            cfg.model.width = int(width)
            train(cfg)

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_evaluate(cfg: DictConfig) -> None:
    import numpy as np
    import pandas as pd 
    from scaling_recipes.util import sweep_plot
    logs = []

    for width in cfg.sweep.widths: #, 64, 128, 256, 512]:
        for log2lr in np.linspace(cfg.sweep.lr_range[0], cfg.sweep.lr_range[1], cfg.sweep.lr_intervals):
            cfg.trainer.optimizer.lr = float(f"{2**log2lr:.5f}")
            cfg.model.width = int(width)
            print("Evaluating model width: {} and with lr: {}".format(width, cfg.trainer.optimizer.lr))
            loss, accuracy = evaluate(cfg)

            logs.append(dict(
                epoch=0,
                model_type='MLP',
                log2lr=2**log2lr,
                eval_loss=loss,
                eval_accuracy=accuracy,
                width=cfg.model.width,
            ))
        

    logs_df = pd.DataFrame(logs)
    sweep_plot(logs_df, cfg)

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_train_and_evaluate(cfg: DictConfig) -> None:
    sweep_train(cfg)
    sweep_evaluate(cfg)


# ============================================================================
# Flow Matching Functions
# ============================================================================

def compute_flow_validation_loss(model, val_data, loss_fn):
    """Compute validation loss for flow matching model."""
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    val_loss = 0.0
    for x_val, y_val in val_data:
        if torch.cuda.is_available():
            x_val = x_val.cuda()
        loss = loss_fn(model, x_val)
        val_loss += loss.item()

    val_loss /= len(val_data)
    model.train()
    return val_loss


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def train_flow(cfg: DictConfig) -> None:
    """Train a flow matching model."""
    import math
    import gc

    model = hydra.utils.instantiate(cfg.flow_model, width=cfg.flow_model.width, parametrization=cfg.flow_model.parametrization)

    if cfg.flow_model.parametrization == "mup":
        optimizer, optimizer_settings = model.configure_optimizers(
            weight_decay=cfg.flow_trainer.optimizer.weight_decay,
            learning_rate=cfg.flow_trainer.optimizer.lr,
            betas=(1 - (cfg.data.size/5e5) * (1 - 0.9), 1 - (cfg.data.size/5e5) * (1 - 0.95)),
        )
    elif cfg.flow_model.parametrization == "sp":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.flow_trainer.optimizer.lr, weight_decay=cfg.flow_trainer.optimizer.weight_decay)
        optimizer_settings = {}
    else:
        raise ValueError(f"Invalid parametrization: {cfg.flow_model.parametrization}")

    # Learning rate scheduler
    if cfg.flow_trainer.optimizer.lr_scheduler == "linear_decay":
        get_lr = lambda step: linear_decay_lr(step, cfg.flow_trainer.max_epochs, cfg.flow_trainer.optimizer.lr)
    else:  # warmup_cooldown
        warmup_iters = cfg.flow_trainer.max_epochs // 10
        warmdown_iters = cfg.flow_trainer.max_epochs - warmup_iters
        get_lr = lambda step: warmup_cooldown_lr(
            step, cfg.flow_trainer.max_epochs, cfg.flow_trainer.optimizer.lr, warmup_iters, warmdown_iters
        )

    logger = hydra.utils.instantiate(cfg.flow_logger)

    dataset = hydra.utils.instantiate(cfg.data, size=cfg.data.size)
    loss_fn = hydra.utils.instantiate(cfg.flow_trainer.loss)
    train_data = dataset.create(type="train")
    val_data = dataset.create(type="val")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        model.cuda()

    for epoch in tqdm(range(cfg.flow_trainer.max_epochs), desc="Training Flow Model"):

        for it, (x, y) in enumerate(train_data):
            model.zero_grad()
            if torch.cuda.is_available():
                x = x.cuda()
            loss = loss_fn(model, x)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.flow_trainer.optimizer.gradient_clip)

            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()

        if epoch % cfg.flow_logger.log_interval_epochs == 0:
            logger.log_train_loss(epoch+1, loss.item())
            # Compute validation loss
            val_loss = compute_flow_validation_loss(model, val_data, loss_fn)

            print(f"Flow Validation: Width: {cfg.flow_model.width} | Epoch: {epoch} | Loss: {val_loss}")
            logger.log_val_loss(epoch+1, val_loss, 0.0)  # No accuracy for flow matching

        # Switch back to training mode
        if math.isnan(loss.item()):
            break

    # Save the model and optimizer state
    logger.save_model(model, optimizer)
    print(cfg.flow_model.width, 2**cfg.flow_trainer.optimizer.lr, loss.item())
    # Free up memory after each iteration
    del model
    torch.cuda.empty_cache()
    gc.collect()


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def evaluate_flow(cfg: DictConfig) -> torch.Tensor:
    """Evaluate a flow matching model."""
    model = hydra.utils.instantiate(cfg.flow_model, width=cfg.flow_model.width, parametrization=cfg.flow_model.parametrization)
    logger = hydra.utils.instantiate(cfg.flow_logger)
    dataset = hydra.utils.instantiate(cfg.data, size=cfg.data.size)
    loss_fn = hydra.utils.instantiate(cfg.flow_trainer.loss)

    model = logger.load_model(model)
    model.eval()

    test_data = dataset.create(type="test")

    # Evaluate the model on the test dataset
    loss = compute_flow_validation_loss(model, test_data, loss_fn)
    print(f"Flow Test: Width: {cfg.flow_model.width} | Loss: {loss}")
    logger.log_val_loss("Test", loss, 0.0)

    return loss


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_train_flow(cfg: DictConfig) -> None:
    """Sweep training for flow matching across widths and learning rates."""
    import numpy as np
    for width in cfg.flow_sweep.widths:
        for log2lr in np.linspace(cfg.flow_sweep.lr_range[0], cfg.flow_sweep.lr_range[1], cfg.flow_sweep.lr_intervals):
            cfg.flow_trainer.optimizer.lr = float(f"{2**log2lr:.5f}")
            cfg.flow_model.width = int(width)
            train_flow(cfg)


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_evaluate_flow(cfg: DictConfig) -> None:
    """Evaluate flow matching sweep and generate plots."""
    import numpy as np
    import pandas as pd
    from scaling_recipes.util import flow_sweep_plot
    logs = []

    for width in cfg.flow_sweep.widths:
        for log2lr in np.linspace(cfg.flow_sweep.lr_range[0], cfg.flow_sweep.lr_range[1], cfg.flow_sweep.lr_intervals):
            cfg.flow_trainer.optimizer.lr = float(f"{2**log2lr:.5f}")
            cfg.flow_model.width = int(width)
            print("Evaluating flow model width: {} and with lr: {}".format(width, cfg.flow_trainer.optimizer.lr))
            loss = evaluate_flow(cfg)

            logs.append(dict(
                epoch=0,
                model_type='FlowMLP',
                log2lr=2**log2lr,
                eval_loss=loss,
                width=cfg.flow_model.width,
            ))

    logs_df = pd.DataFrame(logs)
    flow_sweep_plot(logs_df, cfg)


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_train_and_evaluate_flow(cfg: DictConfig) -> None:
    """Complete sweep: train and evaluate flow matching models."""
    sweep_train_flow(cfg)
    sweep_evaluate_flow(cfg)


if __name__ == "__main__":
    sweep_train_and_evaluate()
