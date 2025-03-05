import hydra
from omegaconf import DictConfig, OmegaConf

import torch 
from tqdm import tqdm

from slfm.util import linear_decay_lr, warmup_cooldown_lr

def compute_validation_loss(model, val_data, loss_fn):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_data:
            val_loss += loss_fn(model, x_val, y_val).item()
    # Compute validation accuracy
    correct = 0
    total = 0
    val_loss = 0.0
    for x_val, y_val in val_data:
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

    model = hydra.utils.instantiate(cfg.model, width=cfg.model.width)

    optimizer, optimizer_settings = model.configure_optimizers(
        weight_decay=cfg.trainer.optimizer.weight_decay,
        learning_rate=cfg.trainer.optimizer.lr,
        betas=(1 - (cfg.data.size/5e5) * (1 - 0.9), 1 - (cfg.data.size/5e5) * (1 - 0.95)),
    )

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
    for epoch in tqdm(range(cfg.trainer.max_epochs), desc="Training"):
        
        for it, (x, y) in enumerate(train_data):
            model.zero_grad()
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
def mup_coord_check(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    import numpy as np
    from mup import get_shapes, make_base_shapes, set_base_shapes
    from slfm.coord_check import get_coord_data, plot_coord_data

    ### This is for making the base shapes
    model = hydra.utils.instantiate(cfg.model)
    scaled_model = hydra.utils.instantiate(cfg.model, width=cfg.model.width*2)
    
    base_shapes = get_shapes(model)
    delta_shapes = get_shapes(scaled_model)
                    
    make_base_shapes(base_shapes, delta_shapes, savefile=cfg.mup.save_base_shapes)

    # Use the base shaped here
    mup = cfg.mup.switch
    def gen(w, standparam=False):
        def f():
            print("Creating model with width: ", w)
            model = hydra.utils.instantiate(cfg.model, width=w)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert cfg.mup.save_base_shapes, 'load_base_shapes needs to be nonempty'
                set_base_shapes(model, cfg.mup.save_base_shapes)
            return model
        return f

    widths = 2**np.arange(2, 5)
    widths = np.array([10, 50, 100, 150, 200])
    models = {w: gen(w, standparam=not mup) for w in widths}
    loss_fn = hydra.utils.instantiate(cfg.trainer.loss)
    
    # make a dataloader with small batch size/seq len
    dataset = hydra.utils.instantiate(cfg.data, size = cfg.data.size)    
    dataloader = dataset.create()
    # record data from the model activations over a few steps of training
    # this returns a pandas dataframe
    df = get_coord_data(loss_fn, models, dataloader, cuda=False, dict_in_out=True, nsteps=4)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # This saves the coord check plots to filename.
    plot_coord_data(df, save_to=cfg.mup.coord_check_filename, legend=True)


@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def mup_train(cfg: DictConfig) -> None:
    # Initialize variables from the config
    
    import numpy as np
    for width in [10, 32, 64, 128, 256, 512][::-1]:
        for log2lr in np.linspace(-8, -1, 10):
            cfg.trainer.optimizer.lr = float(f"{2**log2lr:.3f}")
            cfg.model.width = int(width)
            train(cfg)

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def mup_evaluate(cfg: DictConfig) -> None:
    # Initialize variables from the config
    
    import numpy as np
    logs = []

    for width in [10, 32]: #, 64, 128, 256, 512]:
        for log2lr in np.linspace(-6, -1, 10):
            cfg.trainer.optimizer.lr = float(f"{2**log2lr:.3f}")
            cfg.model.width = int(width)
            print("Evaluating model width: {} and with lr: {}".format(width, cfg.trainer.optimizer.lr))
            loss = evaluate(cfg)

            logs.append(dict(
                epoch=0,
                model_type='MLP',
                log2lr=log2lr,
                eval_loss=loss.item(),
                width=cfg.model.width,
            ))
        
    import pandas as pd 
    import seaborn as sns  
    import matplotlib.pyplot as plt

    logs_df = pd.DataFrame(logs)
    sns.lineplot(x='log2lr', y='eval_loss', hue='width', data=logs_df[(logs_df['model_type']=='MLP')&
                                                                   (logs_df['eval_loss']<5)&
                                                                   (logs_df['epoch']==0)])
    plt.savefig("u_mup_eval_loss.png")  # You can specify other formats such as 'pdf', 'svg', etc.

    # Show the plot (optional)
    plt.show()

