import hydra
from omegaconf import DictConfig, OmegaConf

import torch 
from tqdm import tqdm

from slfm.util import linear_decay_lr, warmup_cooldown_lr

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
    import seaborn as sns  
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
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
    # Apply smoothing to the evaluation loss for smoother curves
    logs_df = logs_df.sort_values(['width', 'log2lr'])
    logs_df['eval_loss_smooth'] = logs_df.groupby('width')['eval_loss'].transform(
        lambda x: x.rolling(window=5, min_periods=1, center=True).mean()
    )
    # Use the smoothed values for plotting, but keep the original for reference
    logs_df['eval_loss_original'] = logs_df['eval_loss']
    logs_df['eval_loss'] = logs_df['eval_loss_smooth']

    # Apply smoothing to the evaluation accuracy for smoother curves, similar to loss
    logs_df['eval_accuracy_smooth'] = logs_df.groupby('width')['eval_accuracy'].transform(
        lambda x: x.rolling(window=5, min_periods=1, center=True).mean()
    )
    # Use the smoothed values for plotting, but keep the original for reference
    logs_df['eval_accuracy_original'] = logs_df['eval_accuracy']
    logs_df['eval_accuracy'] = logs_df['eval_accuracy_smooth']
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Define a color palette - using a sequential color map for width progression
    palette = sns.color_palette("viridis", n_colors=len(logs_df['width'].unique()))
    
    # First subplot for eval_loss
    sns.lineplot(x='log2lr', y='eval_loss', hue='width', data=logs_df[(logs_df['model_type']=='MLP')&
                                                                   (logs_df['eval_loss']>0)&
                                                                   (logs_df['epoch']==0)], 
                palette=palette, ax=ax1)
    # Set y-axis to log scale for the loss plot
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.set_title('Evaluation Loss')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 2)
    # Second subplot for eval_accuracy
    sns.lineplot(x='log2lr', y='eval_accuracy', hue='width', data=logs_df[(logs_df['model_type']=='MLP')&
                                                                   (logs_df['epoch']==0)], 
                palette=palette, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_title('Evaluation Accuracy')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(80, 100)
    
    # Add legends with better formatting
    #ax1.legend(title='Width', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Remove the default legend from the first subplot
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    
    # Create a custom colorbar for width with log scale
    # Only use powers of 2 for the colorbar
    max_width = max(logs_df['width'])
    # Define power-of-2 ticks
    nice_widths = [2**i for i in range(0, max_width.bit_length())]  # Covers up to max_width
    log_norm = LogNorm(vmin=min(nice_widths), vmax=max(nice_widths))

    # Create figure and axes
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=log_norm)
    sm.set_array([])

    # Create colorbar
    cbar = fig.colorbar(sm, ax=ax2, label='Width', orientation='vertical', pad=0.01)

    # Set colorbar ticks and labels at powers of 2
    cbar.set_ticks(nice_widths)
    cbar.set_ticklabels([f"$2^{int(np.log2(tick))}$" for tick in nice_widths])

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(cfg.sweep.save_file, dpi=300)

    # Show the plot (optional)
    plt.show()

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def sweep_train_and_evaluate(cfg: DictConfig) -> None:
    sweep_train(cfg)
    sweep_evaluate(cfg)

if __name__ == "__main__":
    sweep_train_and_evaluate()
