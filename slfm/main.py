import hydra
from omegaconf import DictConfig, OmegaConf

import torch 
from tqdm import tqdm

from slfm.util import linear_decay_lr, warmup_cooldown_lr

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
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in tqdm(range(cfg.trainer.max_epochs), desc="Training"):
        model.zero_grad()
        loss = loss_fn(model, dataset.create())
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.optimizer.gradient_clip)

        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
                
        if epoch % cfg.trainer.log_interval == 0:
            print("Width: {} | Epoch: {} | LR: {} | Loss: {}".format(cfg.model.width, epoch, lr, loss.item()))
            logger.log_train_loss(epoch, loss.item())
        if math.isnan(loss.item()):
            break   
    
    # Save the model and optimizer state
    logger.save_model(model, optimizer)
       
    if not math.isnan(loss):        
        print(cfg.model.width, 2**cfg.trainer.optimizer.lr, loss.item())
        logger.save_visuals(model, dataset.create())

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

    data = dataset.create()

    # Evaluate the model on the dataset
    loss = loss_fn(model, data)
    logger.log_eval_loss(0, loss)

    # Save the visualizations
    logger.save_visuals(model, data)

    return loss

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

