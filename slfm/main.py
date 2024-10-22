import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def train(cfg: DictConfig) -> None:
    # Initialize variables from the config
    logger = hydra.utils.instantiate(cfg.logger)

    model = hydra.utils.instantiate(cfg.model)
    dataset = hydra.utils.instantiate(cfg.data, size = cfg.data.size)
    
    optimizer = hydra.utils.instantiate(cfg.trainer.optimizer, model.parameters(), lr=cfg.trainer.optimizer.lr)
    loss_fn = hydra.utils.instantiate(cfg.trainer.loss)
    
    model.train()
    for epoch in tqdm(range(cfg.trainer.max_epochs), desc="Training"):
        model.zero_grad()
        loss = loss_fn(model, dataset.create())
        loss.backward()
        optimizer.step()

        if epoch % cfg.trainer.log_interval == 0:
            logger.log_train_loss(epoch, loss.item())

    logger.save_model(model, optimizer)
        
@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def evaluate(cfg: DictConfig) -> None:
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

@hydra.main(
    version_base=None,
    config_name="base",
    config_path="cli/conf",
)
def train_and_evaluate(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train(cfg)
    evaluate(cfg)
