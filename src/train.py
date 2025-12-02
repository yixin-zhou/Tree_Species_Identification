import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import pyrootutils
import logging
import hydra
import wandb
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import DictConfig, OmegaConf
from Utils.utils import seed, get_divice 
from datasets.treeai_swiss_dataset import TreeAISwissDataset

from model.MultiModalGatedFusion import SimpleCNN
from model.loss import FocalLoss


# Set the basic config for looger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log")
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"The root path of this project is {root_path}")


# Main train function
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.wandb.use:
        default_name = cfg.wandb.name
        user_input = input(
        f"\n WandB run name will be: '{default_name}'\n"
        f"Press Enter to confirm, or input a new name: "
        ).strip()

        final_run_name = user_input if user_input != "" else default_name

        wandb.login(key=cfg.wandb.key)
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=final_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        logger.info(f"Successfully Log in Wandb: {cfg.wandb.entity}-{cfg.wandb.project}-{final_run_name}")

    logger.info("Load config from configs/config.yaml")
    logger.info(cfg)

    seed(cfg.seed) # Fix the seed to ensure reproductivity
    
    device = get_divice() if cfg.device == "auto" else torch.device(cfg.device)
    
    # Load configs of dataset
    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)

    train_dataset = TreeAISwissDataset(split="train", **dataset_kwargs)
    val_dataset   = TreeAISwissDataset(split="val", **dataset_kwargs)

    # Decide whether to use validation
    if cfg.train.validation:
        final_train_dataset = train_dataset
        final_val_dataset = val_dataset
    else:
        final_train_dataset = ConcatDataset([train_dataset, val_dataset])
        final_val_dataset = None

    train_loader = DataLoader(
        final_train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    if final_val_dataset is not None:
        val_loader = DataLoader(
            final_val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=True
        )

    model = SimpleCNN(num_classes=cfg.model.num_classes).to(device) # Just for Test, need to be replaced later
    criterion = FocalLoss(gamma=2.0) # Just for Test, need to be replaced later
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    
    

if __name__ == "__main__":
    main()