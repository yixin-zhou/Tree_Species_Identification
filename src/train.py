import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import logging
import hydra
from hydra.utils import to_absolute_path
import wandb
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import DictConfig, OmegaConf
from Utils.utils import seed, get_divice 
from datasets.treeai_swiss_dataset import TreeAISwissDataset

from model.TreeDetector import TreeDetector
from model.loss import FCOSLoss


# Set the basic config for logger
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


def detection_collate_fn(batch):
    input_list = [item[0] for item in batch]
    target_list = [item[1] for item in batch]
    
    batch_input = {}
    for key in input_list[0].keys():
        batch_input[key] = torch.stack([d[key] for d in input_list], dim=0)
        
    batch_target = target_list
    
    return batch_input, batch_target


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
            entity=cfg.wandb.get("entity", None),
            project=cfg.wandb.project,
            name=final_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        logger.info(f"Successfully Log in Wandb: {cfg.wandb.entity}-{cfg.wandb.project}-{final_run_name}")

    logger.info("Load config from configs/config.yaml")

    seed(cfg.seed)
    
    device = get_divice() if cfg.device == "auto" else torch.device(cfg.device)

    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if "folder" in dataset_kwargs:
        dataset_kwargs["folder"] = to_absolute_path(dataset_kwargs["folder"])

    train_dataset = TreeAISwissDataset(split="train", **dataset_kwargs)
    val_dataset   = TreeAISwissDataset(split="val", **dataset_kwargs)

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
        drop_last=True,
        collate_fn=detection_collate_fn 
    )

    val_loader = None
    if final_val_dataset is not None:
        val_loader = DataLoader(
            final_val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=detection_collate_fn
        )

    logger.info("Initializing TreeDetector...")
    model = TreeDetector(
        device=device,
        num_classes=cfg.model.num_classes,
        fusion_channels=256
    ).to(device)
    
    criterion = FCOSLoss(
        strides=[4, 8, 16],
        sparse_ignore_threshold=0.5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    from src.trainer.trainer import Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        # Make sure checkpoints are always saved under the project root,
        # not inside Hydra's per-run working directory.
        pth_savepath=to_absolute_path(cfg.paths.save_dir),
        wandb_run=wandb if cfg.wandb.use else None
    )
    
    trainer.fit(train_loader, cfg.train.epochs, val_loader)

if __name__ == "__main__":
    main()
