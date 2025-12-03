import pyrootutils
root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)

import torch
import logging
import hydra
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from Utils.utils import get_divice 
from datasets.treeai_swiss_dataset import TreeAISwissDataset
from model.TreeDetector import TreeDetector
from trainer.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def detection_collate_fn(batch):
    input_list = [item[0] for item in batch]
    target_list = [item[1] for item in batch]
    
    batch_input = {}
    keys = input_list[0].keys()
    for key in keys:
        batch_input[key] = torch.stack([d[key] for d in input_list], dim=0)
        
    batch_target = target_list
    return batch_input, batch_target

def move_to_device(inputs, targets, device):
    inputs_device = {k: v.to(device) for k, v in inputs.items()}
    targets_device = []
    for t in targets:
        t_device = {
            'boxes': t['boxes'].to(device),
            'labels': t['labels'].to(device)
        }
        targets_device.append(t_device)
    return inputs_device, targets_device

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if "checkpoint_path" not in cfg:
        logger.error("Please specify the checkpoint path using +checkpoint_path='...'")
        return

    ckpt_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    if "folder" in cfg.data.dataset:
        cfg.data.dataset.folder = hydra.utils.to_absolute_path(cfg.data.dataset.folder)
    
    device = get_divice() if cfg.device == "auto" else torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    logger.info("Loading Test Dataset...")
    dataset_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    test_dataset = TreeAISwissDataset(split="test", **dataset_kwargs)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    logger.info("Initializing Model...")
    model = TreeDetector(
        device=device,
        num_classes=cfg.model.num_classes,
        fusion_channels=cfg.model.get("fusion_channels", 256)
    ).to(device)

    logger.info(f"Loading checkpoint from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = Evaluator(
        device=device, 
        num_classes=cfg.model.num_classes,
        iou_threshold=0.4, 
        score_threshold=0.2 
    )

    logger.info("Starting Inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = move_to_device(inputs, targets, device)
            
            outputs = model(inputs)
            
            evaluator.process_batch(outputs, targets)

    metrics = evaluator.get_metrics()
    
    print("\n" + "="*50)
    print("             EVALUATION RESULTS             ")
    print("="*50)
    print(f"Detection Recall (Found?)     : {metrics['Recall_Total']:.2%}")
    print("-" * 50)
    print(f"Identification Acc (Micro)    : {metrics['Acc_Micro']:.2%}  <-- Biased by major classes")
    print(f"Balanced Accuracy  (Macro)    : {metrics['Acc_Macro']:.2%}  <-- REAL Performance")
    print("-" * 50)
    print("Per-Class Accuracy (on matched boxes):")
    for k, v in metrics['Class_Acc'].items():
        print(f"  {k:<15} : {v:.2%}")
    print("="*50)

    # Keep evaluation artifacts under the repo's results directory instead of Hydra's run dir
    output_dir = hydra.utils.to_absolute_path("results")
    os.makedirs(output_dir, exist_ok=True)
    cm_save_path = os.path.join(output_dir, "confusion_matrix_test.png")
    evaluator.plot_cm(metrics['Confusion_Matrix'], cm_save_path)
    logger.info(f"Confusion Matrix saved to {cm_save_path}")

if __name__ == "__main__":
    main()
