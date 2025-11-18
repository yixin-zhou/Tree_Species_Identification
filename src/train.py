import torch
import pyrootutils
import logging
from Utils.utils import seed
from datasets.treeai_swiss_dataset import TreeAISwissDataset


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

root_path = pyrootutils.setup_root(__file__, indicator='.git', pythonpath=True)
logger.info(f"The root path of this project is {root_path}")

# Wandb key: 78ca1f862ad9d2da640d58d1d18e1162a9358659

seed(42) # Fix the seed for all libraries

