from torch.utils.data import Dataset, DataLoader
from datasets.treeai_swiss_dataset import TreeAISwissDataset
import time
from tqdm import tqdm

train_dataset = TreeAISwissDataset(dataset_folder='data/TreeAI_Swiss_60/splits', split='train')

dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )

start_time = time.time()

for i, data in tqdm(enumerate(dataloader)):
    current_batch_size = data[0]['vhm'].size(0)

end_time = time.time()
