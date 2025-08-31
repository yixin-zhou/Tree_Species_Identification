import requests
from tqdm import tqdm
from pathlib import Path


def download_dataset(dataset_name, url, save_path, data_size=None, chunk_size=1024 * 1024):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url=url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    total_size = int(data_size * 1024 * 1024 * 1024) if (total_size == 0 and data_size is not None) else total_size
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f" downloading {dataset_name}"
    )

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()


if __name__ == '__main__':
    quebec_tree_url = 'https://zenodo.org/api/records/8148479/files-archive'
    download_dataset(dataset_name='Quebec Trees Dataset', url=quebec_tree_url, data_size=149.3,
                     save_path='../data/Quebec_Tree.zip')


