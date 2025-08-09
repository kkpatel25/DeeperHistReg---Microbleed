##### External Imports #####
import os
import pathlib
import urllib.request
from tqdm import tqdm
import gdown


# Path to the files directory relative to this script
FILES_DIR = pathlib.Path(__file__).parent / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, filename: str) -> pathlib.Path:
    """
    Download a file from `url` into the arvind/files directory if it doesn't exist.

    Args:
        url (str): The URL to download.
        filename (str): The name to save the file as.

    Returns:
        pathlib.Path: Path to the downloaded file.
    """
    file_path = FILES_DIR / filename
    if file_path.exists():
        return file_path

    print(f"[INFO] Downloading {filename} from {url}...")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.getheader('Content-Length', 0))
        with open(file_path, "wb") as out_file, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

    print(f"[INFO] Download complete: {file_path}")
    return file_path


def download_file_gd(url:str, filename: str):
    """
    Download a file from Google Drive using its file ID.

    Args:
        url (str): The Google Drive url to download.
        dest_name (str): Dest Name.
    """
    file_path = FILES_DIR / filename

    if not os.path.exists(file_path):
        print(f"[INFO] Downloading {filename} from {url}...")
        gdown.download(url, str(file_path), quiet=False)
