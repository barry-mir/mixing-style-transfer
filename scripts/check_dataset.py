import os
import librosa
from glob import glob
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count

dataset_path = '/ssd2/barry/fma_large_separated/'
folders = glob(os.path.join(dataset_path, '*'))
stems = ['vocals.mp3', 'bass.mp3', 'drums.mp3', 'other.mp3']

def check_folder(folder):
    for stem in stems:
        stem_path = os.path.join(folder, stem)
        if not os.path.exists(stem_path):
            print(f"Folder {folder} does not contain stem {stem}")
            return folder
        audio, _ = librosa.load(stem_path, sr=44100)
        if audio.shape[0] <= 44100 * 25:
            print(f"Folder {folder} does not contain at least 25 seconds of audio")
            return folder
    return None

if __name__ == '__main__':
    num_workers = min(16, cpu_count())
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(check_folder, folders), total=len(folders)))
    invalid_folders = [folder for folder in results if folder is not None]

    with open('invalid_folders.json', 'w') as f:
        json.dump(invalid_folders, f, indent=4)

    print(f"Found {len(invalid_folders)} invalid folders")
    print(f"Invalid folders: {invalid_folders}")