import json
import os

invalid_folders = json.load(open('invalid_folders.json'))
print(f"Found {len(invalid_folders)} invalid folders")

for folder in invalid_folders:
    os.system(f"rm -rf {folder}")