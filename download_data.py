import kagglehub
import shutil
import os

print("Starting download (this might take a few minutes)...")
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print(f"Downloaded to: {path}")

dest = "data/HAM10000"
os.makedirs(dest, exist_ok=True)

print(f"Copying files to {dest}...")
shutil.copytree(path, dest, dirs_exist_ok=True)
print("Done copying.")
