# src/clean.py
import os
import shutil
import argparse

# Default folders to clean (relative to project root)
FOLDERS = {
    "preprocessing": "data/preprocessing",
    "features": "data/features",
    "model": "data/model",
    "dashboard": "data/dashboard"
}

def clean_folders(folders_to_clean):
    print("The following folders will be emptied:")
    for folder in folders_to_clean:
        print(f" - {folder}")
    confirm = input("Are you sure? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    for folder in folders_to_clean:
        if os.path.exists(folder):
            # Remove all contents
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        print(f"Emptied {folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean project data folders safely")
    parser.add_argument(
        "--preprocessing", action="store_true", help="Clean data/preprocessing"
    )
    parser.add_argument(
        "--features", action="store_true", help="Clean data/features"
    )
    parser.add_argument(
        "--model", action="store_true", help="Clean data/model"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Clean data/dashboard"
    )
    args = parser.parse_args()

    # Determine which folders to clean
    selected_folders = []
    if args.preprocessing:
        selected_folders.append(FOLDERS["preprocessing"])
    if args.features:
        selected_folders.append(FOLDERS["features"])
    if args.model:
        selected_folders.append(FOLDERS["model"])
    if args.dashboard:
        selected_folders.append(FOLDERS["dashboard"])
    if not selected_folders:
        # If none specified, clean all default folders
        selected_folders = list(FOLDERS.values())

    clean_folders(selected_folders)
