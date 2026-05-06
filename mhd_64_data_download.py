from huggingface_hub import hf_hub_download, HfFileSystem

fs          = HfFileSystem()
train_files = fs.glob("datasets/polymathic-ai/MHD_64/data/train/*.hdf5")
val_files   = fs.glob("datasets/polymathic-ai/MHD_64/data/valid/*.hdf5")

for path in train_files + val_files:
    filename = path.replace("datasets/polymathic-ai/MHD_64/", "")
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id   = "polymathic-ai/MHD_64",
        filename  = filename,
        repo_type = "dataset",
        local_dir = "C:/mhd_data",
    )

print("Done")