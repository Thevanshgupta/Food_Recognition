import os
import shutil

# Paths
base_dir = "dataset/food-101/images"
train_txt = "dataset/food-101/meta/train.txt"
val_txt = "dataset/food-101/meta/test.txt"

# Destination
train_dir = "dataset/food101/train"
val_dir = "dataset/food101/val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def copy_images(txt_file, split_dir):
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        label, img_name = line.split('/')
        label_dir = os.path.join(split_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join(base_dir, f"{line}.jpg")
        dst = os.path.join(label_dir, f"{img_name}.jpg")
        if not os.path.exists(dst):
            shutil.copy(src, dst)

print("Preparing train split...")
copy_images(train_txt, train_dir)

print("Preparing val split...")
copy_images(val_txt, val_dir)

print("Dataset prepared successfully!")
