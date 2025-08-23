import argparse, os, random, shutil
from pathlib import Path

def copy_subset(files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    random.seed(42)

    for cls_dir in [d for d in src.iterdir() if d.is_dir()]:
        cls_name = cls_dir.name
        files = [str(p) for p in cls_dir.glob("*") if p.is_file()]
        random.shuffle(files)

        n = len(files)
        n_val = int(n * args.val)
        n_test = int(n * args.test)
        n_train = n - n_val - n_test

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:],
        }
        for split, flist in splits.items():
            out_dir = dst / split / cls_name
            copy_subset(flist, out_dir)

    print(f"Done. Train/val/test created under: {dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder with class subfolders (ImageFolder style).")
    parser.add_argument("--dst", required=True, help="Destination root to create train/val/test structure.")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
