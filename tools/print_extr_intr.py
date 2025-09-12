from re import A
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

mtq = lambda m: R.from_matrix(m).as_quat()[[3, 0, 1, 2]].tolist()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=int, required=True)
    return parser.parse_args()


def main(args):
    data = np.load(f"{args.serial}.npy", allow_pickle=True).item()
    intr = data["intrinsics"]
    extr = data["extrinsics"]
    print(f"Camera serial number: {args.serial}")
    print(f"fx: {intr[0, 0]}")
    print(f"fy: {intr[1, 1]}")
    print(f"cx: {intr[0, 2]}")
    print(f"cy: {intr[1, 2]}")

    print(f"position: {extr[:3, 3].tolist()}")
    print(f"orientation: {mtq(extr[:3, :3])}")


if __name__ == "__main__":
    args = get_args()
    main(args)
