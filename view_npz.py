"""Load a saved predictions .npz and launch the viser viewer locally."""

import argparse
import numpy as np

from lingbot_map.vis import PointCloudViewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to predictions .npz saved by demo.py --save_predictions")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--conf_threshold", type=float, default=1.5)
    ap.add_argument("--downsample_factor", type=int, default=10)
    ap.add_argument("--point_size", type=float, default=0.00001)
    ap.add_argument("--image_folder", default=None,
                    help="Original image folder (only needed if you want to re-apply sky masks).")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    pred = {k: data[k] for k in data.files}
    print(f"Loaded {args.npz} with keys: {list(pred.keys())}")

    viewer = PointCloudViewer(
        pred_dict=pred,
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        mask_sky=False,
        image_folder=args.image_folder,
    )
    print(f"3D viewer at http://localhost:{args.port}")
    viewer.run()


if __name__ == "__main__":
    main()
