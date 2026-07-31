from pathlib import Path

import cv2
import numpy as np


def square_resize(img:np.ndarray, output_size:int=256, padding_val:int=0) -> np.ndarray:
    h, w, c = img.shape
    side = max(h, w)

    canvas = np.full((side, side, c), fill_value=padding_val, dtype=img.dtype)

    offset_x = (side - w) // 2
    offset_y = (side - h) // 2

    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = img

    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    img_path = "/home/dmartin/phd/projects/pose_estimation/proof_of_concept/external/PoseAnything/examples/002.jpg"

    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError("Could not read the image to resize.")

    prepared = square_resize(img, output_size=256)

    output_path = "/home/dmartin/phd/projects/pose_estimation/proof_of_concept/external/PoseAnything/examples/002_res.png"
    if not cv2.imwrite(output_path, prepared):
        raise RuntimeError("Could not write image.")
