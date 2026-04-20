import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def generate_random_binary_structure(image_size: int) -> np.ndarray:
    """Generate a random binary structure of size image_size x image_size."""
    return np.random.randint(0, 2, (image_size, image_size), dtype=np.uint8) * 255


def apply_gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to an image."""
    return gaussian_filter(image, sigma=sigma)


def binarize_image(image: np.ndarray, threshold: float = 255 / 2) -> np.ndarray:
    """Binarize an image based on a threshold."""
    return ((image > threshold).astype(np.uint8) * 255)


def save_image_as_png(array: np.ndarray, filename: Path) -> None:
    """Save the image as an 8-bit grayscale PNG."""
    Image.fromarray(array, mode="L").save(filename)


def sigma_label(sigma: float) -> str:
    sigma_value = float(sigma)
    if sigma_value.is_integer():
        return str(int(sigma_value))
    return str(sigma_value)


def generate_dataset(
    image_size: int = 128,
    sigma: float = 2,
    datasize: int = 30000,
    base_dir: str = "datasets",
    seed: Optional[int] = None,
) -> Path:
    if seed is not None:
        np.random.seed(seed)

    output_dir = Path(base_dir) / str(image_size) / f"sigma{sigma_label(sigma)}" / "struct"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(datasize):
        binary_structure = generate_random_binary_structure(image_size)
        noisy_structure = apply_gaussian_filter(binary_structure, sigma)
        binarized_structure = binarize_image(noisy_structure)
        save_image_as_png(binarized_structure, output_dir / f"{i}.png")

    sample_path = output_dir / "1.png"
    if sample_path.exists():
        with Image.open(sample_path) as img:
            width, height = img.size
            print(f"sample image size: {width}x{height}, mode={img.mode}")

    print(f"saved {datasize} images to {output_dir}")
    return output_dir


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=2)
    parser.add_argument("--datasize", type=int, default=30000)
    parser.add_argument("--base_dir", type=str, default="datasets")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = create_argparser().parse_args()
    generate_dataset(
        image_size=args.image_size,
        sigma=args.sigma,
        datasize=args.datasize,
        base_dir=args.base_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
