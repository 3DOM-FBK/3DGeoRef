"""
DINO-based Image Alignment Module

This module provides a class for finding the best alignment (position, rotation, scale)
between two images using DINO (DINOv2 or DINOv3) feature descriptors.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image, ImageOps

# Logger configuration
logger = logging.getLogger(__name__)


class DinoImageMatcher:
    """
    Finds the best alignment between a base image and a template image
    using DINO feature descriptors and cross-correlation.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

    # Default search parameters
    DEFAULT_SCALES = [0.5, 1.0, 1.5]
    DEFAULT_ROTATIONS = list(range(-180, 180, 90))

    def __init__(
        self,
        dino_version: str = "v2",
        downscale_factor: float = 1.0,
        scales: Optional[List[float]] = None,
        rotations: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the DinoImageMatcher.

        Args:
            dino_version: DINO model version ('v2' or 'v3').
            downscale_factor: Factor to downscale images before processing (1.0 = no scaling).
            scales: List of scales to try for the template image.
            rotations: List of rotation angles (degrees) to try for the template.
            device: Torch device ('cuda' or 'cpu'). Auto-detected if None.
        """
        if downscale_factor <= 0:
            raise ValueError("downscale_factor must be positive.")

        self.dino_version = dino_version.lower()
        self.downscale_factor = downscale_factor
        self.scales = scales if scales is not None else self.DEFAULT_SCALES
        self.rotations = rotations if rotations is not None else self.DEFAULT_ROTATIONS
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._processor = None # Only used for DINOv3
        self._model_name: str = ""
        self._patch_size: int = 14

        self._configure_model()
        self._load_model()

    def _configure_model(self):
        """Configure model name and patch size based on DINO version."""
        if self.dino_version == "v2":
            self._model_name = "dinov2_vits14"
            self._patch_size = 14
        elif self.dino_version == "v3":
            self._model_name = "facebook/dinov3-vitl16-pretrain-sat493m"
            self._patch_size = 16
        else:
            raise ValueError(f"Unknown DINO version: {self.dino_version}. Use 'v2' or 'v3'.")

    def _load_model(self):
        """Load the DINO model."""
        logger.info(f"Loading DINO model: {self._model_name} on {self.device}...")

        if self.dino_version == "v2":
            self._model = torch.hub.load(
                "facebookresearch/dinov2",
                self._model_name,
                pretrained=True,
            ).to(self.device)
            self._model.eval()
            logger.info("DINOv2 model loaded successfully.")

        elif self.dino_version == "v3":
            from transformers import AutoImageProcessor, AutoModel
            self._processor = AutoImageProcessor.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).to(self.device)
            self._model.eval()
            logger.info("DINOv3 model loaded successfully.")

    # --- Image Preprocessing ---

    def _pad_to_divisible(self, img: Image.Image) -> Image.Image:
        """Pad image (right/bottom) so dimensions are divisible by patch size."""
        w, h = img.size
        pw = (self._patch_size - w % self._patch_size) % self._patch_size
        ph = (self._patch_size - h % self._patch_size) % self._patch_size
        if pw or ph:
            img = ImageOps.expand(img, (0, 0, pw, ph))
        return img

    def _transform_image(self, img: Image.Image) -> torch.Tensor:
        """Apply standard ImageNet normalization."""
        t = transforms.ToTensor()(img)
        t = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(t)
        return t

    def _scale_image(self, img: Image.Image, scale: float) -> Image.Image:
        """Resize image by a given scale factor."""
        w, h = img.size
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)

    def _rotate_image(self, img: Image.Image, angle: float) -> Image.Image:
        """Rotate image clockwise by a given angle."""
        return img.rotate(-angle, resample=Image.BICUBIC, expand=True)

    # --- Feature Extraction ---

    def _compute_patch_mask(self, img: Image.Image) -> torch.Tensor:
        """
        Create a binary mask indicating valid (non-black) patches.
        Returns: mask tensor of shape (1, 1, H_patches, W_patches).
        """
        img = self._pad_to_divisible(img)
        arr = np.array(img)
        non_black = np.any(arr > 0, axis=2).astype(np.float32)

        H, W = non_black.shape
        Hm = H // self._patch_size
        Wm = W // self._patch_size

        non_black = non_black[:Hm * self._patch_size, :Wm * self._patch_size]
        non_black = non_black.reshape(Hm, self._patch_size, Wm, self._patch_size)
        patch_mask = (non_black.mean(axis=(1, 3)) > 0.5).astype(np.float32)
        
        return torch.from_numpy(patch_mask).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def _extract_dense_features(self, img: Image.Image) -> torch.Tensor:
        """
        Extract L2-normalized patch features from a (padded) image.
        Returns: features tensor of shape (C, H_patches, W_patches).
        """
        img_w, img_h = img.size
        H = img_h // self._patch_size
        W = img_w // self._patch_size

        x = self._transform_image(img).unsqueeze(0).to(self.device)

        if self.dino_version == "v2":
            feats = self._model.forward_features(x)["x_norm_patchtokens"]
            feats = feats.squeeze(0)
            feats = F.normalize(feats, dim=1, p=2)

        elif self.dino_version == "v3":
            outputs = self._model(pixel_values=x)
            H = x.shape[-2] // self._patch_size
            W = x.shape[-1] // self._patch_size
            expected_patches = H * W

            all_tokens = outputs.last_hidden_state.squeeze(0)
            feats = all_tokens[1:1 + expected_patches]
            
            if feats.shape[0] != expected_patches:
                logger.warning(f"Expected {expected_patches} patches, got {feats.shape[0]}. Adjusting.")
                feats = all_tokens[-expected_patches:]
            
            feats = F.normalize(feats, dim=1, p=2)

        feats = feats.reshape(H, W, -1).permute(2, 0, 1)
        return feats.cpu()

    # --- Correlation ---

    def _create_gaussian_weights(self, H: int, W: int, sigma_factor: float = 0.3) -> torch.Tensor:
        """Create 2D Gaussian weight map for center-weighted correlation."""
        y = torch.arange(0, H, dtype=torch.float32)
        x = torch.arange(0, W, dtype=torch.float32)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        sigma_y = sigma_factor * H
        sigma_x = sigma_factor * W
        weights = torch.exp(-((yy - cy)**2 / (2 * sigma_y**2) + (xx - cx)**2 / (2 * sigma_x**2)))
        weights = (weights / weights.max()) * 0.9 + 0.1
        return weights.unsqueeze(0).unsqueeze(0)

    def _masked_correlate(
        self,
        feats1: torch.Tensor,
        feats2: torch.Tensor,
        mask2: torch.Tensor,
        use_gaussian_weighting: bool = True,
    ) -> torch.Tensor:
        """Compute normalized cross-correlation between feature maps."""
        feats1 = feats1.unsqueeze(0)
        feats2 = feats2.unsqueeze(0)
        H2, W2 = feats2.shape[2], feats2.shape[3]

        if use_gaussian_weighting:
            gaussian_weights = self._create_gaussian_weights(H2, W2).to(feats2.device)
            combined_weights = mask2 * gaussian_weights
        else:
            combined_weights = mask2

        weights_expanded = combined_weights.expand(-1, feats2.shape[1], -1, -1)
        feats2_weighted = feats2 * weights_expanded
        corr = F.conv2d(feats1, feats2_weighted)

        ones = torch.ones_like(mask2)
        sum_weights = F.conv2d(ones, combined_weights)
        corr = corr / (sum_weights + 1e-6)
        
        return corr.squeeze(0).squeeze(0)

    # --- Visualization ---

    def _save_overlay(
        self, base_img: Image.Image, tmpl_img: Image.Image, row: int, col: int, out_path: str
    ):
        """Save an image overlay showing the matched template on the base image."""
        base = np.array(base_img)
        tmpl = np.array(tmpl_img)

        r1 = min(row + tmpl.shape[0], base.shape[0])
        c1 = min(col + tmpl.shape[1], base.shape[1])

        blended = cv2.addWeighted(base[row:r1, col:c1], 0.5, tmpl[:r1 - row, :c1 - col], 0.5, 0)
        base[row:r1, col:c1] = blended
        cv2.rectangle(base, (col, row), (c1, r1), (0, 255, 0), 3)
        cv2.imwrite(out_path, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

    # --- Main Matching Logic ---

    def match(
        self,
        base_image_path: str,
        template_image_path: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Find the best alignment of the template image within the base image.

        Args:
            base_image_path: Path to the base (larger) image.
            template_image_path: Path to the template image to be located.
            output_dir: Optional directory to save visualizations and results.

        Returns:
            A tuple (center_row, center_col) representing the center position
            of the matched template in the original base image's pixel coordinates.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        img1 = Image.open(base_image_path).convert("RGB")
        img2 = Image.open(template_image_path).convert("RGB")

        if self.downscale_factor != 1.0:
            logger.info(f"Downscaling images by factor {self.downscale_factor}...")
            img1 = self._scale_image(img1, self.downscale_factor)
            img2 = self._scale_image(img2, self.downscale_factor)

        img1_pad = self._pad_to_divisible(img1)
        feats1 = self._extract_dense_features(img1_pad)

        best: Dict[str, Any] = {"score": -1e9}

        for rot in self.rotations:
            for scale in self.scales:
                img2r = self._rotate_image(img2, rot)
                img2rs = self._scale_image(img2r, scale)
                img2p = self._pad_to_divisible(img2rs)

                feats2 = self._extract_dense_features(img2p)
                mask2 = self._compute_patch_mask(img2p)

                if mask2.shape[-2:] != feats2.shape[-2:]:
                    mask2 = F.interpolate(mask2.float(), size=feats2.shape[-2:], mode='nearest')

                if feats2.shape[1] > feats1.shape[1] or feats2.shape[2] > feats1.shape[2]:
                    continue

                corr = self._masked_correlate(feats1, feats2, mask2.to(feats1.device))
                corr_np = corr.numpy()
                idx = corr_np.argmax()
                Hc, Wc = corr_np.shape
                r, c = divmod(idx, Wc)
                score = corr_np[r, c]

                if score > best["score"]:
                    best.update(
                        score=score, rotation=rot, scale=scale,
                        row=r, col=c, img2_transformed=img2p, corr=corr_np
                    )

        # Calculate pixel coordinates from patch indices
        pr = best["row"] * self._patch_size
        pc = best["col"] * self._patch_size
        
        img2_h, img2_w = np.array(best["img2_transformed"]).shape[:2]

        # Save visualizations if output directory is provided
        if output_dir:
            best["img2_transformed"].save(os.path.join(output_dir, "template_transformed.png"))
            self._save_overlay(img1_pad, best["img2_transformed"], int(pr), int(pc),
                               os.path.join(output_dir, "overlay.png"))
            
            corr = best["corr"]
            corr_vis = ((corr - corr.min()) / (np.ptp(corr) + 1e-6) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "correlation_map.png"),
                        cv2.applyColorMap(corr_vis, cv2.COLORMAP_JET))

        # Scale back to original image coordinates if downscaling was applied
        if self.downscale_factor != 1.0:
            pr /= self.downscale_factor
            pc /= self.downscale_factor
            img2_h /= self.downscale_factor
            img2_w /= self.downscale_factor

        center_row = pr + img2_h / 2.0
        center_col = pc + img2_w / 2.0

        logger.info(f"Best Match Found:")
        logger.info(f"  Center (row, col): ({center_row:.2f}, {center_col:.2f}) px")
        logger.info(f"  Rotation: {best['rotation']}°, Scale: {best['scale']}, Score: {best['score']:.4f}")

        if output_dir:
            results_file = os.path.join(output_dir, "results.txt")
            with open(results_file, 'w') as f:
                f.write(f"{center_row},{center_col},{best['rotation']},{best['scale']},{best['score']}\n")
            logger.info(f"  Results saved to: {results_file}")

        return center_row, center_col

    @staticmethod
    def list_images(directory: str) -> List[str]:
        """Find all image files in a directory."""
        images = []
        if not os.path.isdir(directory):
            return images
        for name in os.listdir(directory):
            ext = os.path.splitext(name)[1].lower()
            if ext in DinoImageMatcher.IMAGE_EXTENSIONS:
                images.append(os.path.join(directory, name))
        return sorted(images)


class OrthoCropper:
    """
    Crops an orthophoto (GeoTIFF) centered on a given position,
    with dimensions based on a reference image size.
    """

    def __init__(self, scale_factor: float = 4.0):
        """
        Initialize the OrthoCropper.

        Args:
            scale_factor: Multiplier for the crop size relative to the reference image.
                          Default is 2.0 (crop size = 2x reference image size).
        """
        self.scale_factor = scale_factor

    def crop(
        self,
        ortho_path: str,
        reference_image_path: str,
        center_row: float,
        center_col: float,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Crop the orthophoto centered on the given position.

        Args:
            ortho_path: Path to the orthophoto (TIFF) to crop.
            reference_image_path: Path to the reference image (e.g., top_view.png)
                                  used to determine the crop dimensions.
            center_row: Center row (Y) position in pixels from DINO matching.
            center_col: Center column (X) position in pixels from DINO matching.
            output_path: Optional output path for the cropped image.
                         If None, saves as '{original_name}_cropped.tif'.

        Returns:
            Path to the cropped orthophoto.
        """
        # Load reference image to get dimensions
        ref_img = Image.open(reference_image_path)
        ref_width, ref_height = ref_img.size
        logger.info(f"Reference image size: {ref_width}x{ref_height} px")

        # Calculate crop dimensions (2x reference size by default)
        crop_width = int(ref_width * self.scale_factor)
        crop_height = int(ref_height * self.scale_factor)
        logger.info(f"Crop size (scale={self.scale_factor}x): {crop_width}x{crop_height} px")

        # Load orthophoto
        ortho_img = Image.open(ortho_path)
        ortho_width, ortho_height = ortho_img.size
        logger.info(f"Orthophoto size: {ortho_width}x{ortho_height} px")

        # Calculate crop boundaries centered on the DINO match position
        # Note: center_row corresponds to Y, center_col corresponds to X
        half_width = crop_width // 2
        half_height = crop_height // 2

        left = int(center_col - half_width)
        upper = int(center_row - half_height)
        right = int(center_col + half_width)
        lower = int(center_row + half_height)

        # Clamp to image boundaries
        left_clamped = max(0, left)
        upper_clamped = max(0, upper)
        right_clamped = min(ortho_width, right)
        lower_clamped = min(ortho_height, lower)

        logger.info(f"Crop bounds: left={left_clamped}, upper={upper_clamped}, "
                    f"right={right_clamped}, lower={lower_clamped}")

        # Perform the crop
        cropped_img = ortho_img.crop((left_clamped, upper_clamped, right_clamped, lower_clamped))

        # If we had to clamp, pad the image to maintain the requested size
        if (left < 0 or upper < 0 or right > ortho_width or lower > ortho_height):
            # Create a new image with the full crop size
            padded_img = Image.new(ortho_img.mode, (crop_width, crop_height), color=0)
            
            # Calculate paste position
            paste_x = max(0, -left)
            paste_y = max(0, -upper)
            padded_img.paste(cropped_img, (paste_x, paste_y))
            cropped_img = padded_img
            logger.warning("Crop extended beyond image bounds, padding applied.")

        # Determine output path
        if output_path is None:
            base, ext = os.path.splitext(ortho_path)
            output_path = f"{base}_cropped{ext}"

        # Try to preserve GeoTIFF metadata if available
        try:
            self._save_with_geotiff_metadata(
                ortho_path, cropped_img, output_path,
                left_clamped, upper_clamped
            )
        except Exception as e:
            logger.warning(f"Could not preserve GeoTIFF metadata: {e}. Saving as regular TIFF.")
            cropped_img.save(output_path, format="TIFF")

        logger.info(f"✅ Cropped orthophoto saved to: {output_path}")
        return output_path

    def _save_with_geotiff_metadata(
        self,
        original_tiff_path: str,
        cropped_image: Image.Image,
        output_path: str,
        offset_x: int,
        offset_y: int
    ):
        """
        Save the cropped image with updated GeoTIFF metadata.

        Args:
            original_tiff_path: Path to the original GeoTIFF file.
            cropped_image: The cropped PIL Image.
            output_path: Output path for the cropped GeoTIFF.
            offset_x: X offset in pixels from original image origin.
            offset_y: Y offset in pixels from original image origin.
        """
        try:
            import rasterio
            from rasterio.transform import Affine

            # Read original metadata
            with rasterio.open(original_tiff_path) as src:
                original_transform = src.transform
                crs = src.crs
                
                # Calculate new transform for the cropped area
                # New origin is shifted by the offset
                new_origin_x = original_transform.c + offset_x * original_transform.a
                new_origin_y = original_transform.f + offset_y * original_transform.e
                
                new_transform = Affine(
                    original_transform.a,  # pixel width
                    original_transform.b,  # rotation (usually 0)
                    new_origin_x,          # x origin (updated)
                    original_transform.d,  # rotation (usually 0)
                    original_transform.e,  # pixel height (negative)
                    new_origin_y           # y origin (updated)
                )

            # Convert PIL Image to numpy array
            cropped_array = np.array(cropped_image)
            
            # Determine number of bands
            if len(cropped_array.shape) == 2:
                count = 1
                cropped_array = cropped_array[np.newaxis, :, :]
            else:
                count = cropped_array.shape[2]
                cropped_array = np.moveaxis(cropped_array, -1, 0)

            # Write the cropped GeoTIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=cropped_image.height,
                width=cropped_image.width,
                count=count,
                dtype=cropped_array.dtype,
                crs=crs,
                transform=new_transform,
            ) as dst:
                dst.write(cropped_array)

            logger.info("GeoTIFF metadata preserved in cropped image.")

        except ImportError:
            logger.warning("rasterio not installed. Saving without geospatial metadata.")
            cropped_image.save(output_path, format="TIFF")


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(message)s')

    parser = argparse.ArgumentParser(description="Find best alignment between two images using DINO.")
    parser.add_argument("directory", help="Directory containing exactly two images")
    parser.add_argument("--dino-version", choices=["v2", "v3"], default="v2",
                        help="DINO model version: v2 (DINOv2) or v3 (DINOv3, requires HF auth)")
    parser.add_argument("--downscale-factor", type=float, default=1.0,
                        help="Downscale factor for input images. Default: 1.0 (no scaling)")
    args = parser.parse_args()

    images = DinoImageMatcher.list_images(args.directory)
    if len(images) < 2:
        print(f"Error: Found only {len(images)} image(s), need at least 2.")
        exit(1)

    if len(images) > 2:
        print(f"Warning: Found {len(images)} images, using the first two.")

    print(f"  Image 1 (Base): {os.path.basename(images[0])}")
    print(f"  Image 2 (Template): {os.path.basename(images[1])}")

    matcher = DinoImageMatcher(
        dino_version=args.dino_version,
        downscale_factor=args.downscale_factor,
    )

    output_dir = os.path.join(args.directory, "outputs")
    center_row, center_col = matcher.match(images[0], images[1], output_dir=output_dir)
    
    print(f"\n>>> Center Position (row, col): ({center_row:.2f}, {center_col:.2f})")