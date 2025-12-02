"""
Keypoint extraction module for provenance analysis.

Supports multiple descriptor types:
- VLFeat SIFT with histogram equalization (default, best for scientific images)
- OpenCV SIFT
- OpenCV RootSIFT
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional
from pathlib import Path
import logging

from .schemas import DescriptorType

logger = logging.getLogger(__name__)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image: Grayscale image as numpy array
        
    Returns:
        Histogram-equalized image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(image)


def extract_vlfeat_sift_heq(
    image: np.ndarray,
    peak_thresh: float = 0.01,
    edge_thresh: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT keypoints and descriptors using VLFeat with histogram equalization.
    
    This is the recommended descriptor for scientific image analysis as it
    provides better matching for images with similar textures.
    
    Args:
        image: Input image (grayscale or RGB)
        peak_thresh: Peak threshold for SIFT
        edge_thresh: Edge threshold for SIFT
        
    Returns:
        Tuple of (keypoints, descriptors)
        - keypoints: Nx2 array of (x, y) coordinates
        - descriptors: NxD array of descriptors
    """
    try:
        from cyvlfeat.sift import sift
    except ImportError:
        raise ImportError(
            "cyvlfeat is required for vlfeat_sift_heq descriptor. "
            "Install via conda: conda install -c conda-forge cyvlfeat"
        )
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply histogram equalization
    gray_heq = histogram_equalization(gray)
    
    # Convert to float32 for VLFeat (expects values in [0, 255])
    gray_float = gray_heq.astype(np.float32)
    
    # Extract SIFT features using VLFeat
    frames, descriptors = sift(
        gray_float,
        peak_thresh=peak_thresh,
        edge_thresh=edge_thresh,
        compute_descriptor=True
    )
    
    if frames is None or len(frames) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)
    
    # VLFeat returns frames as (y, x, scale, orientation)
    # We need (x, y) for keypoints
    keypoints = frames[:, :2][:, ::-1]  # Swap to (x, y)
    
    return keypoints.astype(np.float32), descriptors.astype(np.float32)


def extract_cv_sift(
    image: np.ndarray,
    n_features: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT keypoints and descriptors using OpenCV.
    
    Args:
        image: Input image (grayscale or RGB)
        n_features: Maximum number of features (0 = unlimited)
        contrast_threshold: Contrast threshold for SIFT
        edge_threshold: Edge threshold for SIFT
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold
    )
    
    # Detect and compute
    kps, descriptors = sift.detectAndCompute(gray, None)
    
    if kps is None or len(kps) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)
    
    # Convert keypoints to numpy array
    keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    
    return keypoints, descriptors.astype(np.float32)


def extract_cv_rsift(
    image: np.ndarray,
    n_features: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract RootSIFT keypoints and descriptors using OpenCV.
    
    RootSIFT applies Hellinger normalization to SIFT descriptors,
    which can improve matching performance.
    
    Args:
        image: Input image (grayscale or RGB)
        n_features: Maximum number of features (0 = unlimited)
        contrast_threshold: Contrast threshold for SIFT
        edge_threshold: Edge threshold for SIFT
        
    Returns:
        Tuple of (keypoints, descriptors)
    """
    keypoints, descriptors = extract_cv_sift(
        image, n_features, contrast_threshold, edge_threshold
    )
    
    if len(descriptors) == 0:
        return keypoints, descriptors
    
    # Apply RootSIFT transformation (Hellinger kernel)
    # 1. L1 normalize
    descriptors = descriptors / (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + 1e-7)
    # 2. Square root
    descriptors = np.sqrt(descriptors)
    
    return keypoints, descriptors.astype(np.float32)


def extract_descriptors(
    image_path: str,
    descriptor_type: DescriptorType = DescriptorType.VLFEAT_SIFT_HEQ,
    extract_flip: bool = True
) -> dict:
    """
    Extract keypoint descriptors from an image.
    
    Args:
        image_path: Path to the input image
        descriptor_type: Type of descriptor to extract
        extract_flip: Also extract from horizontally flipped image
        
    Returns:
        Dictionary containing:
        - keypoints: Nx2 array of keypoint coordinates
        - descriptors: NxD array of descriptors
        - flip_keypoints: (optional) Nx2 array for flipped image
        - flip_descriptors: (optional) NxD array for flipped image
    """
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Select extraction function based on descriptor type
    extract_fn = {
        DescriptorType.VLFEAT_SIFT_HEQ: extract_vlfeat_sift_heq,
        DescriptorType.CV_SIFT: extract_cv_sift,
        DescriptorType.CV_RSIFT: extract_cv_rsift,
    }.get(descriptor_type)
    
    if extract_fn is None:
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")
    
    # Extract from original image
    keypoints, descriptors = extract_fn(image)
    logger.info(f"Extracted {len(keypoints)} keypoints from {image_path}")
    
    result = {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'image_shape': image.shape[:2]  # (height, width)
    }
    
    # Extract from flipped image if requested
    if extract_flip:
        flipped_image = np.fliplr(image)
        flip_keypoints, flip_descriptors = extract_fn(flipped_image)
        
        # Adjust x-coordinates for flip (x' = width - 1 - x)
        if len(flip_keypoints) > 0:
            width = image.shape[1]
            flip_keypoints[:, 0] = width - 1 - flip_keypoints[:, 0]
        
        result['flip_keypoints'] = flip_keypoints
        result['flip_descriptors'] = flip_descriptors
        logger.info(f"Extracted {len(flip_keypoints)} keypoints from flipped image")
    
    return result


def save_descriptors(
    result: dict,
    output_dir: str,
    base_name: str
) -> dict:
    """
    Save extracted descriptors to numpy files.
    
    Args:
        result: Dictionary from extract_descriptors()
        output_dir: Directory to save files
        base_name: Base name for output files
        
    Returns:
        Dictionary with paths to saved files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save keypoints
    kps_path = output_path / f"{base_name}_kps.npy"
    np.save(kps_path, result['keypoints'])
    paths['keypoints_path'] = str(kps_path)
    
    # Save descriptors
    desc_path = output_path / f"{base_name}_desc.npy"
    np.save(desc_path, result['descriptors'])
    paths['descriptors_path'] = str(desc_path)
    
    # Save flipped versions if present
    if 'flip_keypoints' in result:
        flip_kps_path = output_path / f"{base_name}_flip_kps.npy"
        np.save(flip_kps_path, result['flip_keypoints'])
        paths['flip_keypoints_path'] = str(flip_kps_path)
        
        flip_desc_path = output_path / f"{base_name}_flip_desc.npy"
        np.save(flip_desc_path, result['flip_descriptors'])
        paths['flip_descriptors_path'] = str(flip_desc_path)
    
    # Save image shape for reference
    shape_path = output_path / f"{base_name}_shape.npy"
    np.save(shape_path, np.array(result['image_shape']))
    paths['shape_path'] = str(shape_path)
    
    return paths


def load_descriptors(
    keypoints_path: str,
    descriptors_path: str,
    flip_keypoints_path: Optional[str] = None,
    flip_descriptors_path: Optional[str] = None
) -> dict:
    """
    Load descriptors from numpy files.
    
    Args:
        keypoints_path: Path to keypoints .npy file
        descriptors_path: Path to descriptors .npy file
        flip_keypoints_path: Optional path to flipped keypoints
        flip_descriptors_path: Optional path to flipped descriptors
        
    Returns:
        Dictionary with loaded arrays
    """
    result = {
        'keypoints': np.load(keypoints_path),
        'descriptors': np.load(descriptors_path)
    }
    
    if flip_keypoints_path and flip_descriptors_path:
        result['flip_keypoints'] = np.load(flip_keypoints_path)
        result['flip_descriptors'] = np.load(flip_descriptors_path)
    
    return result
