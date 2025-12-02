"""
Keypoint matching and geometric verification module.

This module implements:
- G2NN (Generalized 2-Nearest Neighbor) keypoint selection
- Geometric consistency verification
- Homography-based alignment (RANSAC, MAGSAC, etc.)
- Shared content area calculation
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional
import logging

from .schemas import AlignmentStrategy, MatchingMethod

logger = logging.getLogger(__name__)


def g2nn_keypoint_selection(
    keypoints1: np.ndarray,
    descriptions1: np.ndarray,
    keypoints2: np.ndarray,
    descriptions2: np.ndarray,
    k_rate: float = 0.5,
    nndr_threshold: float = 0.75,
    matching_method: MatchingMethod = MatchingMethod.BF,
    eps: float = 1e-7
) -> Tuple[List[int], List[int]]:
    """
    Perform G2NN (Generalized 2-Nearest Neighbor) keypoint selection.
    
    This method selects keypoints that have consistent matches based on
    the nearest neighbor distance ratio test.
    
    Args:
        keypoints1: Keypoints from image 1 (Nx2)
        descriptions1: Descriptors from image 1 (NxD)
        keypoints2: Keypoints from image 2 (Mx2)
        descriptions2: Descriptors from image 2 (MxD)
        k_rate: Rate to define number of neighbors to match
        nndr_threshold: NNDR threshold for match selection
        matching_method: BF (Brute Force) or FLANN
        eps: Small value to avoid division by zero
        
    Returns:
        Tuple of (indices1, indices2) for matched keypoints
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return [], []
    
    # Ensure correct dtype for matchers
    descriptions1 = descriptions1.astype(np.float32)
    descriptions2 = descriptions2.astype(np.float32)
    
    # Swap so smaller set is keypoints1
    swapped = False
    if len(keypoints2) < len(keypoints1):
        keypoints1, keypoints2 = keypoints2, keypoints1
        descriptions1, descriptions2 = descriptions2, descriptions1
        swapped = True
    
    # Compute k for kNN
    k = max(2, int(round(len(keypoints1) * k_rate)))
    
    # Match keypoints
    if matching_method == MatchingMethod.BF:
        matcher = cv2.BFMatcher()
        knn_matches = matcher.knnMatch(descriptions1, descriptions2, k=k)
    else:  # FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        knn_matches = matcher.knnMatch(descriptions1, descriptions2, k=min(k, 2))
    
    # G2NN match selection
    selected_matches = []
    for matches in knn_matches:
        if len(matches) < 2:
            continue
        for i in range(len(matches) - 1):
            if matches[i].distance / (matches[i + 1].distance + eps) < nndr_threshold:
                selected_matches.append(matches[i])
            else:
                break
    
    # Select unique keypoint pairs
    indices1 = []
    indices2 = []
    distances = []
    
    for match in selected_matches:
        if match.queryIdx not in indices1 and match.trainIdx not in indices2:
            indices1.append(match.queryIdx)
            indices2.append(match.trainIdx)
            distances.append(match.distance)
        else:
            # If already matched, keep the one with smaller distance
            if match.queryIdx in indices1:
                i = indices1.index(match.queryIdx)
            else:
                i = indices2.index(match.trainIdx)
            
            if distances[i] > match.distance:
                indices1[i] = match.queryIdx
                indices2[i] = match.trainIdx
                distances[i] = match.distance
    
    # Undo swap if necessary
    if swapped:
        indices1, indices2 = indices2, indices1
    
    return indices1, indices2


def verify_geometric_consistency(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC,
    displacement_thresh: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify geometric consistency of matched keypoints using homography or
    fundamental matrix estimation.
    
    Args:
        keypoints1: Matched keypoints from image 1 (Nx2)
        keypoints2: Matched keypoints from image 2 (Nx2)
        alignment_strategy: Method for geometric verification
        displacement_thresh: Maximum displacement for inliers
        
    Returns:
        Tuple of (consistent_kpts1, consistent_kpts2)
    """
    if len(keypoints1) < 4:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    keypoints1 = keypoints1.astype(np.float32)
    keypoints2 = keypoints2.astype(np.float32)
    
    # Use fundamental matrix for MAGSAC/DEGENSAC, homography for others
    use_homography = alignment_strategy in [
        AlignmentStrategy.CV_RANSAC,
        AlignmentStrategy.CV_LMEDS,
    ]
    
    if use_homography:
        # Estimate homography
        method_map = {
            AlignmentStrategy.CV_RANSAC: cv2.RANSAC,
            AlignmentStrategy.CV_LMEDS: cv2.LMEDS,
        }
        method = method_map.get(alignment_strategy, cv2.RANSAC)
        
        H, inliers = cv2.findHomography(
            keypoints1, keypoints2, method,
            ransacReprojThreshold=3.0,
            confidence=0.999,
            maxIters=100000
        )
        
        if H is None or inliers is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        # Verify alignment by transforming points
        pts = keypoints1.reshape(-1, 1, 2)
        aligned = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        
        # Check displacement
        displacements = np.sqrt(np.sum((aligned - keypoints2) ** 2, axis=1))
        consistent = displacements < displacement_thresh
        
    else:
        # Use fundamental matrix (MAGSAC++ or DEGENSAC)
        method_map = {
            AlignmentStrategy.CV_MAGSAC: cv2.USAC_MAGSAC,
        }
        method = method_map.get(alignment_strategy, cv2.USAC_MAGSAC)
        
        F, inliers = cv2.findFundamentalMat(
            keypoints1, keypoints2, method,
            ransacReprojThreshold=0.5,
            confidence=0.999,
            maxIters=100000
        )
        
        if F is None or inliers is None:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        consistent = inliers.flatten() > 0
    
    return keypoints1[consistent], keypoints2[consistent]


def compute_shared_area(
    image_shape: Tuple[int, int],
    keypoints: np.ndarray
) -> float:
    """
    Compute the fraction of image area covered by keypoints using convex hull.
    
    Args:
        image_shape: (height, width) of the image
        keypoints: Nx2 array of keypoint coordinates
        
    Returns:
        Fraction of image area covered (0.0 to 1.0)
    """
    if len(keypoints) < 3:
        return 0.0
    
    # Compute convex hull
    hull = cv2.convexHull(keypoints.astype(np.int32))
    
    # Create mask and compute area
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    
    hull_area = np.sum(mask)
    total_area = image_shape[0] * image_shape[1]
    
    return hull_area / total_area


def align_and_compute_shared_area(
    keypoints1: np.ndarray,
    descriptors1: np.ndarray,
    keypoints2: np.ndarray,
    descriptors2: np.ndarray,
    image1_shape: Tuple[int, int],
    image2_shape: Tuple[int, int],
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC,
    matching_method: MatchingMethod = MatchingMethod.BF,
    min_keypoints: int = 20
) -> Tuple[float, float, int]:
    """
    Match two images and compute shared content area.
    
    Args:
        keypoints1: Keypoints from image 1
        descriptors1: Descriptors from image 1
        keypoints2: Keypoints from image 2
        descriptors2: Descriptors from image 2
        image1_shape: (height, width) of image 1
        image2_shape: (height, width) of image 2
        alignment_strategy: Geometric verification method
        matching_method: Keypoint matching method
        min_keypoints: Minimum matches required
        
    Returns:
        Tuple of (shared_area1, shared_area2, num_matches)
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return 0.0, 0.0, 0
    
    # Step 1: G2NN keypoint selection
    indices1, indices2 = g2nn_keypoint_selection(
        keypoints1, descriptors1,
        keypoints2, descriptors2,
        matching_method=matching_method
    )
    
    if len(indices1) < min_keypoints:
        return 0.0, 0.0, 0
    
    matched_kpts1 = keypoints1[indices1]
    matched_kpts2 = keypoints2[indices2]
    
    # Step 2: Geometric consistency verification
    consistent_kpts1, consistent_kpts2 = verify_geometric_consistency(
        matched_kpts1, matched_kpts2,
        alignment_strategy=alignment_strategy
    )
    
    if len(consistent_kpts1) < min_keypoints:
        return 0.0, 0.0, 0
    
    # Step 3: Compute shared area
    area1 = compute_shared_area(image1_shape, consistent_kpts1)
    area2 = compute_shared_area(image2_shape, consistent_kpts2)
    
    return area1, area2, len(consistent_kpts1)


def match_images(
    image1_path: str,
    keypoints1: np.ndarray,
    descriptors1: np.ndarray,
    image2_path: str,
    keypoints2: np.ndarray,
    descriptors2: np.ndarray,
    flip_keypoints1: Optional[np.ndarray] = None,
    flip_descriptors1: Optional[np.ndarray] = None,
    alignment_strategy: AlignmentStrategy = AlignmentStrategy.CV_MAGSAC,
    matching_method: MatchingMethod = MatchingMethod.BF,
    min_keypoints: int = 20,
    min_area: float = 0.01,
    check_flip: bool = True
) -> dict:
    """
    Match two images and determine if they share content.
    
    This function:
    1. Matches keypoints between the two images
    2. Verifies geometric consistency
    3. Computes shared content area
    4. Optionally checks flipped version of image 1
    
    Args:
        image1_path: Path to first image
        keypoints1: Keypoints from image 1
        descriptors1: Descriptors from image 1
        image2_path: Path to second image
        keypoints2: Keypoints from image 2
        descriptors2: Descriptors from image 2
        flip_keypoints1: Keypoints from flipped image 1
        flip_descriptors1: Descriptors from flipped image 1
        alignment_strategy: Geometric verification method
        matching_method: Keypoint matching method
        min_keypoints: Minimum matches required
        min_area: Minimum shared area threshold
        check_flip: Whether to check flipped version
        
    Returns:
        Dictionary with match results
    """
    # Load images to get shapes
    img1 = np.array(Image.open(image1_path))
    img2 = np.array(Image.open(image2_path))
    shape1 = img1.shape[:2]
    shape2 = img2.shape[:2]
    
    # Try regular matching
    area1, area2, num_matches = align_and_compute_shared_area(
        keypoints1, descriptors1,
        keypoints2, descriptors2,
        shape1, shape2,
        alignment_strategy=alignment_strategy,
        matching_method=matching_method,
        min_keypoints=min_keypoints
    )
    
    is_flipped = False
    
    # Try flipped matching if enabled and regular matching didn't find significant overlap
    if check_flip and flip_keypoints1 is not None and flip_descriptors1 is not None:
        if max(area1, area2) < min_area:
            flip_area1, flip_area2, flip_matches = align_and_compute_shared_area(
                flip_keypoints1, flip_descriptors1,
                keypoints2, descriptors2,
                shape1, shape2,
                alignment_strategy=alignment_strategy,
                matching_method=matching_method,
                min_keypoints=min_keypoints
            )
            
            # Use flipped result if better
            if max(flip_area1, flip_area2) > max(area1, area2):
                area1, area2 = flip_area1, flip_area2
                num_matches = flip_matches
                is_flipped = True
    
    return {
        'shared_area_img1': area1,
        'shared_area_img2': area2,
        'matched_keypoints': num_matches,
        'is_flipped_match': is_flipped,
        'is_match': min(area1, area2) >= min_area
    }


def draw_match_visualization(
    image1_path: str,
    keypoints1: np.ndarray,
    image2_path: str,
    keypoints2: np.ndarray,
    output_path: str
) -> str:
    """
    Draw visualization of matched regions using convex hulls.
    
    Args:
        image1_path: Path to first image
        keypoints1: Matched keypoints in image 1
        image2_path: Path to second image
        keypoints2: Matched keypoints in image 2
        output_path: Path to save visualization
        
    Returns:
        Path to saved visualization
    """
    img1 = np.array(Image.open(image1_path).convert('RGB'))
    img2 = np.array(Image.open(image2_path).convert('RGB'))
    
    if len(keypoints1) < 3 or len(keypoints2) < 3:
        # Just concatenate images if not enough points for hull
        result = np.hstack([img1, img2])
        Image.fromarray(result).save(output_path)
        return output_path
    
    # Draw convex hulls
    hull1 = cv2.convexHull(keypoints1.astype(np.int32))
    hull2 = cv2.convexHull(keypoints2.astype(np.int32))
    
    img1_vis = cv2.drawContours(img1.copy(), [hull1], 0, (0, 255, 0), 2)
    img2_vis = cv2.drawContours(img2.copy(), [hull2], 0, (0, 255, 0), 2)
    
    # Draw keypoints
    for kp in keypoints1.astype(np.int32):
        cv2.circle(img1_vis, tuple(kp), 3, (255, 0, 0), -1)
    for kp in keypoints2.astype(np.int32):
        cv2.circle(img2_vis, tuple(kp), 3, (255, 0, 0), -1)
    
    # Resize to same height for side-by-side display
    h1, w1 = img1_vis.shape[:2]
    h2, w2 = img2_vis.shape[:2]
    
    if h1 != h2:
        target_h = max(h1, h2)
        if h1 < target_h:
            scale = target_h / h1
            img1_vis = cv2.resize(img1_vis, (int(w1 * scale), target_h))
        else:
            scale = target_h / h2
            img2_vis = cv2.resize(img2_vis, (int(w2 * scale), target_h))
    
    # Concatenate
    result = np.hstack([img1_vis, img2_vis])
    Image.fromarray(result).save(output_path)
    
    return output_path
