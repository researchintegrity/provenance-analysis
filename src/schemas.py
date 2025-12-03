"""
Pydantic schemas for provenance analysis input/output validation.

These schemas define the contract between the ELIS backend and the
provenance Docker container.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class DescriptorType(str, Enum):
    """Supported keypoint descriptor types."""
    VLFEAT_SIFT_HEQ = "vlfeat_sift_heq"  # VLFeat SIFT with histogram equalization
    CV_SIFT = "cv_sift"                   # OpenCV SIFT
    CV_RSIFT = "cv_rsift"                 # OpenCV RootSIFT


class AlignmentStrategy(str, Enum):
    """Supported geometric alignment strategies."""
    CV_MAGSAC = "CV_MAGSAC"       # MAGSAC++ (recommended)
    CV_RANSAC = "CV_RANSAC"       # Classic RANSAC
    CV_LMEDS = "CV_LMEDS"         # Least Median of Squares
    CLUSTER = "cluster"           # Cluster-based alignment


class MatchingMethod(str, Enum):
    """Keypoint matching methods."""
    BF = "BF"          # Brute Force
    FLANN = "FLANN"    # FLANN-based


# ============================================================================
# Descriptor Extraction Schemas
# ============================================================================

class ExtractDescriptorsRequest(BaseModel):
    """Request to extract keypoint descriptors from an image."""
    image_path: str = Field(..., description="Path to the input image")
    descriptor_type: DescriptorType = Field(
        default=DescriptorType.VLFEAT_SIFT_HEQ,
        description="Type of descriptor to extract"
    )
    extract_flip: bool = Field(
        default=True,
        description="Also extract descriptors from horizontally flipped image"
    )
    output_dir: str = Field(..., description="Directory to save descriptor files")


class ExtractDescriptorsResponse(BaseModel):
    """Response from descriptor extraction."""
    success: bool
    message: str
    keypoints_path: Optional[str] = Field(None, description="Path to keypoints .npy file")
    descriptors_path: Optional[str] = Field(None, description="Path to descriptors .npy file")
    flip_keypoints_path: Optional[str] = Field(None, description="Path to flipped keypoints .npy file")
    flip_descriptors_path: Optional[str] = Field(None, description="Path to flipped descriptors .npy file")
    keypoint_count: int = Field(default=0, description="Number of keypoints extracted")
    flip_keypoint_count: int = Field(default=0, description="Number of flipped keypoints extracted")


# ============================================================================
# Pairwise Matching Schemas
# ============================================================================

class ImageDescriptorInfo(BaseModel):
    """Information about an image and its pre-computed descriptors."""
    image_path: str = Field(..., description="Path to the image file")
    keypoints_path: str = Field(..., description="Path to keypoints .npy file")
    descriptors_path: str = Field(..., description="Path to descriptors .npy file")
    flip_keypoints_path: Optional[str] = Field(None, description="Path to flipped keypoints")
    flip_descriptors_path: Optional[str] = Field(None, description="Path to flipped descriptors")


class PairwiseMatchRequest(BaseModel):
    """Request to match two images and compute shared content area."""
    image1: ImageDescriptorInfo = Field(..., description="First image info")
    image2: ImageDescriptorInfo = Field(..., description="Second image info")
    alignment_strategy: AlignmentStrategy = Field(
        default=AlignmentStrategy.CV_MAGSAC,
        description="Geometric alignment strategy"
    )
    matching_method: MatchingMethod = Field(
        default=MatchingMethod.BF,
        description="Keypoint matching method"
    )
    min_keypoints: int = Field(
        default=20,
        description="Minimum number of matching keypoints required"
    )
    min_area: float = Field(
        default=0.01,
        description="Minimum shared area threshold (0-1)"
    )
    check_flip: bool = Field(
        default=True,
        description="Check flipped version of images"
    )
    output_dir: Optional[str] = Field(
        None,
        description="Optional directory to save visualization"
    )


class MatchResult(BaseModel):
    """Result of matching two images."""
    shared_area_img1: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Fraction of image 1 area that is shared (0-1)"
    )
    shared_area_img2: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Fraction of image 2 area that is shared (0-1)"
    )
    matched_keypoints: int = Field(
        default=0,
        description="Number of matched keypoints"
    )
    is_flipped_match: bool = Field(
        default=False,
        description="Whether the match was found with flipped image"
    )
    visualization_path: Optional[str] = Field(
        None,
        description="Path to visualization image if generated"
    )


class PairwiseMatchResponse(BaseModel):
    """Response from pairwise matching."""
    success: bool
    message: str
    result: Optional[MatchResult] = None


# ============================================================================
# Batch Processing Schemas
# ============================================================================

class BatchMatchRequest(BaseModel):
    """Request to match multiple image pairs."""
    pairs: List[PairwiseMatchRequest] = Field(
        ...,
        description="List of image pairs to match"
    )
    parallel: bool = Field(
        default=True,
        description="Process pairs in parallel"
    )
    max_workers: int = Field(
        default=4,
        description="Maximum parallel workers"
    )


class BatchMatchResult(BaseModel):
    """Result for a single pair in batch processing."""
    pair_index: int
    image1_path: str
    image2_path: str
    result: Optional[MatchResult] = None
    error: Optional[str] = None


class BatchMatchResponse(BaseModel):
    """Response from batch matching."""
    success: bool
    message: str
    total_pairs: int
    successful_matches: int
    failed_matches: int
    results: List[BatchMatchResult]


# ============================================================================
# CLI Command Schemas
# ============================================================================

class CommandType(str, Enum):
    """Available commands for the container."""
    EXTRACT = "extract"       # Extract descriptors from image
    MATCH = "match"           # Match two images
    BATCH_MATCH = "batch"     # Batch match multiple pairs
    PROVENANCE = "provenance" # Full provenance analysis


# ============================================================================
# Provenance Analysis Schemas (Full Pipeline)
# ============================================================================

class ImageDescriptorPaths(BaseModel):
    """Pre-computed descriptor file paths for an image."""
    keypoints_path: str = Field(..., description="Path to keypoints .npy file")
    descriptors_path: str = Field(..., description="Path to descriptors .npy file")
    flip_keypoints_path: Optional[str] = Field(None, description="Path to flipped keypoints .npy file")
    flip_descriptors_path: Optional[str] = Field(None, description="Path to flipped descriptors .npy file")


class ImageInfo(BaseModel):
    """Information about an image for provenance analysis."""
    id: str = Field(..., description="Unique image identifier")
    path: str = Field(..., description="Path to the image file")
    label: str = Field(default="", description="Display label for the image")
    is_query: bool = Field(default=False, description="Whether this is a query image")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")
    descriptor_paths: Optional[ImageDescriptorPaths] = Field(
        default=None,
        description="Pre-computed descriptor file paths (if available)"
    )


class ProvenanceRequest(BaseModel):
    """Request for full provenance analysis."""
    images: List[ImageInfo] = Field(..., description="List of images to analyze")
    query_image_ids: List[str] = Field(..., description="IDs of query images")
    descriptor_type: DescriptorType = Field(
        default=DescriptorType.VLFEAT_SIFT_HEQ,
        description="Type of descriptor to use"
    )
    alignment_strategy: AlignmentStrategy = Field(
        default=AlignmentStrategy.CV_MAGSAC,
        description="Geometric alignment strategy"
    )
    matching_method: MatchingMethod = Field(
        default=MatchingMethod.BF,
        description="Keypoint matching method"
    )
    min_keypoints: int = Field(default=20, description="Minimum matching keypoints")
    min_area: float = Field(default=0.01, description="Minimum shared area threshold")
    check_flip: bool = Field(default=True, description="Check flipped images")
    output_dir: str = Field(..., description="Directory to save results")
    parallel: bool = Field(default=True, description="Process pairs in parallel")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    save_descriptors: bool = Field(default=True, description="Save computed descriptors")


class ProvenanceMatchedPair(BaseModel):
    """A matched pair in provenance analysis."""
    image1_id: str
    image2_id: str
    shared_area_img1: float
    shared_area_img2: float
    matched_keypoints: int
    is_flipped: bool = False


class ProvenanceGraphNode(BaseModel):
    """Node in provenance graph."""
    id: str
    label: str
    image_path: str
    is_query: bool = False
    metadata: Optional[dict] = None


class ProvenanceGraphEdge(BaseModel):
    """Edge in provenance graph."""
    source: str
    target: str
    weight: float
    shared_area_source: float
    shared_area_target: float
    matched_keypoints: int
    is_flipped: bool = False


class ProvenanceGraphResult(BaseModel):
    """Result graph from provenance analysis."""
    nodes: List[ProvenanceGraphNode]
    edges: List[ProvenanceGraphEdge]
    spanning_tree_edges: Optional[List[ProvenanceGraphEdge]] = None
    connected_components: Optional[List[List[str]]] = None
    adjacency_matrix: Optional[dict] = None


class ProvenanceResponse(BaseModel):
    """Response from provenance analysis."""
    success: bool
    message: str
    total_images: int = 0
    total_pairs_checked: int = 0
    matched_pairs_count: int = 0
    processing_time_seconds: float = 0.0
    graph: Optional[ProvenanceGraphResult] = None
    matched_pairs: Optional[List[ProvenanceMatchedPair]] = None
    visualization_data: Optional[dict] = None
    output_files: Optional[dict] = None


class ContainerInput(BaseModel):
    """Root input schema for the container."""
    command: CommandType
    extract_request: Optional[ExtractDescriptorsRequest] = None
    match_request: Optional[PairwiseMatchRequest] = None
    batch_request: Optional[BatchMatchRequest] = None
    provenance_request: Optional[ProvenanceRequest] = None


class ContainerOutput(BaseModel):
    """Root output schema from the container."""
    success: bool
    command: CommandType
    message: str
    extract_response: Optional[ExtractDescriptorsResponse] = None
    match_response: Optional[PairwiseMatchResponse] = None
    batch_response: Optional[BatchMatchResponse] = None
    provenance_response: Optional[ProvenanceResponse] = None


# ============================================================================
# Microservice API Schemas
# ============================================================================

class MicroserviceImageInput(BaseModel):
    """Image input for microservice requests."""
    id: str = Field(..., description="Unique image identifier")
    path: str = Field(..., description="Path to the image file")
    label: Optional[str] = Field(default=None, description="Image label/category")


class MicroserviceAnalysisRequest(BaseModel):
    """
    Request for provenance analysis microservice.
    
    The microservice will:
    1. Check visibility of all images in CBIR
    2. Index any missing images in batches
    3. Query CBIR for Top-K similar images
    4. Extract descriptors in parallel
    5. Perform matching and expansion (Top-Q)
    6. Build provenance graph
    """
    # User identification (required for CBIR isolation)
    user_id: str = Field(
        ...,
        description="User ID for CBIR multi-tenant isolation. Each user has their own image index."
    )
    
    # Input images
    images: List[MicroserviceImageInput] = Field(
        ..., 
        description="List of images available for analysis"
    )
    query_image: MicroserviceImageInput = Field(
        ..., 
        description="The query image to analyze provenance for"
    )
    
    # CBIR parameters
    k: int = Field(
        default=10, 
        ge=1, 
        le=100,
        description="Number of top candidates from initial CBIR search"
    )
    q: int = Field(
        default=5, 
        ge=1, 
        le=50,
        description="Number of top candidates for expansion searches"
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum expansion depth (1 = no expansion)"
    )
    
    # Descriptor parameters
    descriptor_type: DescriptorType = Field(
        default=DescriptorType.CV_RSIFT,
        description="Type of keypoint descriptor to use"
    )
    extract_flip: bool = Field(
        default=True,
        description="Also extract descriptors from flipped images"
    )
    
    # Matching parameters
    alignment_strategy: AlignmentStrategy = Field(
        default=AlignmentStrategy.CV_MAGSAC,
        description="Geometric alignment strategy"
    )
    matching_method: MatchingMethod = Field(
        default=MatchingMethod.BF,
        description="Keypoint matching method"
    )
    min_keypoints: int = Field(
        default=20,
        ge=4,
        description="Minimum matching keypoints for a valid match"
    )
    min_area: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum shared area threshold"
    )
    
    # Processing parameters
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum parallel workers for descriptor extraction"
    )
    cbir_batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for CBIR indexing"
    )
    
    # Output parameters
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save results (uses temp if not provided)"
    )


class IndexingStatus(BaseModel):
    """Status of image indexing in CBIR."""
    total_images: int = Field(description="Total images provided")
    already_indexed: int = Field(description="Images already visible in CBIR")
    newly_indexed: int = Field(description="Images indexed during this request")
    failed_to_index: int = Field(description="Images that failed to index")
    failed_ids: List[str] = Field(default_factory=list, description="IDs of failed images")


class ExtractionStatus(BaseModel):
    """Status of descriptor extraction."""
    total_images: int = Field(description="Total images needing descriptors")
    extracted: int = Field(description="Successfully extracted")
    from_cache: int = Field(description="Loaded from disk cache")
    failed: int = Field(description="Failed extractions")


class MatchingStatus(BaseModel):
    """Status of matching process."""
    total_pairs_checked: int = Field(description="Total image pairs checked")
    matched_pairs: int = Field(description="Pairs with valid matches")
    expansion_count: int = Field(description="Number of CBIR expansions performed")


class MicroserviceAnalysisResponse(BaseModel):
    """Response from provenance analysis microservice."""
    success: bool = Field(description="Whether analysis completed successfully")
    message: str = Field(description="Status message")
    
    # Timing
    processing_time_seconds: float = Field(
        default=0.0, 
        description="Total processing time"
    )
    
    # Status details
    indexing_status: Optional[IndexingStatus] = Field(
        default=None,
        description="CBIR indexing status"
    )
    extraction_status: Optional[ExtractionStatus] = Field(
        default=None,
        description="Descriptor extraction status"
    )
    matching_status: Optional[MatchingStatus] = Field(
        default=None,
        description="Matching process status"
    )
    
    # Results
    graph: Optional[ProvenanceGraphResult] = Field(
        default=None,
        description="Provenance graph result"
    )
    matched_pairs: Optional[List[ProvenanceMatchedPair]] = Field(
        default=None,
        description="List of matched image pairs"
    )
    
    # Debug/logging info
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings during processing"
    )


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(default="1.0.0", description="Service version")
    cbir_connected: bool = Field(description="CBIR connection status")

