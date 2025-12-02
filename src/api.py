"""
Provenance Analysis Microservice API.

FastAPI-based REST API for provenance analysis with async processing.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging
from pathlib import Path

from .service import ProvenanceMicroservice, ProvenanceService
from .cbir import MockCBIRClient, RestCBIRClient, CBIRClient
from .schemas import (
    MicroserviceAnalysisRequest,
    MicroserviceAnalysisResponse,
    MicroserviceImageInput,
    HealthCheckResponse,
    DescriptorType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = os.getenv("PROVENANCE_OUTPUT_DIR", "/tmp/provenance_output")
CBIR_ENDPOINT = os.getenv("CBIR_ENDPOINT", "http://localhost:8001")
CBIR_USER_ID = os.getenv("CBIR_USER_ID", "provenance_service")
USE_MOCK_CBIR = os.getenv("USE_MOCK_CBIR", "false").lower() == "true"

# Path mapping for CBIR (local path -> remote path)
PATH_MAPPING = {}
local_path = os.getenv("LOCAL_DATA_PATH")
remote_path = os.getenv("REMOTE_DATA_PATH")
if local_path and remote_path:
    PATH_MAPPING[local_path] = remote_path

# ============================================================================
# App Setup
# ============================================================================

app = FastAPI(
    title="Provenance Analysis Microservice",
    description="""
    Microservice for analyzing image provenance and content sharing relationships.
    
    ## Features
    - Automatic CBIR indexing for new images
    - Parallel descriptor extraction
    - Priority-based processing queue
    - Graph-based provenance analysis
    
    ## Pipeline
    1. Check/index images in CBIR system
    2. Search for Top-K similar images
    3. Extract descriptors in parallel
    4. Match image pairs
    5. Expand search on matches (Top-Q)
    6. Build provenance graph
    """,
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Service Initialization
# ============================================================================

def get_cbir_client() -> CBIRClient:
    """Get the appropriate CBIR client based on configuration."""
    if USE_MOCK_CBIR:
        logger.info("Using Mock CBIR client")
        return MockCBIRClient()
    else:
        logger.info(f"Using REST CBIR client: {CBIR_ENDPOINT}")
        return RestCBIRClient(
            endpoint_url=CBIR_ENDPOINT,
            user_id=CBIR_USER_ID,
            path_mapping=PATH_MAPPING if PATH_MAPPING else None
        )

# Global service instance
_cbir_client: Optional[CBIRClient] = None
_service: Optional[ProvenanceMicroservice] = None

def get_service() -> ProvenanceMicroservice:
    """Get or create the service instance."""
    global _cbir_client, _service
    
    if _service is None:
        _cbir_client = get_cbir_client()
        _service = ProvenanceMicroservice(
            cbir_client=_cbir_client,
            default_output_dir=OUTPUT_DIR
        )
        logger.info(f"Initialized ProvenanceMicroservice with output dir: {OUTPUT_DIR}")
    
    return _service

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/analyze", response_model=MicroserviceAnalysisResponse)
async def analyze_provenance(request: MicroserviceAnalysisRequest):
    """
    Analyze provenance for a query image against a set of available images.
    
    The service will:
    1. Check which images are visible in the CBIR system
    2. Index any missing images in batches
    3. Search for Top-K similar images to the query
    4. Extract descriptors in parallel (with caching)
    5. Perform pairwise matching
    6. Expand search on matches (Top-Q)
    7. Build and return the provenance graph
    
    **Request Body:**
    - `images`: List of available images with id, path, and optional label
    - `query_image`: The query image to analyze
    - `k`: Number of top candidates from initial CBIR search (default: 10)
    - `q`: Number of candidates for expansion (default: 5)
    - `max_depth`: Maximum expansion depth (default: 3)
    - `descriptor_type`: Keypoint descriptor type (default: cv_rsift)
    - `max_workers`: Parallel workers for extraction (default: 4)
    
    **Returns:**
    - Provenance graph with nodes and edges
    - Matched pairs with shared area information
    - Processing status and timing
    """
    try:
        service = get_service()
        response = service.analyze(request)
        return response
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Legacy Endpoint (Backward Compatibility)
# ============================================================================

class LegacyImageItem(BaseModel):
    """Legacy image item format."""
    id: str
    path: str
    label: Optional[str] = None


class LegacyAnalysisRequest(BaseModel):
    """Legacy analysis request format."""
    query_image_id: str
    available_images: List[LegacyImageItem]
    k: int = 10
    q: int = 5
    max_depth: int = 3
    descriptor_type: DescriptorType = DescriptorType.CV_RSIFT


@app.post("/analyze/legacy")
async def analyze_provenance_legacy(request: LegacyAnalysisRequest):
    """
    Legacy endpoint for backward compatibility.
    
    Converts old request format to new microservice format.
    """
    # Find query image
    query_img = next(
        (img for img in request.available_images if img.id == request.query_image_id), 
        None
    )
    if not query_img:
        raise HTTPException(
            status_code=404, 
            detail="Query image not found in available images"
        )
    
    # Convert to new format
    new_request = MicroserviceAnalysisRequest(
        images=[
            MicroserviceImageInput(id=img.id, path=img.path, label=img.label)
            for img in request.available_images
            if img.id != request.query_image_id
        ],
        query_image=MicroserviceImageInput(
            id=query_img.id, 
            path=query_img.path, 
            label=query_img.label
        ),
        k=request.k,
        q=request.q,
        max_depth=request.max_depth,
        descriptor_type=request.descriptor_type
    )
    
    return await analyze_provenance(new_request)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """
    Health check endpoint.
    
    Returns service status and CBIR connectivity.
    """
    cbir_connected = False
    
    try:
        # Try to get CBIR client and check connectivity
        if _cbir_client is not None:
            # Try a simple operation
            _cbir_client.check_visibility([])
            cbir_connected = True
    except Exception as e:
        logger.warning(f"CBIR health check failed: {e}")
    
    return HealthCheckResponse(
        status="healthy",
        version="2.0.0",
        cbir_connected=cbir_connected
    )


@app.get("/config")
def get_config():
    """
    Get current service configuration.
    """
    return {
        "output_dir": OUTPUT_DIR,
        "cbir_endpoint": CBIR_ENDPOINT if not USE_MOCK_CBIR else "mock",
        "cbir_user_id": CBIR_USER_ID,
        "use_mock_cbir": USE_MOCK_CBIR,
        "path_mapping": PATH_MAPPING
    }


# ============================================================================
# Index Management Endpoints
# ============================================================================

class IndexImagesRequest(BaseModel):
    """Request to index images in CBIR."""
    images: List[MicroserviceImageInput]
    batch_size: int = 32


@app.post("/index")
async def index_images(request: IndexImagesRequest):
    """
    Index images in the CBIR system.
    
    Useful for pre-indexing images before analysis.
    """
    try:
        service = get_service()
        
        # Convert to dict format for CBIR
        images = [
            {'id': img.id, 'path': img.path, 'label': img.label}
            for img in request.images
        ]
        
        result = service.cbir.index_images(images, batch_size=request.batch_size)
        
        return {
            "success": True,
            "indexed_count": result.get('indexed_count', 0),
            "failed_count": result.get('failed_count', 0),
            "failed_ids": result.get('failed_ids', [])
        }
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CheckVisibilityRequest(BaseModel):
    """Request to check image visibility in CBIR."""
    image_ids: List[str]


@app.post("/visibility")
async def check_visibility(request: CheckVisibilityRequest):
    """
    Check which images are visible in the CBIR system.
    """
    try:
        service = get_service()
        visibility = service.cbir.check_visibility(request.image_ids)
        
        return {
            "visibility": visibility,
            "total": len(request.image_ids),
            "visible_count": sum(1 for v in visibility.values() if v)
        }
    except Exception as e:
        logger.error(f"Visibility check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
