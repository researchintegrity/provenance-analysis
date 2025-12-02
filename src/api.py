from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from pathlib import Path

from .service import ProvenanceService
from .cbir import MockCBIRClient
from .schemas import ProvenanceResponse

app = FastAPI(title="Provenance Analysis Microservice")

# Configuration
OUTPUT_DIR = os.getenv("PROVENANCE_OUTPUT_DIR", "/tmp/provenance_output")
# In a real scenario, we would initialize the real CBIR client here
# For now, we use a Mock with an empty DB (will be populated by request or loaded)
# Assuming the "list of images" provided in the request acts as the "database" for the mock
# OR we assume the service has access to the full dataset.
# Let's assume the request provides the "context" (list of available images).

class ImageItem(BaseModel):
    id: str
    path: str
    label: Optional[str] = None

class AnalysisRequest(BaseModel):
    query_image_id: str
    available_images: List[ImageItem]
    k: int = 10
    q: int = 5

# Global service instance (can be initialized per request if needed)
# We need to initialize it with a CBIR client.
# Since the CBIR client depends on the "available images" in our Mock scenario,
# we will instantiate it inside the endpoint.

@app.post("/analyze", response_model=ProvenanceResponse)
async def analyze_provenance(request: AnalysisRequest):
    """
    Analyze provenance for a query image against a set of available images.
    """
    # 1. Setup Mock CBIR with the provided images
    # Convert Pydantic models to dicts for the Mock Client
    image_db = [img.model_dump() for img in request.available_images]
    cbir_client = MockCBIRClient(image_db)
    
    # 2. Setup Service
    service = ProvenanceService(cbir_client, OUTPUT_DIR)
    
    # 3. Find query image object
    query_img = next((img for img in image_db if img['id'] == request.query_image_id), None)
    if not query_img:
        raise HTTPException(status_code=404, detail="Query image not found in available images")
        
    # 4. Run Analysis
    try:
        response = service.analyze_workflow(query_img, request.k, request.q)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
