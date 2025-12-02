"""
Provenance Analysis - Content Sharing Detection Module

This module provides functionality for detecting shared content between images
using keypoint-based matching and geometric verification.

Microservice Architecture:
- ProvenanceMicroservice: Main service with async descriptor extraction
- DescriptorManager: Parallel descriptor extraction with priority queue
- CBIRClient: Abstract interface for CBIR systems
"""

__version__ = "2.0.0"

from .service import ProvenanceMicroservice, ProvenanceService
from .descriptor_manager import DescriptorManager
from .cbir import CBIRClient, MockCBIRClient, RestCBIRClient
from .schemas import (
    MicroserviceAnalysisRequest,
    MicroserviceAnalysisResponse,
    MicroserviceImageInput,
    DescriptorType,
)

__all__ = [
    "ProvenanceMicroservice",
    "ProvenanceService",
    "DescriptorManager",
    "CBIRClient",
    "MockCBIRClient",
    "RestCBIRClient",
    "MicroserviceAnalysisRequest",
    "MicroserviceAnalysisResponse",
    "MicroserviceImageInput",
    "DescriptorType",
]
