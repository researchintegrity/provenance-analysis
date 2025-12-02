from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random
import cv2
import numpy as np
import requests
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CBIRClient(ABC):
    """Abstract base class for CBIR system integration."""
    
    @abstractmethod
    def search(self, query_image_path: str, k: int) -> List[Dict[str, Any]]:
        """
        Search for similar images.
        
        Args:
            query_image_path: Path to the query image
            k: Number of results to return
            
        Returns:
            List of dicts containing 'id', 'path', 'score'
        """
        pass

    def index_images(self, images: List[Dict[str, Any]]):
        """
        Index a list of images (optional, for systems that need it).
        """
        pass

class MockCBIRClient(CBIRClient):
    """Mock CBIR client for testing."""
    
    def __init__(self, image_db: List[Dict[str, Any]]):
        self.image_db = image_db
    
    def index_images(self, images: List[Dict[str, Any]]):
        self.image_db = images
        
    def search(self, query_image_path: str, k: int) -> List[Dict[str, Any]]:
        # In a real system, this would call an external API
        # Here we just return random images from our "database" excluding the query
        candidates = [img for img in self.image_db if img['path'] != query_image_path]
        
        # Return random k images
        k = min(k, len(candidates))
        results = random.sample(candidates, k)
        
        # Add fake scores
        for img in results:
            img['score'] = random.random()
            
        return results

class RestCBIRClient(CBIRClient):
    """REST API implementation of CBIR client."""
    
    def __init__(self, endpoint_url: str, user_id: str, path_mapping: Dict[str, str] = None):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.user_id = user_id
        self.path_mapping = path_mapping

    def _map_path(self, path: str) -> str:
        if not self.path_mapping:
            return path
        for local, remote in self.path_mapping.items():
            if path.startswith(local):
                return path.replace(local, remote, 1)
        return path

    def _unmap_path(self, path: str) -> str:
        if not self.path_mapping:
            return path
        for local, remote in self.path_mapping.items():
            if path.startswith(remote):
                return path.replace(remote, local, 1)
        return path
        
    def index_images(self, images: List[Dict[str, Any]]):
        """
        Index images in the CBIR system using batch processing.
        
        Args:
            images: List of dicts with 'path' and optional 'label'
        """
        try:
            # Prepare batch payload
            items = []
            for img in images:
                items.append({
                    "image_path": self._map_path(img['path']),
                    "labels": [img.get('label', 'unknown')] # Optional labels
                })
            
            payload = {
                "user_id": self.user_id,
                "items": items
            }
            
            # Call batch index endpoint
            response = requests.post(f"{self.endpoint_url}/index/batch", json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Batch indexing completed: {result.get('indexed_count', 0)}/{len(images)} images indexed successfully")
            
        except Exception as e:
            logger.error(f"CBIR batch indexing failed: {e}")

    def search(self, query_image_path: str, k: int) -> List[Dict[str, Any]]:
        """
        Search for similar images using the CBIR API.
        
        POST /search
        {
          "user_id": "user_123",
          "image_path": "/workspace/data/query.jpg",
          "top_k": 10,
          "labels": ["Western Blot"]
        }
        """
        try:
            payload = {
                "user_id": self.user_id,
                "image_path": self._map_path(query_image_path),
                "top_k": k
                # "labels": [] # We search all labels for provenance
            }
            
            response = requests.post(f"{self.endpoint_url}/search", json=payload)
            response.raise_for_status()
            
            # Response format from CBIR system:
            # {'results': [{'id': ..., 'distance': ..., 'image_path': ...}, ...]}
            data = response.json()
            results = data.get('results', [])
            
            # Map to our expected format
            mapped_results = []
            for res in results:
                # Assuming result has 'image_path' and 'distance' (which seems to be similarity)
                path = res.get('image_path')
                if not path:
                    continue
                
                # Unmap path
                local_path = self._unmap_path(path)
                    
                # Create a simple ID from filename if not provided
                # The ID returned by Milvus is an integer, we prefer the filename-based ID
                img_id = Path(local_path).stem
                
                mapped_results.append({
                    "id": img_id,
                    "path": local_path,
                    "score": res.get('distance', 0.0),
                    "metadata": res
                })
            
            return mapped_results
            
        except Exception as e:
            logger.error(f"CBIR search failed: {e}")
            return []
