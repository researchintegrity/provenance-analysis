from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Optional
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
    def search(self, query_image_path: str, k: int, 
               filter_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar images.
        
        Args:
            query_image_path: Path to the query image
            k: Number of results to return
            filter_ids: Optional set of image IDs to restrict search to
            
        Returns:
            List of dicts containing 'id', 'path', 'score'
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the CBIR service is alive and healthy.
        
        Returns:
            Dict with 'healthy' (bool) and optional details
        """
        return {"healthy": True, "status": "mock"}

    def index_images(self, images: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """
        Index a list of images in batches.
        
        Args:
            images: List of dicts with 'id', 'path' and optional 'label'
            batch_size: Number of images per batch
            
        Returns:
            Dict with 'indexed_count', 'failed_count', 'failed_ids'
        """
        return {"indexed_count": 0, "failed_count": 0, "failed_ids": []}
    
    def check_visibility(self, images: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Check which images are visible/indexed in the CBIR system.
        
        Args:
            images: List of image dicts with 'id' and 'path'
            
        Returns:
            Dict mapping image_id -> is_visible (bool)
        """
        # Default: all images are visible (for mock/testing)
        return {img['id']: True for img in images}
    
    def get_missing_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get list of images that are not indexed in the CBIR system.
        
        Args:
            images: List of image dicts with 'id', 'path'
            
        Returns:
            List of images that need to be indexed
        """
        visibility = self.check_visibility(images)
        return [img for img in images if not visibility.get(img['id'], False)]

class MockCBIRClient(CBIRClient):
    """Mock CBIR client for testing."""
    
    def __init__(self, image_db: List[Dict[str, Any]] = None):
        self.image_db = image_db or []
        self.indexed_ids: Set[str] = set()
        if image_db:
            self.indexed_ids = {img['id'] for img in image_db}
    
    def health_check(self) -> Dict[str, Any]:
        """Mock always returns healthy."""
        return {"healthy": True, "status": "mock", "indexed_count": len(self.indexed_ids)}
    
    def index_images(self, images: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """Index images in the mock database."""
        indexed = 0
        failed_ids = []
        
        for img in images:
            if img['id'] not in self.indexed_ids:
                self.image_db.append(img)
                self.indexed_ids.add(img['id'])
                indexed += 1
                
        return {
            "indexed_count": indexed,
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids
        }
    
    def check_visibility(self, images: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Check which images are indexed."""
        return {img['id']: img['id'] in self.indexed_ids for img in images}
        
    def search(self, query_image_path: str, k: int,
               filter_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Search for similar images in the mock database."""
        # Filter candidates
        candidates = [img for img in self.image_db if img['path'] != query_image_path]
        
        # Apply filter if provided
        if filter_ids is not None:
            candidates = [img for img in candidates if img['id'] in filter_ids]
        
        # Return random k images
        k = min(k, len(candidates))
        if k == 0:
            return []
            
        results = random.sample(candidates, k)
        
        # Add fake scores
        for i, img in enumerate(results):
            img['score'] = 1.0 - (i * 0.1)  # Decreasing scores
            
        return results

class RestCBIRClient(CBIRClient):
    """REST API implementation of CBIR client."""
    
    def __init__(self, endpoint_url: str, user_id: str, path_mapping: Dict[str, str] = None):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.user_id = user_id
        self.path_mapping = path_mapping
        self._session = requests.Session()  # Reuse connections
        self._visibility_supported = None  # Cache whether /check_visibility is supported

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
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the CBIR service is alive and healthy.
        
        Calls: GET /health
        """
        try:
            response = self._session.get(
                f"{self.endpoint_url}/health",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return {
                "healthy": True,
                "status": data.get("status", "unknown"),
                "model": data.get("model", False),
                "database": data.get("database", False),
                "endpoint": self.endpoint_url
            }
        except requests.exceptions.ConnectionError as e:
            logger.error(f"CBIR service not reachable at {self.endpoint_url}: {e}")
            return {
                "healthy": False,
                "status": "unreachable",
                "error": f"Connection failed: {e}",
                "endpoint": self.endpoint_url
            }
        except Exception as e:
            logger.error(f"CBIR health check failed: {e}")
            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "endpoint": self.endpoint_url
            }
    
    def check_visibility(self, images: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Check which images are visible/indexed in the CBIR system for this user.
        
        Note: If the CBIR system doesn't support /check_visibility endpoint,
        this will assume all images need indexing (return all False).
        
        Args:
            images: List of image dicts with 'id' and 'path'
        
        Calls: POST /check_visibility
        {
            "user_id": "user_123",
            "image_paths": ["/path/to/img1.jpg", "/path/to/img2.jpg", ...]
        }
        
        Returns:
            Dict mapping image_id -> is_visible (bool)
        """
        if not images:
            return {}
            
        # If we already know visibility is not supported, skip the call
        if self._visibility_supported is False:
            logger.debug("Visibility check not supported, assuming all images need indexing")
            return {img['id']: False for img in images}
        
        # Build mapping from remote path to image id
        path_to_id = {}
        remote_paths = []
        for img in images:
            remote_path = self._map_path(img['path'])
            path_to_id[remote_path] = img['id']
            remote_paths.append(remote_path)
        
        try:
            payload = {
                "user_id": self.user_id,
                "image_paths": remote_paths
            }
            response = self._session.post(
                f"{self.endpoint_url}/check_visibility", 
                json=payload,
                timeout=30
            )
            
            # Check if endpoint exists
            if response.status_code == 404:
                logger.info("CBIR /check_visibility endpoint not available. Will index all images.")
                self._visibility_supported = False
                return {img['id']: False for img in images}
            
            response.raise_for_status()
            self._visibility_supported = True
            data = response.json()
            
            # Convert path-based visibility to id-based visibility
            path_visibility = data.get("visibility", {})
            id_visibility = {}
            for remote_path, img_id in path_to_id.items():
                id_visibility[img_id] = path_visibility.get(remote_path, False)
            
            logger.info(f"Visibility check: {data.get('indexed_count', 0)}/{data.get('total_checked', len(images))} images already indexed for user {self.user_id}")
            return id_visibility
            
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.info("CBIR /check_visibility endpoint not available. Will index all images.")
                self._visibility_supported = False
                return {img['id']: False for img in images}
            logger.warning(f"CBIR visibility check failed: {e}. Assuming all images need indexing.")
            return {img['id']: False for img in images}
        except Exception as e:
            logger.warning(f"CBIR visibility check failed: {e}. Assuming all images need indexing.")
            return {img['id']: False for img in images}
        
    def index_images(self, images: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """
        Index images in the CBIR system using batch processing.
        
        Args:
            images: List of dicts with 'path' and optional 'label'
            batch_size: Number of images per batch
        """
        total_indexed = 0
        total_failed = 0
        failed_ids = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            try:
                # Prepare batch payload
                items = []
                for img in batch:
                    items.append({
                        "image_path": self._map_path(img['path']),
                        "labels": [img.get('label', 'unknown')]
                    })
                
                payload = {
                    "user_id": self.user_id,
                    "items": items
                }
                
                # Call batch index endpoint
                response = self._session.post(
                    f"{self.endpoint_url}/index/batch", 
                    json=payload,
                    timeout=120  # Longer timeout for batch operations
                )
                response.raise_for_status()
                
                result = response.json()
                batch_indexed = result.get('indexed_count', 0)
                total_indexed += batch_indexed
                
                # Track failed items if reported
                if 'failed_ids' in result:
                    failed_ids.extend(result['failed_ids'])
                    total_failed += len(result['failed_ids'])
                    
                logger.info(f"Batch {i//batch_size + 1}: indexed {batch_indexed}/{len(batch)} images")
                
            except Exception as e:
                logger.error(f"CBIR batch indexing failed for batch {i//batch_size + 1}: {e}")
                failed_ids.extend([img.get('id', img['path']) for img in batch])
                total_failed += len(batch)
        
        logger.info(f"Total indexing: {total_indexed} indexed, {total_failed} failed")
        return {
            "indexed_count": total_indexed,
            "failed_count": total_failed,
            "failed_ids": failed_ids
        }

    def search(self, query_image_path: str, k: int,
               filter_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar images using the CBIR API.
        
        Args:
            query_image_path: Path to query image
            k: Number of results
            filter_ids: Optional set of image IDs to restrict search to
        
        POST /search
        {
          "user_id": "user_123",
          "image_path": "/workspace/data/query.jpg",
          "top_k": 10,
          "filter_ids": ["img1", "img2"]  # Optional
        }
        """
        try:
            payload = {
                "user_id": self.user_id,
                "image_path": self._map_path(query_image_path),
                "top_k": k+1
            }
            
            # Add filter if provided
            if filter_ids is not None:
                payload["filter_ids"] = list(filter_ids)
            
            response = self._session.post(
                f"{self.endpoint_url}/search", 
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Response format from CBIR system:
            # {'results': [{'id': ..., 'distance': ..., 'image_path': ...}, ...]}
            data = response.json()
            results = data.get('results', [])
            
            # Map to our expected format with deduplication
            mapped_results = []
            seen_ids = set()  # Track seen IDs to deduplicate
            
            k_results=0
            for res in results:
                path = res.get('image_path')
                if not path:
                    continue
                if path == query_image_path:
                    continue  # Skip self-match
                
                
                # Keep path as-is (container path from CBIR)
                # TODO: Here we can use the unmapped path if needed
                container_path = path
                    
                # Create ID from filename
                img_id = Path(container_path).stem
                
                # Skip duplicates - keep only the first (highest scoring) result per ID
                if img_id in seen_ids:
                    continue
                seen_ids.add(img_id)
                
                # Apply client-side filter if CBIR doesn't support it
                if filter_ids is not None and img_id not in filter_ids:
                    continue
                
                mapped_results.append({
                    "id": img_id,
                    "path": container_path,
                    "score": res.get('distance', 0.0),
                    "metadata": res
                })

                # Limit to k results
                k_results += 1
                if k_results >= k:
                    break
            return mapped_results
            
        except Exception as e:
            logger.error(f"CBIR search failed: {e}")
            return []
