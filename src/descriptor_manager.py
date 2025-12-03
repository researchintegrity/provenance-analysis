"""
Asynchronous descriptor extraction manager for provenance analysis.

This module provides:
- Async/parallel descriptor extraction with ThreadPoolExecutor
- Priority queue for extraction ordering
- Thread-safe caching of computed descriptors
- Progress tracking and status reporting
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
import time

from .schemas import DescriptorType
from .keypoint_extractor import extract_descriptors, load_descriptors, save_descriptors

logger = logging.getLogger(__name__)


class ExtractionStatus(str, Enum):
    """Status of descriptor extraction for an image."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(order=True)
class ExtractionTask:
    """A descriptor extraction task with priority ordering."""
    priority: int  # Lower = higher priority
    image_id: str = field(compare=False)
    image_path: str = field(compare=False)
    descriptor_type: DescriptorType = field(compare=False)
    submitted_at: float = field(default_factory=time.time, compare=False)


class DescriptorManager:
    """
    Manages asynchronous descriptor extraction with caching and prioritization.
    
    Features:
    - Thread pool for parallel extraction
    - Priority-based task ordering
    - Thread-safe descriptor cache
    - Disk persistence for descriptors
    - Status tracking per image
    
    Usage:
        manager = DescriptorManager(output_dir="/tmp/descriptors", max_workers=4)
        manager.start()
        
        # Submit extraction tasks (higher priority = lower number)
        manager.submit(image_id="img1", image_path="/path/to/img1.jpg", priority=1)
        manager.submit(image_id="img2", image_path="/path/to/img2.jpg", priority=2)
        
        # Wait for specific descriptor
        desc = manager.get_descriptors("img1", timeout=30.0)
        
        # Shutdown when done
        manager.shutdown()
    """
    
    def __init__(
        self,
        output_dir: str,
        max_workers: int = 4,
        descriptor_type: DescriptorType = DescriptorType.CV_RSIFT,
        extract_flip: bool = True,
        path_mapping: Optional[Dict[str, str]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.descriptor_type = descriptor_type
        self.extract_flip = extract_flip
        self.path_mapping = path_mapping or {}  # local_prefix -> container_prefix
        
        # Thread-safe structures
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}  # image_id -> descriptors
        self._status: Dict[str, ExtractionStatus] = {}  # image_id -> status
        self._futures: Dict[str, Future] = {}  # image_id -> future
        self._events: Dict[str, threading.Event] = {}  # image_id -> completion event
        
        # Task management
        self._pending_tasks: Dict[str, ExtractionTask] = {}  # Tasks waiting to be submitted
        self._submitted_ids: Set[str] = set()  # Images already submitted
        
        # Executor
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        
    def start(self):
        """Start the descriptor extraction service."""
        with self._lock:
            if self._running:
                return
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._running = True
            logger.info(f"DescriptorManager started with {self.max_workers} workers")
    
    def shutdown(self, wait: bool = True):
        """Shutdown the descriptor extraction service."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
            
        logger.info("DescriptorManager shutdown complete")
    
    def _apply_path_mapping(self, path: str) -> str:
        """
        Apply path mapping to convert external paths to internal container paths.
        
        If path_mapping is {'/host/data': '/container/data'} and path is
        '/host/data/images/img.png', returns '/container/data/images/img.png'.
        """
        for local_prefix, remote_prefix in self.path_mapping.items():
            if path.startswith(local_prefix):
                mapped = path.replace(local_prefix, remote_prefix, 1)
                logger.debug(f"Path mapping: {path} -> {mapped}")
                return mapped
        return path
    
    def _get_descriptor_paths(self, image_id: str) -> Dict[str, str]:
        """Get the file paths for an image's descriptors."""
        base = self.output_dir / f"{image_id}_{self.descriptor_type.value}"
        return {
            'keypoints': f"{base}_kps.npy",
            'descriptors': f"{base}_desc.npy",
            'flip_keypoints': f"{base}_flip_kps.npy",
            'flip_descriptors': f"{base}_flip_desc.npy"
        }
    
    def _try_load_from_disk(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Try to load descriptors from disk cache."""
        paths = self._get_descriptor_paths(image_id)
        
        kp_path = paths['keypoints']
        desc_path = paths['descriptors']
        
        if Path(kp_path).exists() and Path(desc_path).exists():
            try:
                flip_kp = paths['flip_keypoints'] if Path(paths['flip_keypoints']).exists() else None
                flip_desc = paths['flip_descriptors'] if Path(paths['flip_descriptors']).exists() else None
                
                result = load_descriptors(kp_path, desc_path, flip_kp, flip_desc)
                logger.debug(f"Loaded descriptors from disk for {image_id}")
                return result
            except Exception as e:
                logger.warning(f"Failed to load descriptors from disk for {image_id}: {e}")
        
        return None
    
    def _extract_and_cache(self, task: ExtractionTask) -> Dict[str, Any]:
        """
        Extract descriptors for an image and cache them.
        This runs in a worker thread.
        """
        image_id = task.image_id
        
        try:
            # Update status
            with self._lock:
                self._status[image_id] = ExtractionStatus.IN_PROGRESS
            
            # Check disk cache first
            cached = self._try_load_from_disk(image_id)
            if cached is not None:
                with self._lock:
                    self._cache[image_id] = cached
                    self._status[image_id] = ExtractionStatus.COMPLETED
                    if image_id in self._events:
                        self._events[image_id].set()
                return cached
            
            # Apply path mapping to get the actual file path
            actual_path = task.image_path if Path(task.image_path).exists() else self._apply_path_mapping(task.image_path)
            
            # Extract descriptors
            logger.info(f"Extracting descriptors for {image_id} using {task.descriptor_type.value}")
            result = extract_descriptors(
                actual_path, 
                descriptor_type=task.descriptor_type,
                extract_flip=self.extract_flip
            )
            
            # Save to disk
            save_descriptors(
                result, 
                str(self.output_dir), 
                f"{image_id}_{task.descriptor_type.value}"
            )
            
            # Update cache and status
            with self._lock:
                self._cache[image_id] = result
                self._status[image_id] = ExtractionStatus.COMPLETED
                if image_id in self._events:
                    self._events[image_id].set()
            
            logger.info(f"Completed descriptor extraction for {image_id}: {len(result['keypoints'])} keypoints")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract descriptors for {image_id}: {e}")
            with self._lock:
                self._status[image_id] = ExtractionStatus.FAILED
                if image_id in self._events:
                    self._events[image_id].set()
            raise
    
    def submit(
        self, 
        image_id: str, 
        image_path: str, 
        priority: int = 100,
        descriptor_type: Optional[DescriptorType] = None
    ) -> bool:
        """
        Submit an image for descriptor extraction.
        
        Args:
            image_id: Unique identifier for the image
            image_path: Path to the image file
            priority: Extraction priority (lower = higher priority)
            descriptor_type: Override default descriptor type
            
        Returns:
            True if task was submitted, False if already submitted/completed
        """
        with self._lock:
            # Already completed or in progress?
            if image_id in self._cache:
                return False
            if image_id in self._submitted_ids:
                return False
            
            # Check if we can load from disk immediately
            cached = self._try_load_from_disk(image_id)
            if cached is not None:
                self._cache[image_id] = cached
                self._status[image_id] = ExtractionStatus.COMPLETED
                self._submitted_ids.add(image_id)
                return False  # No extraction needed
            
            # Create task
            task = ExtractionTask(
                priority=priority,
                image_id=image_id,
                image_path=image_path,
                descriptor_type=descriptor_type or self.descriptor_type
            )
            
            # Create completion event
            self._events[image_id] = threading.Event()
            self._status[image_id] = ExtractionStatus.PENDING
            self._submitted_ids.add(image_id)
            
            # Submit to executor
            if self._executor and self._running:
                future = self._executor.submit(self._extract_and_cache, task)
                self._futures[image_id] = future
                logger.debug(f"Submitted extraction task for {image_id} with priority {priority}")
                return True
            else:
                logger.warning(f"DescriptorManager not running, task for {image_id} not submitted")
                return False
    
    def submit_batch(
        self, 
        images: List[Dict[str, Any]], 
        base_priority: int = 100
    ) -> int:
        """
        Submit multiple images for extraction with sequential priorities.
        
        Args:
            images: List of dicts with 'id', 'path' keys
            base_priority: Starting priority (increments for each image)
            
        Returns:
            Number of tasks actually submitted
        """
        submitted = 0
        for i, img in enumerate(images):
            if self.submit(img['id'], img['path'], priority=base_priority + i):
                submitted += 1
        return submitted
    
    def get_status(self, image_id: str) -> ExtractionStatus:
        """Get the extraction status for an image."""
        with self._lock:
            return self._status.get(image_id, ExtractionStatus.PENDING)
    
    def is_ready(self, image_id: str) -> bool:
        """Check if descriptors are ready for an image."""
        with self._lock:
            return image_id in self._cache
    
    def get_descriptors(
        self, 
        image_id: str, 
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get descriptors for an image, waiting if necessary.
        
        Args:
            image_id: Image identifier
            timeout: Maximum time to wait (None = wait forever)
            
        Returns:
            Descriptor dict or None if failed/timeout
        """
        # Check cache first
        with self._lock:
            if image_id in self._cache:
                return self._cache[image_id]
            
            # Get the event to wait on
            event = self._events.get(image_id)
        
        if event is None:
            logger.warning(f"No extraction task found for {image_id}")
            return None
        
        # Wait for completion
        if not event.wait(timeout=timeout):
            logger.warning(f"Timeout waiting for descriptors of {image_id}")
            return None
        
        # Return from cache
        with self._lock:
            return self._cache.get(image_id)
    
    def wait_for_all(
        self, 
        image_ids: List[str], 
        timeout: Optional[float] = None
    ) -> Dict[str, bool]:
        """
        Wait for multiple images to complete extraction.
        
        Args:
            image_ids: List of image IDs to wait for
            timeout: Maximum total time to wait
            
        Returns:
            Dict mapping image_id -> success (True if completed)
        """
        results = {}
        start_time = time.time()
        
        for image_id in image_ids:
            remaining = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = max(0, timeout - elapsed)
                if remaining <= 0:
                    results[image_id] = self.is_ready(image_id)
                    continue
            
            desc = self.get_descriptors(image_id, timeout=remaining)
            results[image_id] = desc is not None
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the manager's current state."""
        with self._lock:
            status_counts = {}
            for status in ExtractionStatus:
                status_counts[status.value] = sum(
                    1 for s in self._status.values() if s == status
                )
            
            return {
                "total_submitted": len(self._submitted_ids),
                "cached": len(self._cache),
                "pending": status_counts.get(ExtractionStatus.PENDING.value, 0),
                "in_progress": status_counts.get(ExtractionStatus.IN_PROGRESS.value, 0),
                "completed": status_counts.get(ExtractionStatus.COMPLETED.value, 0),
                "failed": status_counts.get(ExtractionStatus.FAILED.value, 0),
            }
    
    def clear_cache(self):
        """Clear the in-memory cache (disk cache remains)."""
        with self._lock:
            self._cache.clear()
            self._status.clear()
            self._events.clear()
            self._submitted_ids.clear()
            self._futures.clear()
