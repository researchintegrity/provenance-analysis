"""
Provenance Analysis Microservice.

This module implements the core service logic for provenance analysis
with asynchronous descriptor extraction and optimized pipeline processing.

Pipeline:
1. Image Indexing: Check visibility and index missing images in CBIR
2. Initial CBIR Search: Get Top-K candidates for query
3. Parallel Descriptor Extraction: Extract descriptors asynchronously
4. Matching Pipeline: Match pairs and expand with Top-Q
5. Graph Building: Build provenance graph from matches
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import tempfile

from .cbir import CBIRClient
from .descriptor_manager import DescriptorManager
from .graph_builder import ProvenanceGraphBuilder
from .matcher import match_images
from .schemas import (
    DescriptorType,
    MicroserviceAnalysisRequest,
    MicroserviceAnalysisResponse,
    MicroserviceImageInput,
    ProvenanceGraphResult,
    ProvenanceMatchedPair,
    IndexingStatus,
    ExtractionStatus,
    MatchingStatus,
    ProvenanceResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchingTask:
    """A task for matching two images."""
    source_id: str
    target_id: str
    depth: int
    priority: int  # Lower = higher priority


class ProvenanceMicroservice:
    """
    Provenance Analysis Microservice with optimized async processing.
    
    Key optimizations:
    - Parallel descriptor extraction with priority queue
    - Pre-fetching descriptors for upcoming matches
    - Batch CBIR indexing
    - Efficient queue management to avoid duplicate work
    
    Example usage:
        service = ProvenanceMicroservice(cbir_client)
        response = service.analyze(request)
    """
    
    def __init__(
        self,
        cbir_client: CBIRClient,
        default_output_dir: Optional[str] = None
    ):
        """
        Initialize the microservice.
        
        Args:
            cbir_client: CBIR client for image search
            default_output_dir: Default directory for outputs
        """
        self.cbir = cbir_client
        self.default_output_dir = default_output_dir or tempfile.mkdtemp(prefix="provenance_")
    
    def _apply_path_mapping(self, path: str) -> str:
        """Apply path mapping from CBIR client to convert external paths to internal paths."""
        path_mapping = getattr(self.cbir, 'path_mapping', None)
        if not path_mapping:
            return path
        for local_prefix, remote_prefix in path_mapping.items():
            if path.startswith(local_prefix):
                return path.replace(local_prefix, remote_prefix, 1)
        return path
        
    def analyze(self, request: MicroserviceAnalysisRequest) -> MicroserviceAnalysisResponse:
        """
        Execute the full provenance analysis pipeline.
        
        Args:
            request: Analysis request with images and parameters
            
        Returns:
            Analysis response with graph and status
        """
        start_time = time.time()
        timings: Dict[str, float] = {}  # Profiling timings
        warnings: List[str] = []
        
        # Setup output directory
        output_dir = request.output_dir or self.default_output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build image lookup
        all_images = {img.id: img for img in request.images}
        all_images[request.query_image.id] = request.query_image
        image_ids = set(all_images.keys())
        
        logger.info(f"Starting provenance analysis with {len(all_images)} images")
        
        # ================================================================
        # Step 1: Image Indexing
        # ================================================================
        step_start = time.time()
        indexing_status = self._ensure_images_indexed(
            list(all_images.values()),
            request.cbir_batch_size
        )
        timings['indexing'] = time.time() - step_start
        logger.debug(f"[PROFILE] Step 1 - Indexing: {timings['indexing']:.3f}s")
        
        if indexing_status.failed_to_index > 0:
            warnings.append(
                f"Failed to index {indexing_status.failed_to_index} images: "
                f"{indexing_status.failed_ids[:5]}{'...' if len(indexing_status.failed_ids) > 5 else ''}"
            )
        
        # ================================================================
        # Step 2: Initialize Descriptor Manager
        # ================================================================
        step_start = time.time()
        
        # Get path mapping from CBIR client if available
        path_mapping = getattr(self.cbir, 'path_mapping', None)
        
        descriptor_manager = DescriptorManager(
            output_dir=output_dir,
            max_workers=request.max_workers,
            descriptor_type=request.descriptor_type,
            extract_flip=request.extract_flip,
            path_mapping=path_mapping
        )
        descriptor_manager.start()
        timings['init_descriptor_manager'] = time.time() - step_start
        
        try:
            # ================================================================
            # Step 3: Initial CBIR Search (Top-K)
            # ================================================================
            step_start = time.time()
            query_img = all_images[request.query_image.id]
            
            logger.info(f"Searching Top-{request.k} for query {query_img.id}")
            top_k_candidates = self.cbir.search(
                query_img.path, 
                request.k,
                filter_ids=image_ids  # Restrict to provided images
            )
            timings['cbir_initial_search'] = time.time() - step_start
            logger.debug(f"[PROFILE] Step 3 - Initial CBIR Search: {timings['cbir_initial_search']:.3f}s ({len(top_k_candidates)} candidates)")
            
            if not top_k_candidates:
                return MicroserviceAnalysisResponse(
                    success=False,
                    message="No candidates found in CBIR search",
                    processing_time_seconds=time.time() - start_time,
                    indexing_status=indexing_status,
                    warnings=warnings
                )
            
            # ================================================================
            # Step 4: Initialize Processing Queues
            # ================================================================
            graph_builder = ProvenanceGraphBuilder()
            processed_pairs: Set[Tuple[str, str]] = set()
            queued_pairs: Set[Tuple[str, str]] = set()
            matching_queue: deque[MatchingTask] = deque()
            
            # Add query node
            graph_builder.add_node(
                image_id=query_img.id,
                label=query_img.label or "Query",
                image_path=query_img.path,
                is_query=True
            )
            
            # Submit query for descriptor extraction (highest priority)
            descriptor_manager.submit(
                query_img.id, 
                query_img.path, 
                priority=0
            )
            
            # Process Top-K candidates
            priority = 1
            for cand in top_k_candidates:
                # Get image info (may be from CBIR or from our list)
                cand_img = all_images.get(cand['id'])
                if cand_img is None:
                    # CBIR returned an image not in our list, use CBIR data
                    cand_img = MicroserviceImageInput(
                        id=cand['id'],
                        path=cand['path'],
                        label=cand.get('label')
                    )
                    all_images[cand['id']] = cand_img
                
                # Add to graph
                graph_builder.add_node(
                    cand_img.id,
                    cand_img.label or cand_img.id,
                    cand_img.path
                )
                
                # Queue for matching
                pair_key = tuple(sorted([query_img.id, cand_img.id]))
                if pair_key not in queued_pairs:
                    matching_queue.append(MatchingTask(
                        source_id=query_img.id,
                        target_id=cand_img.id,
                        depth=1,
                        priority=priority
                    ))
                    queued_pairs.add(pair_key)
                    priority += 1
                
                # Submit for descriptor extraction
                descriptor_manager.submit(
                    cand_img.id,
                    cand_img.path,
                    priority=priority
                )
            
            # ================================================================
            # Step 5: Matching Pipeline
            # ================================================================
            step_start = time.time()
            extraction_stats = {"extracted": 0, "from_cache": 0, "failed": 0}
            expansion_count = 0
            total_descriptor_wait_time = 0.0
            total_matching_time = 0.0
            total_cbir_expansion_time = 0.0
            pairs_matched = 0
            
            while matching_queue:
                task = matching_queue.popleft()
                
                # Skip if already processed
                pair_key = tuple(sorted([task.source_id, task.target_id]))
                if pair_key in processed_pairs:
                    continue
                if task.source_id == task.target_id:
                    continue
                    
                processed_pairs.add(pair_key)
                
                # Get images
                source_img = all_images.get(task.source_id)
                target_img = all_images.get(task.target_id)
                
                if source_img is None or target_img is None:
                    warnings.append(f"Missing image for pair {task.source_id}-{task.target_id}")
                    continue
                
                # Wait for descriptors
                logger.debug(f"Matching {task.source_id} vs {task.target_id} (depth={task.depth})")
                
                desc_wait_start = time.time()
                desc1 = descriptor_manager.get_descriptors(task.source_id, timeout=120.0)
                desc2 = descriptor_manager.get_descriptors(task.target_id, timeout=120.0)
                total_descriptor_wait_time += time.time() - desc_wait_start
                
                if desc1 is None or desc2 is None:
                    extraction_stats["failed"] += 1
                    warnings.append(f"Failed to get descriptors for pair {task.source_id}-{task.target_id}")
                    continue
                
                # Perform matching (apply path mapping for image loading)
                try:
                    match_start = time.time()
                    match_result = match_images(
                        image1_path=self._apply_path_mapping(source_img.path),
                        keypoints1=desc1['keypoints'],
                        descriptors1=desc1['descriptors'],
                        image2_path=self._apply_path_mapping(target_img.path),
                        keypoints2=desc2['keypoints'],
                        descriptors2=desc2['descriptors'],
                        flip_keypoints1=desc1.get('flip_keypoints'),
                        flip_descriptors1=desc1.get('flip_descriptors')
                    )
                    total_matching_time += time.time() - match_start
                    pairs_matched += 1
                    
                    if match_result['is_match']:
                        logger.debug(f"Match found: {task.source_id} <-> {task.target_id}")
                        
                        # Add edge to graph
                        graph_builder.add_match(
                            img1_id=task.source_id,
                            img2_id=task.target_id,
                            shared_area_img1=match_result['shared_area_img1'],
                            shared_area_img2=match_result['shared_area_img2'],
                            matched_keypoints=match_result['matched_keypoints'],
                            is_flipped=match_result.get('is_flipped_match', False)
                        )
                        
                        # Expansion (Top-Q) if not at max depth
                        if task.depth < request.max_depth:
                            logger.debug(f"Expanding search from {task.target_id} (Top-{request.q})")
                            expansion_count += 1
                            
                            expansion_start = time.time()
                            top_q_candidates = self.cbir.search(
                                target_img.path,
                                request.q,
                                filter_ids=image_ids
                            )
                            total_cbir_expansion_time += time.time() - expansion_start
                            
                            for sub_cand in top_q_candidates:
                                sub_cand_id = sub_cand['id']
                                
                                # Skip if same as source
                                if sub_cand_id == task.source_id:
                                    continue
                                
                                # Get or create image info
                                sub_cand_img = all_images.get(sub_cand_id)
                                if sub_cand_img is None:
                                    sub_cand_img = MicroserviceImageInput(
                                        id=sub_cand_id,
                                        path=sub_cand['path'],
                                        label=sub_cand.get('label')
                                    )
                                    all_images[sub_cand_id] = sub_cand_img
                                
                                # Add to graph
                                graph_builder.add_node(
                                    sub_cand_img.id,
                                    sub_cand_img.label or sub_cand_img.id,
                                    sub_cand_img.path
                                )
                                
                                # Queue for matching if not already
                                new_pair_key = tuple(sorted([task.target_id, sub_cand_id]))
                                if new_pair_key not in queued_pairs and new_pair_key not in processed_pairs:
                                    priority += 1
                                    matching_queue.append(MatchingTask(
                                        source_id=task.target_id,
                                        target_id=sub_cand_id,
                                        depth=task.depth + 1,
                                        priority=priority
                                    ))
                                    queued_pairs.add(new_pair_key)
                                    
                                    # Pre-fetch descriptors
                                    descriptor_manager.submit(
                                        sub_cand_img.id,
                                        sub_cand_img.path,
                                        priority=priority + 100  # Lower priority than current matches
                                    )
                    
                except Exception as e:
                    logger.error(f"Error matching {task.source_id} vs {task.target_id}: {e}")
                    warnings.append(f"Match error for {task.source_id}-{task.target_id}: {str(e)}")
            
            # Log matching pipeline profiling
            timings['matching_pipeline_total'] = time.time() - step_start
            timings['descriptor_wait'] = total_descriptor_wait_time
            timings['matching_compute'] = total_matching_time
            timings['cbir_expansion'] = total_cbir_expansion_time
            
            logger.debug(f"[PROFILE] Step 5 - Matching Pipeline: {timings['matching_pipeline_total']:.3f}s")
            logger.debug(f"[PROFILE]   - Descriptor wait: {timings['descriptor_wait']:.3f}s")
            logger.debug(f"[PROFILE]   - Matching compute: {timings['matching_compute']:.3f}s ({pairs_matched} pairs, {timings['matching_compute']/max(pairs_matched,1)*1000:.1f}ms/pair)")
            logger.debug(f"[PROFILE]   - CBIR expansion: {timings['cbir_expansion']:.3f}s ({expansion_count} expansions)")
            
            # ================================================================
            # Step 6: Build Graph
            # ================================================================
            step_start = time.time()
            graph_data = graph_builder.build_graph()
            timings['graph_building'] = time.time() - step_start
            logger.debug(f"[PROFILE] Step 6 - Graph Building: {timings['graph_building']:.3f}s")
            
            graph_result = ProvenanceGraphResult(
                nodes=graph_data['nodes'],
                edges=graph_data['edges'],
                spanning_tree_edges=graph_data['spanning_tree_edges'],
                connected_components=graph_data['connected_components'],
                adjacency_matrix=graph_data['adjacency_matrix']
            )
            
            # Build matched pairs list
            matched_pairs = [
                ProvenanceMatchedPair(
                    image1_id=mp['image1_id'],
                    image2_id=mp['image2_id'],
                    shared_area_img1=mp['shared_area_img1'],
                    shared_area_img2=mp['shared_area_img2'],
                    matched_keypoints=mp['matched_keypoints'],
                    is_flipped=mp.get('is_flipped', False)
                )
                for mp in graph_builder.matched_pairs
            ]
            
            # Get extraction stats
            dm_stats = descriptor_manager.get_stats()
            extraction_status = ExtractionStatus(
                total_images=dm_stats['total_submitted'],
                extracted=dm_stats['completed'],
                from_cache=dm_stats.get('from_cache', 0),
                failed=dm_stats['failed']
            )
            
            matching_status = MatchingStatus(
                total_pairs_checked=len(processed_pairs),
                matched_pairs=len(graph_builder.matched_pairs),
                expansion_count=expansion_count
            )
            
            processing_time = time.time() - start_time
            
            # Final profiling summary
            logger.debug(f"[PROFILE] ========== SUMMARY ==========")
            logger.debug(f"[PROFILE] Total time: {processing_time:.3f}s")
            logger.debug(f"[PROFILE] Indexing: {timings.get('indexing', 0):.3f}s ({timings.get('indexing', 0)/processing_time*100:.1f}%)")
            logger.debug(f"[PROFILE] Initial CBIR: {timings.get('cbir_initial_search', 0):.3f}s ({timings.get('cbir_initial_search', 0)/processing_time*100:.1f}%)")
            logger.debug(f"[PROFILE] Matching Pipeline: {timings.get('matching_pipeline_total', 0):.3f}s ({timings.get('matching_pipeline_total', 0)/processing_time*100:.1f}%)")
            logger.debug(f"[PROFILE]   - Descriptor wait: {timings.get('descriptor_wait', 0):.3f}s")
            logger.debug(f"[PROFILE]   - Matching compute: {timings.get('matching_compute', 0):.3f}s")
            logger.debug(f"[PROFILE]   - CBIR expansion: {timings.get('cbir_expansion', 0):.3f}s")
            logger.debug(f"[PROFILE] Graph Building: {timings.get('graph_building', 0):.3f}s ({timings.get('graph_building', 0)/processing_time*100:.1f}%)")
            logger.debug(f"[PROFILE] ==============================")
            
            logger.info(
                f"Analysis complete: {len(graph_builder.nodes)} nodes, "
                f"{len(graph_builder.matched_pairs)} matches, "
                f"{processing_time:.2f}s"
            )
            
            return MicroserviceAnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                processing_time_seconds=processing_time,
                indexing_status=indexing_status,
                extraction_status=extraction_status,
                matching_status=matching_status,
                graph=graph_result,
                matched_pairs=matched_pairs,
                warnings=warnings
            )
            
        finally:
            # Always shutdown the descriptor manager
            descriptor_manager.shutdown(wait=False)
    
    def _ensure_images_indexed(
        self,
        images: List[MicroserviceImageInput],
        batch_size: int
    ) -> IndexingStatus:
        """
        Ensure all images are indexed in the CBIR system.
        
        Args:
            images: List of images to check/index
            batch_size: Batch size for indexing
            
        Returns:
            Indexing status
        """
        # Convert to dict format for CBIR client
        image_dicts = [{'id': img.id, 'path': img.path, 'label': img.label} for img in images]
        
        # Check visibility (now passes full image objects with paths)
        logger.info(f"Checking visibility for {len(images)} images")
        visibility = self.cbir.check_visibility(image_dicts)
        
        # Separate indexed and missing
        already_indexed = sum(1 for v in visibility.values() if v)
        missing_images = [
            img_dict for img_dict in image_dicts
            if not visibility.get(img_dict['id'], False)
        ]
        
        logger.info(f"Found {already_indexed} indexed, {len(missing_images)} need indexing")
        
        # Index missing images
        newly_indexed = 0
        failed_ids = []
        
        if missing_images:
            result = self.cbir.index_images(missing_images, batch_size=batch_size)
            newly_indexed = result.get('indexed_count', 0)
            failed_ids = result.get('failed_ids', [])
        
        return IndexingStatus(
            total_images=len(images),
            already_indexed=already_indexed,
            newly_indexed=newly_indexed,
            failed_to_index=len(failed_ids),
            failed_ids=failed_ids
        )


# Legacy compatibility - keep old class name working
class ProvenanceService(ProvenanceMicroservice):
    """Legacy alias for ProvenanceMicroservice."""
    
    def __init__(self, cbir_client: CBIRClient, output_dir: str):
        super().__init__(cbir_client, output_dir)
    
    def analyze_workflow(
        self,
        query_image: Dict[str, Any],
        k: int,
        q: int,
        max_depth: float = float('inf'),
        descriptor: DescriptorType = DescriptorType.CV_RSIFT
    ) -> ProvenanceResponse:
        """
        Legacy method for backward compatibility.
        Converts old-style call to new microservice request.
        """
        # Convert to new format
        request = MicroserviceAnalysisRequest(
            images=[],  # Will be populated by CBIR
            query_image=MicroserviceImageInput(
                id=query_image['id'],
                path=query_image['path'],
                label=query_image.get('label', 'Query')
            ),
            k=k,
            q=q,
            max_depth=int(max_depth) if max_depth != float('inf') else 10,
            descriptor_type=descriptor
        )
        
        # Run new pipeline
        response = self.analyze(request)
        
        # Convert to old response format
        return ProvenanceResponse(
            success=response.success,
            message=response.message,
            total_images=len(response.graph.nodes) if response.graph else 0,
            total_pairs_checked=response.matching_status.total_pairs_checked if response.matching_status else 0,
            matched_pairs_count=response.matching_status.matched_pairs if response.matching_status else 0,
            processing_time_seconds=response.processing_time_seconds,
            graph=response.graph
        )
