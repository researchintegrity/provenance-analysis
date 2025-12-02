import logging
from typing import List, Dict, Any, Set
from pathlib import Path
import os

from .schemas import ProvenanceRequest, ProvenanceResponse, ProvenanceGraphResult, DescriptorType
from .keypoint_extractor import extract_descriptors, load_descriptors, save_descriptors
from .matcher import match_images
from .graph_builder import ProvenanceGraphBuilder
from .cbir import CBIRClient

logger = logging.getLogger(__name__)

class ProvenanceService:
    def __init__(self, cbir_client: CBIRClient, output_dir: str):
        self.cbir = cbir_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.descriptors_cache: Dict[str, Dict[str, Any]] = {}

    def _get_descriptors(self, image_path: str, image_id: str, descriptor_type: DescriptorType):
        """Get descriptors from cache, disk, or extract them."""
        if image_id in self.descriptors_cache:
            return self.descriptors_cache[image_id]

        # Try loading from disk
        desc_base = self.output_dir / f"{image_id}_{descriptor_type.value}"
        kp_path = f"{desc_base}_kps.npy"
        desc_path = f"{desc_base}_desc.npy"
        flip_kp_path = f"{desc_base}_flip_kps.npy"
        flip_desc_path = f"{desc_base}_flip_desc.npy"
        
        if Path(kp_path).exists() and Path(desc_path).exists():
            # Check if flipped versions exist
            fkp = flip_kp_path if Path(flip_kp_path).exists() else None
            fdesc = flip_desc_path if Path(flip_desc_path).exists() else None
            
            result = load_descriptors(kp_path, desc_path, fkp, fdesc)
        else:
            # Extract
            result = extract_descriptors(image_path, descriptor_type=descriptor_type, extract_flip=True)
            # Save
            save_descriptors(result, str(self.output_dir), f"{image_id}_{descriptor_type.value}")
            
        self.descriptors_cache[image_id] = result
        return result

    def analyze_workflow(self, 
                         query_image: Dict[str, Any],
                         k: int,
                         q: int,
                         max_depth: float = float('inf'),
                         descriptor: DescriptorType = DescriptorType.CV_RSIFT) -> ProvenanceResponse:
        """
        Execute the provenance analysis workflow:
        1. Query -> CBIR -> Top-K
        2. Match Query vs Top-K
        3. If Match -> CBIR (on match) -> Top-Q
        4. Match Match vs Top-Q
        5. Build Graph & MST
        
        Args:
            query_image: Query image dict with 'id', 'path', 'label'
            k: Number of top candidates from initial CBIR search
            q: Number of top candidates from expansion searches
            max_depth: Maximum depth for expansion (default: inf for unlimited)
        """
        graph_builder = ProvenanceGraphBuilder()
        processed_pairs = set() # Track pairs that have been processed
        queued_pairs = set()  # Track pairs that are queued for processing
        
        # 1. Add Query Node
        graph_builder.add_node(
            image_id=query_image['id'],
            label=query_image.get('label', 'Query'),
            image_path=query_image['path'],
            is_query=True
        )
        
        # 2. Get Top-K candidates
        logger.info(f"Searching Top-{k} for query {query_image['id']}")
        top_k_candidates = self.cbir.search(query_image['path'], k)
        
        candidates_to_process = [] # List of (Source_Image_Dict, Candidate_Image_Dict, depth)
        
        for cand in top_k_candidates:
            candidates_to_process.append((query_image, cand, 1))  # depth 1
            graph_builder.add_node(cand['id'], cand.get('label', cand['id']), cand['path'])
            # Mark this pair as queued
            queued_pairs.add(tuple(sorted([query_image['id'], cand['id']])))

        # 3. Process Candidates
        # We use a queue to handle the "Expansion" (Top-Q) logic
        
        while candidates_to_process:
            source_img, target_img, depth = candidates_to_process.pop(0)
            if source_img['id'] == target_img['id']:
               continue
            
            pair_key = tuple(sorted([source_img['id'], target_img['id']]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Match
            logger.info(f"Matching {source_img['id']} vs {target_img['id']}")
            try:
                desc1 = self._get_descriptors(source_img['path'], source_img['id'], descriptor)
                desc2 = self._get_descriptors(target_img['path'], target_img['id'], descriptor)
                
                match_result = match_images(
                    image1_path=source_img['path'],
                    keypoints1=desc1['keypoints'],
                    descriptors1=desc1['descriptors'],
                    image2_path=target_img['path'],
                    keypoints2=desc2['keypoints'],
                    descriptors2=desc2['descriptors'],
                    flip_keypoints1=desc1.get('flip_keypoints'),
                    flip_descriptors1=desc1.get('flip_descriptors')
                )
                
                if match_result['is_match']:
                    logger.info(f"Match found! Expanding search for {target_img['id']} (Top-{q})")
                    
                    # Add edge to graph
                    graph_builder.add_match(
                        img1_id=source_img['id'],
                        img2_id=target_img['id'],
                        shared_area_img1=match_result['shared_area_img1'],
                        shared_area_img2=match_result['shared_area_img2'],
                        matched_keypoints=match_result['matched_keypoints']
                    )
                    
                    # 4. Expansion: Get Top-Q for the MATCHED image
                    # We only expand if we haven't expanded this node before? 
                    # For simplicity, let's just get neighbors and add them to the queue
                    # checking against the MATCHED image (chaining)
                    
                    # Note: In a real recursion we might want to limit depth. 
                    # Here we just do one level of expansion as requested (Top-Q of the matching image).
                    
                    # Check if this was already an expansion (to avoid infinite loops if A matches B and B matches A)
                    # The prompt implies: Query -> Top-K -> Match -> Top-Q. It doesn't explicitly say "recurse forever".
                    # I will assume 1 level of expansion for now.
                    
                    if depth < max_depth:  # Only expand if we haven't reached max depth
                        top_q_candidates = self.cbir.search(target_img['path'], q)
                        for sub_cand in top_q_candidates:
                            if sub_cand['id'] != source_img['id']: # Don't go back to query
                                graph_builder.add_node(sub_cand['id'], sub_cand.get('label', sub_cand['id']), sub_cand['path'])
                                
                                # Only add to queue if not already queued or processed
                                pair_key = tuple(sorted([target_img['id'], sub_cand['id']]))
                                if pair_key not in queued_pairs and pair_key not in processed_pairs:
                                    candidates_to_process.append((target_img, sub_cand, depth + 1))
                                    queued_pairs.add(pair_key)

            except Exception as e:
                logger.error(f"Error matching {source_img['id']} vs {target_img['id']}: {e}")

        # 5. Build Graph
        graph_data = graph_builder.build_graph()
        
        graph_result = ProvenanceGraphResult(
            nodes=graph_data['nodes'],
            edges=graph_data['edges'],
            spanning_tree_edges=graph_data['spanning_tree_edges'],
            connected_components=graph_data['connected_components'],
            adjacency_matrix=graph_data['adjacency_matrix']
        )

        return ProvenanceResponse(
            success=True,
            message="Analysis complete",
            total_images=len(graph_builder.nodes),
            total_pairs_checked=len(processed_pairs),
            matched_pairs_count=len(graph_builder.matched_pairs),
            processing_time_seconds=0.0,
            graph=graph_result
        )
