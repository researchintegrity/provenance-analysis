"""
Main entry point for the provenance analysis container.

Supports four commands:
- extract: Extract keypoint descriptors from an image
- match: Match two images and compute shared content area
- batch: Batch match multiple image pairs
- provenance: Full provenance analysis with graph building
"""

import sys
import json
import logging
import time
import click
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schemas import (
    CommandType,
    ContainerInput,
    ContainerOutput,
    ExtractDescriptorsRequest,
    ExtractDescriptorsResponse,
    PairwiseMatchRequest,
    PairwiseMatchResponse,
    BatchMatchRequest,
    BatchMatchResponse,
    BatchMatchResult,
    MatchResult,
    DescriptorType,
    ProvenanceRequest,
    ProvenanceResponse,
    ProvenanceGraphResult,
    ProvenanceGraphNode,
    ProvenanceGraphEdge,
    ProvenanceMatchedPair,
)
from .keypoint_extractor import extract_descriptors, save_descriptors, load_descriptors
from .matcher import match_images
from .graph_builder import ProvenanceGraphBuilder, generate_visualization_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def handle_extract(request: ExtractDescriptorsRequest) -> ExtractDescriptorsResponse:
    """Handle descriptor extraction command."""
    try:
        # Extract descriptors
        result = extract_descriptors(
            image_path=request.image_path,
            descriptor_type=request.descriptor_type,
            extract_flip=request.extract_flip
        )
        
        # Generate base name from image path
        base_name = Path(request.image_path).stem
        
        # Save to output directory
        paths = save_descriptors(
            result=result,
            output_dir=request.output_dir,
            base_name=f"{base_name}_{request.descriptor_type.value}"
        )
        
        return ExtractDescriptorsResponse(
            success=True,
            message="Descriptors extracted successfully",
            keypoints_path=paths['keypoints_path'],
            descriptors_path=paths['descriptors_path'],
            flip_keypoints_path=paths.get('flip_keypoints_path'),
            flip_descriptors_path=paths.get('flip_descriptors_path'),
            keypoint_count=len(result['keypoints']),
            flip_keypoint_count=len(result.get('flip_keypoints', []))
        )
        
    except Exception as e:
        logger.exception("Error extracting descriptors")
        return ExtractDescriptorsResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


def handle_match(request: PairwiseMatchRequest) -> PairwiseMatchResponse:
    """Handle pairwise matching command."""
    try:
        # Load descriptors for image 1
        desc1 = load_descriptors(
            keypoints_path=request.image1.keypoints_path,
            descriptors_path=request.image1.descriptors_path,
            flip_keypoints_path=request.image1.flip_keypoints_path if request.check_flip else None,
            flip_descriptors_path=request.image1.flip_descriptors_path if request.check_flip else None
        )
        
        # Load descriptors for image 2
        desc2 = load_descriptors(
            keypoints_path=request.image2.keypoints_path,
            descriptors_path=request.image2.descriptors_path
        )
        
        # Perform matching
        result = match_images(
            image1_path=request.image1.image_path,
            keypoints1=desc1['keypoints'],
            descriptors1=desc1['descriptors'],
            image2_path=request.image2.image_path,
            keypoints2=desc2['keypoints'],
            descriptors2=desc2['descriptors'],
            flip_keypoints1=desc1.get('flip_keypoints'),
            flip_descriptors1=desc1.get('flip_descriptors'),
            alignment_strategy=request.alignment_strategy,
            matching_method=request.matching_method,
            min_keypoints=request.min_keypoints,
            min_area=request.min_area,
            check_flip=request.check_flip
        )
        
        # Generate visualization if output directory provided
        visualization_path = None
        if request.output_dir and result['is_match']:
            vis_filename = f"match_{Path(request.image1.image_path).stem}_{Path(request.image2.image_path).stem}.png"
            visualization_path = str(Path(request.output_dir) / vis_filename)
            # Note: Would need to re-run matching to get actual matched keypoints for visualization
            # For now, skip visualization in the match response
        
        return PairwiseMatchResponse(
            success=True,
            message="Matching completed",
            result=MatchResult(
                shared_area_img1=result['shared_area_img1'],
                shared_area_img2=result['shared_area_img2'],
                matched_keypoints=result['matched_keypoints'],
                is_flipped_match=result['is_flipped_match'],
                visualization_path=visualization_path
            )
        )
        
    except Exception as e:
        logger.exception("Error matching images")
        return PairwiseMatchResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


def handle_batch(request: BatchMatchRequest) -> BatchMatchResponse:
    """Handle batch matching command."""
    results = []
    successful = 0
    failed = 0
    
    def process_pair(idx: int, pair: PairwiseMatchRequest) -> BatchMatchResult:
        response = handle_match(pair)
        if response.success and response.result:
            return BatchMatchResult(
                pair_index=idx,
                image1_path=pair.image1.image_path,
                image2_path=pair.image2.image_path,
                result=response.result
            )
        else:
            return BatchMatchResult(
                pair_index=idx,
                image1_path=pair.image1.image_path,
                image2_path=pair.image2.image_path,
                error=response.message
            )
    
    if request.parallel and len(request.pairs) > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
            futures = {
                executor.submit(process_pair, idx, pair): idx
                for idx, pair in enumerate(request.pairs)
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.error:
                    failed += 1
                else:
                    successful += 1
    else:
        # Sequential processing
        for idx, pair in enumerate(request.pairs):
            result = process_pair(idx, pair)
            results.append(result)
            if result.error:
                failed += 1
            else:
                successful += 1
    
    # Sort results by pair index
    results.sort(key=lambda x: x.pair_index)
    
    return BatchMatchResponse(
        success=True,
        message=f"Batch matching completed: {successful}/{len(request.pairs)} successful",
        total_pairs=len(request.pairs),
        successful_matches=successful,
        failed_matches=failed,
        results=results
    )


def handle_provenance(request: ProvenanceRequest) -> ProvenanceResponse:
    """
    Handle full provenance analysis.
    
    This processes all image pairs, builds a provenance graph, and computes
    the maximum spanning tree and connected components.
    """
    start_time = time.time()
    logger.info(f"Starting provenance analysis with {len(request.images)} images")
    
    # Build image lookup - include pre-computed descriptor paths if provided
    image_lookup: Dict[str, Any] = {}
    for img in request.images:
        image_lookup[img.id] = {
            "path": img.path,
            "label": img.label or Path(img.path).name,
            "is_query": img.id in request.query_image_ids,
            "metadata": img.metadata,
            "descriptor_paths": img.descriptor_paths.model_dump() if img.descriptor_paths else None
        }
    
    # Create output directory
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors_dir = output_dir / "descriptors"
    descriptors_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract descriptors for all images
    logger.info("Step 1: Extracting descriptors...")
    descriptors: Dict[str, Dict[str, Any]] = {}
    
    for img_id, img_info in image_lookup.items():
        try:
            # First, check if pre-computed descriptor paths were provided
            provided_paths = img_info.get("descriptor_paths")
            if provided_paths:
                kp_path = provided_paths.get("keypoints_path")
                desc_path = provided_paths.get("descriptors_path")
                
                if kp_path and desc_path and Path(kp_path).exists() and Path(desc_path).exists():
                    logger.info(f"Loading pre-computed descriptors for {img_id}")
                    flip_kp_path = provided_paths.get("flip_keypoints_path")
                    flip_desc_path = provided_paths.get("flip_descriptors_path")
                    
                    descriptors[img_id] = load_descriptors(
                        keypoints_path=kp_path,
                        descriptors_path=desc_path,
                        flip_keypoints_path=flip_kp_path if flip_kp_path and Path(flip_kp_path).exists() else None,
                        flip_descriptors_path=flip_desc_path if flip_desc_path and Path(flip_desc_path).exists() else None
                    )
                    continue
            
            # Check if descriptors exist in output directory
            desc_base = descriptors_dir / f"{img_id}_{request.descriptor_type.value}"
            kp_path = f"{desc_base}_kps.npy"
            desc_path = f"{desc_base}_desc.npy"
            
            if Path(kp_path).exists() and Path(desc_path).exists():
                logger.info(f"Loading existing descriptors for {img_id}")
                flip_kp_path = f"{desc_base}_flip_kps.npy" if request.check_flip else None
                flip_desc_path = f"{desc_base}_flip_desc.npy" if request.check_flip else None
                
                descriptors[img_id] = load_descriptors(
                    keypoints_path=kp_path,
                    descriptors_path=desc_path,
                    flip_keypoints_path=flip_kp_path if flip_kp_path and Path(flip_kp_path).exists() else None,
                    flip_descriptors_path=flip_desc_path if flip_desc_path and Path(flip_desc_path).exists() else None
                )
            else:
                logger.info(f"Extracting descriptors for {img_id}")
                result = extract_descriptors(
                    image_path=img_info["path"],
                    descriptor_type=request.descriptor_type,
                    extract_flip=request.check_flip
                )
                
                descriptors[img_id] = result
                
                # Optionally save descriptors
                if request.save_descriptors:
                    save_descriptors(
                        result=result,
                        output_dir=str(descriptors_dir),
                        base_name=f"{img_id}_{request.descriptor_type.value}"
                    )
        except Exception as e:
            logger.error(f"Failed to extract descriptors for {img_id}: {e}")
    
    logger.info(f"Extracted descriptors for {len(descriptors)} images")
    
    # Step 2: Build provenance graph by matching all pairs
    logger.info("Step 2: Matching image pairs...")
    graph_builder = ProvenanceGraphBuilder()
    
    # Add all nodes to the graph
    for img_id, img_info in image_lookup.items():
        if img_id in descriptors:
            graph_builder.add_node(
                image_id=img_id,
                label=img_info["label"],
                image_path=img_info["path"],
                is_query=img_info["is_query"],
                metadata=img_info.get("metadata")
            )
    
    # Generate all pairs to match
    image_ids = list(descriptors.keys())
    pairs_to_match = []
    for i, id1 in enumerate(image_ids):
        for id2 in image_ids[i+1:]:
            pairs_to_match.append((id1, id2))
    
    logger.info(f"Matching {len(pairs_to_match)} pairs")
    
    def match_pair(pair):
        id1, id2 = pair
        try:
            desc1 = descriptors[id1]
            desc2 = descriptors[id2]
            
            result = match_images(
                image1_path=image_lookup[id1]["path"],
                keypoints1=desc1['keypoints'],
                descriptors1=desc1['descriptors'],
                image2_path=image_lookup[id2]["path"],
                keypoints2=desc2['keypoints'],
                descriptors2=desc2['descriptors'],
                flip_keypoints1=desc1.get('flip_keypoints'),
                flip_descriptors1=desc1.get('flip_descriptors'),
                alignment_strategy=request.alignment_strategy,
                matching_method=request.matching_method,
                min_keypoints=request.min_keypoints,
                min_area=request.min_area,
                check_flip=request.check_flip
            )
            
            return (id1, id2, result)
        except Exception as e:
            logger.error(f"Error matching {id1} vs {id2}: {e}")
            return (id1, id2, None)
    
    # Process pairs
    matched_count = 0
    if request.parallel and len(pairs_to_match) > 1:
        with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
            futures = [executor.submit(match_pair, pair) for pair in pairs_to_match]
            
            for future in as_completed(futures):
                id1, id2, result = future.result()
                if result and result.get('is_match', False):
                    graph_builder.add_match(
                        img1_id=id1,
                        img2_id=id2,
                        shared_area_img1=result['shared_area_img1'],
                        shared_area_img2=result['shared_area_img2'],
                        matched_keypoints=result['matched_keypoints'],
                        is_flipped=result.get('is_flipped_match', False)
                    )
                    matched_count += 1
    else:
        for pair in pairs_to_match:
            id1, id2, result = match_pair(pair)
            if result and result.get('is_match', False):
                graph_builder.add_match(
                    img1_id=id1,
                    img2_id=id2,
                    shared_area_img1=result['shared_area_img1'],
                    shared_area_img2=result['shared_area_img2'],
                    matched_keypoints=result['matched_keypoints'],
                    is_flipped=result.get('is_flipped_match', False)
                )
                matched_count += 1
    
    logger.info(f"Found {matched_count} matching pairs")
    
    # Step 3: Build final graph
    logger.info("Step 3: Building provenance graph...")
    graph_data = graph_builder.build_graph()
    
    # Convert to response schema
    graph_nodes = [
        ProvenanceGraphNode(
            id=n["id"],
            label=n["label"],
            image_path=n["image_path"],
            is_query=n["is_query"],
            metadata=n.get("metadata")
        )
        for n in graph_data["nodes"]
    ]
    
    graph_edges = [
        ProvenanceGraphEdge(
            source=e["source"],
            target=e["target"],
            weight=e["weight"],
            shared_area_source=e["shared_area_source"],
            shared_area_target=e["shared_area_target"],
            matched_keypoints=e["matched_keypoints"],
            is_flipped=e["is_flipped"]
        )
        for e in graph_data["edges"]
    ]
    
    spanning_tree = None
    if graph_data.get("spanning_tree_edges"):
        spanning_tree = [
            ProvenanceGraphEdge(
                source=e["source"],
                target=e["target"],
                weight=e["weight"],
                shared_area_source=e["shared_area_source"],
                shared_area_target=e["shared_area_target"],
                matched_keypoints=e["matched_keypoints"],
                is_flipped=e["is_flipped"]
            )
            for e in graph_data["spanning_tree_edges"]
        ]
    
    matched_pairs = [
        ProvenanceMatchedPair(
            image1_id=p["image1_id"],
            image2_id=p["image2_id"],
            shared_area_img1=p["shared_area_img1"],
            shared_area_img2=p["shared_area_img2"],
            matched_keypoints=p["matched_keypoints"],
            is_flipped=p["is_flipped"]
        )
        for p in graph_data["matched_pairs"]
    ]
    
    graph_result = ProvenanceGraphResult(
        nodes=graph_nodes,
        edges=graph_edges,
        spanning_tree_edges=spanning_tree,
        connected_components=graph_data.get("connected_components"),
        adjacency_matrix=graph_data.get("adjacency_matrix")
    )
    
    # Generate visualization data
    vis_data = generate_visualization_data(graph_data)
    
    # Save results to files
    output_files = {}
    
    # Save graph JSON
    graph_json_path = output_dir / "provenance_graph.json"
    with open(graph_json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    output_files["graph_json"] = str(graph_json_path)
    
    # Save visualization data
    vis_json_path = output_dir / "visualization_data.json"
    with open(vis_json_path, 'w') as f:
        json.dump(vis_data, f, indent=2)
    output_files["visualization_json"] = str(vis_json_path)
    
    processing_time = time.time() - start_time
    logger.info(f"Provenance analysis completed in {processing_time:.2f}s")
    
    return ProvenanceResponse(
        success=True,
        message=f"Provenance analysis completed: {len(graph_nodes)} images, {matched_count} matches",
        total_images=len(graph_nodes),
        total_pairs_checked=len(pairs_to_match),
        matched_pairs_count=matched_count,
        processing_time_seconds=processing_time,
        graph=graph_result,
        matched_pairs=matched_pairs,
        visualization_data=vis_data,
        output_files=output_files
    )


@click.group()
def cli():
    """Provenance Analysis - Content Sharing Detection"""
    pass


@cli.command()
@click.option('--image', '-i', required=True, help='Path to input image')
@click.option('--output', '-o', required=True, help='Output directory for descriptors')
@click.option('--descriptor-type', '-d', default='vlfeat_sift_heq',
              type=click.Choice(['vlfeat_sift_heq', 'cv_sift', 'cv_rsift']),
              help='Descriptor type to extract')
@click.option('--no-flip', is_flag=True, help='Skip extracting flipped version')
def extract(image: str, output: str, descriptor_type: str, no_flip: bool):
    """Extract keypoint descriptors from an image."""
    request = ExtractDescriptorsRequest(
        image_path=image,
        output_dir=output,
        descriptor_type=DescriptorType(descriptor_type),
        extract_flip=not no_flip
    )
    
    response = handle_extract(request)
    
    output_data = ContainerOutput(
        success=response.success,
        command=CommandType.EXTRACT,
        message=response.message,
        extract_response=response
    )
    
    print(output_data.model_dump_json(indent=2))
    sys.exit(0 if response.success else 1)


@cli.command()
@click.option('--input', '-i', 'input_json', required=True, help='Path to JSON input file or JSON string')
def match(input_json: str):
    """Match two images from JSON input."""
    # Parse input
    if Path(input_json).exists():
        with open(input_json) as f:
            data = json.load(f)
    else:
        data = json.loads(input_json)
    
    request = PairwiseMatchRequest(**data)
    response = handle_match(request)
    
    output_data = ContainerOutput(
        success=response.success,
        command=CommandType.MATCH,
        message=response.message,
        match_response=response
    )
    
    print(output_data.model_dump_json(indent=2))
    sys.exit(0 if response.success else 1)


@cli.command()
@click.option('--input', '-i', 'input_json', required=True, help='Path to JSON input file or JSON string')
def batch(input_json: str):
    """Batch match multiple image pairs from JSON input."""
    # Parse input
    if Path(input_json).exists():
        with open(input_json) as f:
            data = json.load(f)
    else:
        data = json.loads(input_json)
    
    request = BatchMatchRequest(**data)
    response = handle_batch(request)
    
    output_data = ContainerOutput(
        success=response.success,
        command=CommandType.BATCH_MATCH,
        message=response.message,
        batch_response=response
    )
    
    print(output_data.model_dump_json(indent=2))
    sys.exit(0 if response.success else 1)


@cli.command()
@click.option('--input', '-i', 'input_json', required=True, help='Path to JSON input file or JSON string')
def provenance(input_json: str):
    """Run full provenance analysis from JSON input."""
    # Parse input
    if Path(input_json).exists():
        with open(input_json) as f:
            data = json.load(f)
    else:
        data = json.loads(input_json)
    
    try:
        request = ProvenanceRequest(**data)
        response = handle_provenance(request)
        
        output_data = ContainerOutput(
            success=response.success,
            command=CommandType.PROVENANCE,
            message=response.message,
            provenance_response=response
        )
    except Exception as e:
        logger.exception("Error in provenance analysis")
        output_data = ContainerOutput(
            success=False,
            command=CommandType.PROVENANCE,
            message=f"Error: {str(e)}"
        )
    
    print(output_data.model_dump_json(indent=2))
    sys.exit(0 if output_data.success else 1)


@cli.command()
@click.option('--input', '-i', 'input_json', required=True, help='Path to JSON input file with full command')
def run(input_json: str):
    """Run any command from JSON input (for Docker entrypoint)."""
    # Parse input
    if Path(input_json).exists():
        with open(input_json) as f:
            data = json.load(f)
    else:
        data = json.loads(input_json)
    
    container_input = ContainerInput(**data)
    
    if container_input.command == CommandType.EXTRACT:
        if not container_input.extract_request:
            print(json.dumps({"success": False, "message": "Missing extract_request"}))
            sys.exit(1)
        response = handle_extract(container_input.extract_request)
        output = ContainerOutput(
            success=response.success,
            command=CommandType.EXTRACT,
            message=response.message,
            extract_response=response
        )
        
    elif container_input.command == CommandType.MATCH:
        if not container_input.match_request:
            print(json.dumps({"success": False, "message": "Missing match_request"}))
            sys.exit(1)
        response = handle_match(container_input.match_request)
        output = ContainerOutput(
            success=response.success,
            command=CommandType.MATCH,
            message=response.message,
            match_response=response
        )
        
    elif container_input.command == CommandType.BATCH_MATCH:
        if not container_input.batch_request:
            print(json.dumps({"success": False, "message": "Missing batch_request"}))
            sys.exit(1)
        response = handle_batch(container_input.batch_request)
        output = ContainerOutput(
            success=response.success,
            command=CommandType.BATCH_MATCH,
            message=response.message,
            batch_response=response
        )
        
    elif container_input.command == CommandType.PROVENANCE:
        if not container_input.provenance_request:
            print(json.dumps({"success": False, "message": "Missing provenance_request"}))
            sys.exit(1)
        response = handle_provenance(container_input.provenance_request)
        output = ContainerOutput(
            success=response.success,
            command=CommandType.PROVENANCE,
            message=response.message,
            provenance_response=response
        )
        
    else:
        print(json.dumps({"success": False, "message": f"Unknown command: {container_input.command}"}))
        sys.exit(1)
    
    print(output.model_dump_json(indent=2))
    sys.exit(0 if output.success else 1)


if __name__ == '__main__':
    cli()
