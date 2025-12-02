"""
Graph building module for provenance analysis.

This module implements:
- Adjacency matrix construction from pairwise matches
- Maximum spanning tree computation
- Connected component detection
- Graph serialization for visualization
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


class ProvenanceGraphBuilder:
    """
    Builds a provenance graph from pairwise image matches.
    
    The graph represents content sharing relationships between images,
    where edges are weighted by the shared content area.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adjacency_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.matched_pairs: List[Dict[str, Any]] = []
        self.visited_pairs: Set[Tuple[str, str]] = set()
    
    def add_node(
        self,
        image_id: str,
        label: str,
        image_path: str,
        is_query: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a node to the graph."""
        if image_id not in self.nodes:
            self.nodes[image_id] = {
                "id": image_id,
                "label": label,
                "image_path": image_path,
                "is_query": is_query,
                "metadata": metadata or {}
            }
        elif is_query:
            # Update to mark as query if already exists
            self.nodes[image_id]["is_query"] = True
    
    def get_pair_key(self, img1_id: str, img2_id: str) -> Tuple[str, str]:
        """Get canonical key for an image pair (sorted to avoid duplicates)."""
        return tuple(sorted([img1_id, img2_id]))
    
    def is_pair_visited(self, img1_id: str, img2_id: str) -> bool:
        """Check if a pair has already been processed."""
        return self.get_pair_key(img1_id, img2_id) in self.visited_pairs
    
    def mark_pair_visited(self, img1_id: str, img2_id: str):
        """Mark a pair as visited."""
        self.visited_pairs.add(self.get_pair_key(img1_id, img2_id))
    
    def add_match(
        self,
        img1_id: str,
        img2_id: str,
        shared_area_img1: float,
        shared_area_img2: float,
        matched_keypoints: int,
        is_flipped: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a match between two images."""
        # Update adjacency matrix
        self.adjacency_matrix[img1_id][img2_id] = shared_area_img1
        self.adjacency_matrix[img2_id][img1_id] = shared_area_img2
        
        # Record matched pair
        self.matched_pairs.append({
            "image1_id": img1_id,
            "image2_id": img2_id,
            "shared_area_img1": shared_area_img1,
            "shared_area_img2": shared_area_img2,
            "matched_keypoints": matched_keypoints,
            "is_flipped": is_flipped,
            "metadata": metadata or {}
        })
        
        self.mark_pair_visited(img1_id, img2_id)
    
    def build_graph(self) -> Dict[str, Any]:
        """
        Build the final provenance graph.
        
        Returns:
            Dictionary containing graph structure with nodes, edges,
            spanning tree, and connected components.
        """
        # Create edges from adjacency matrix
        edges = []
        edge_set = set()
        
        for src_id, targets in self.adjacency_matrix.items():
            for tgt_id, weight in targets.items():
                # Avoid duplicate edges
                pair_key = self.get_pair_key(src_id, tgt_id)
                if pair_key not in edge_set:
                    edge_set.add(pair_key)
                    
                    # Get matched pair info
                    matched_kpts = 0
                    is_flipped = False
                    for pair in self.matched_pairs:
                        if (pair["image1_id"] == src_id and pair["image2_id"] == tgt_id) or \
                           (pair["image1_id"] == tgt_id and pair["image2_id"] == src_id):
                            matched_kpts = pair["matched_keypoints"]
                            is_flipped = pair["is_flipped"]
                            break
                    
                    edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "weight": max(weight, self.adjacency_matrix.get(tgt_id, {}).get(src_id, 0)),
                        "shared_area_source": weight,
                        "shared_area_target": self.adjacency_matrix.get(tgt_id, {}).get(src_id, 0),
                        "matched_keypoints": matched_kpts,
                        "is_flipped": is_flipped
                    })
        
        # Build NetworkX graph for analysis
        spanning_tree_edges = []
        connected_components = []
        
        if edges:
            G = nx.Graph()
            for edge in edges:
                G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
            
            # Get connected components
            for component in nx.connected_components(G):
                if len(component) > 1:
                    connected_components.append(list(component))
            
            # Compute maximum spanning tree for each component
            for component in connected_components:
                subgraph = G.subgraph(component)
                try:
                    mst = nx.maximum_spanning_tree(subgraph)
                    
                    for u, v, data in mst.edges(data=True):
                        # Find the original edge
                        for edge in edges:
                            if (edge["source"] == u and edge["target"] == v) or \
                               (edge["source"] == v and edge["target"] == u):
                                spanning_tree_edges.append(edge)
                                break
                except Exception as e:
                    logger.warning(f"Error computing spanning tree: {e}")
        
        # Build adjacency matrix dict
        adj_matrix_dict = {
            src: dict(targets) for src, targets in self.adjacency_matrix.items()
        }
        
        return {
            "nodes": list(self.nodes.values()),
            "edges": edges,
            "adjacency_matrix": adj_matrix_dict if adj_matrix_dict else None,
            "spanning_tree_edges": spanning_tree_edges if spanning_tree_edges else None,
            "connected_components": connected_components if connected_components else None,
            "matched_pairs": self.matched_pairs,
            "statistics": {
                "total_nodes": len(self.nodes),
                "total_edges": len(edges),
                "total_matched_pairs": len(self.matched_pairs),
                "num_components": len(connected_components)
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to JSON."""
        return json.dumps(self.build_graph(), indent=indent)


def generate_visualization_data(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate vis.js compatible data for graph visualization.
    
    Args:
        graph: Graph dictionary from build_graph()
        
    Returns:
        Dictionary with vis.js formatted nodes and edges
    """
    # Prepare nodes for vis.js
    vis_nodes = []
    for node in graph["nodes"]:
        label = node["label"]
        if len(label) > 20:
            label = label[:17] + "..."
        
        vis_node = {
            "id": node["id"],
            "label": label,
            "title": f"{node['label']}<br>Path: {node['image_path']}",
            "color": {
                "background": "#4CAF50" if node["is_query"] else "#2196F3",
                "border": "#388E3C" if node["is_query"] else "#1976D2"
            },
            "borderWidth": 3 if node["is_query"] else 1,
            "size": 30 if node["is_query"] else 20,
            "data": {
                "imageId": node["id"],
                "imagePath": node["image_path"],
                "isQuery": node["is_query"],
                "metadata": node.get("metadata", {})
            }
        }
        vis_nodes.append(vis_node)
    
    # Use spanning tree edges if available for cleaner visualization
    edge_list = graph.get("spanning_tree_edges") or graph.get("edges", [])
    
    vis_edges = []
    for edge in edge_list:
        weight_percent = round(edge["weight"] * 100, 1)
        vis_edge = {
            "from": edge["source"],
            "to": edge["target"],
            "value": edge["weight"],
            "title": f"Shared: {weight_percent}%<br>Keypoints: {edge['matched_keypoints']}{'<br>(Flipped)' if edge['is_flipped'] else ''}",
            "color": {
                "color": "#FF9800" if edge["is_flipped"] else "#9E9E9E",
                "highlight": "#F57C00" if edge["is_flipped"] else "#616161"
            },
            "width": max(1, min(5, edge["weight"] * 10)),
            "data": {
                "sharedAreaSource": edge["shared_area_source"],
                "sharedAreaTarget": edge["shared_area_target"],
                "matchedKeypoints": edge["matched_keypoints"],
                "isFlipped": edge["is_flipped"]
            }
        }
        vis_edges.append(vis_edge)
    
    return {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "components": graph.get("connected_components", []),
        "statistics": graph.get("statistics", {})
    }
