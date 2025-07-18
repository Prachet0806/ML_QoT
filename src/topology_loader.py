# topology_loader.py
import networkx as nx
import numpy as np
from typing import List, Tuple
import logging
from itertools import islice
import math

logger = logging.getLogger(__name__)

def calculate_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    R = 6371  # Earth's radius in km
    
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    
    return distance

def load_topology(file_path: str) -> nx.Graph:
    """Load network topology from GML file."""
    try:
        G = nx.read_gml(file_path)
        logger.info(f"Loaded topology with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        for u, v, data in G.edges(data=True):
            if 'length' in data:
                G[u][v]['length'] = float(data['length'])
                G[v][u]['length'] = float(data['length'])
            else:
                logger.warning(f"No length found for edge {u}-{v}, using coordinates")
                # Fallback to coordinate-based calculation
                u_lon = G.nodes[u].get('lon', 0)
                u_lat = G.nodes[u].get('lat', 0)
                v_lon = G.nodes[v].get('lon', 0)
                v_lat = G.nodes[v].get('lat', 0)
                length = calculate_distance(u_lon, u_lat, v_lon, v_lat)
                G[u][v]['length'] = length
                G[v][u]['length'] = length
            
        logger.info("Processed edge lengths from GML file")
        return G
    except Exception as e:
        logger.error(f"Error loading topology: {str(e)}")
        raise

def get_k_shortest_paths(G: nx.Graph, source: str, target: str, k: int = 3) -> List[List[str]]:
    """Get k-shortest paths between source and target nodes using Yen's algorithm."""
    logger.debug(f"Starting k-shortest paths search from {source} to {target} with k={k}")
    try:
        # Get initial shortest path
        try:
            shortest_path = nx.shortest_path(G, source, target, weight='length')
            if not shortest_path:
                logger.warning(f"No path found from {source} to {target}")
                return []
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {source} to {target}")
            return []

        # Initialize list of k-shortest paths
        k_paths = [shortest_path]
        candidates = []

        # Find k-1 more paths
        for _ in range(k-1):
            # Get the last path
            last_path = k_paths[-1]
            
            # For each node in the last path (except target)
            for i in range(len(last_path)-1):
                # Create a modified graph
                G_modified = G.copy()
                
                # Remove edges used in previous paths
                for path in k_paths:
                    for j in range(len(path)-1):
                        if G_modified.has_edge(path[j], path[j+1]):
                            G_modified.remove_edge(path[j], path[j+1])
                
                # Remove edges from spur node to target
                spur_node = last_path[i]
                for j in range(i+1, len(last_path)-1):
                    if G_modified.has_edge(spur_node, last_path[j]):
                        G_modified.remove_edge(spur_node, last_path[j])
                
                # Find shortest path from spur node to target
                try:
                    spur_path = nx.shortest_path(G_modified, spur_node, target, weight='length')
                    if spur_path:
                        # Combine root path and spur path
                        root_path = last_path[:i+1]
                        candidate_path = root_path[:-1] + spur_path
                        
                        # Check if this is a valid path
                        if candidate_path not in k_paths and candidate_path not in candidates:
                            candidates.append(candidate_path)
                except nx.NetworkXNoPath:
                    continue
            
            if not candidates:
                break
                
            # Find the shortest candidate path
            shortest_candidate = min(candidates, key=lambda p: sum(G[u][v]['length'] for u, v in zip(p[:-1], p[1:])))
            k_paths.append(shortest_candidate)
            candidates.remove(shortest_candidate)
            
        logger.debug(f"Found {len(k_paths)} paths from {source} to {target}")
        return k_paths[:k]
            
    except Exception as e:
        logger.error(f"Error in path finding: {str(e)}")
        return []

def calculate_path_length(G: nx.Graph, path: List[str]) -> float:
    """Calculate total length of a path."""
    length = 0
    for i in range(len(path) - 1):
        length += G[path[i]][path[i + 1]]['length']
    return length

def get_path_links(path: List[str]) -> List[Tuple[str, str]]:
    """Get list of links in a path."""
    return list(zip(path[:-1], path[1:]))
