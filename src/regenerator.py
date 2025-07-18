# regenerator.py
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import math
from collections import deque
from src.eon_models import ModulationFormat, EONLink
import networkx as nx
from src.spectrum_manager import SpectrumManager
from src.topology_loader import calculate_path_length, get_path_links

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after adding src to path
from src.ml_qot import MLQoTEstimator

if TYPE_CHECKING:
    from src.ml_qot import MLQoTEstimator

# Configure logging
logger = logging.getLogger(__name__)

class Regenerator:
    def __init__(self, G: nx.Graph, max_segment_length: float = 200.0):
        """
        Initialize the regenerator.
        
        Args:
            G: NetworkX graph object
            max_segment_length: Maximum length of a path segment before regeneration (in km)
        """
        self.G = G
        self.max_segment_length = max_segment_length
        self.qot_estimator = MLQoTEstimator()
        
        # QoT thresholds based on OSNR requirements
        self.qot_thresholds = {
            ModulationFormat.QPSK: 15.0,   # Minimum OSNR for QPSK
            ModulationFormat.QAM8: 18.0,   # Higher OSNR for QAM8
            ModulationFormat.QAM16: 21.0,  # Higher OSNR for QAM16
            ModulationFormat.QAM64: 24.0   # Highest OSNR for QAM64
        }
        
        # Regeneration parameters
        self.regeneration_params = {
            "cost": 1000.0,  # Cost per regeneration in arbitrary units
            "power_consumption": 50.0,  # Watts per regeneration
            "latency": 0.1,  # ms per regeneration
            "max_regenerations": 5  # Maximum number of regenerations per path
        }
        
        # Initialize links dictionary
        self.links = {}
        for u, v, data in self.G.edges(data=True):
            # Get distance from edge data
            if 'length' in data:
                length = data['length']
            else:
                # Calculate from coordinates if length not available
                u_coords = (self.G.nodes[u].get('lon', 0), self.G.nodes[u].get('lat', 0))
                v_coords = (self.G.nodes[v].get('lon', 0), self.G.nodes[v].get('lat', 0))
                length = math.sqrt((u_coords[0] - v_coords[0])**2 + (u_coords[1] - v_coords[1])**2)
            
            # Create EONLink object
            self.links[(u, v)] = EONLink(
                length=length,
                fiber_type="SMF-28"  # Default fiber type
            )
            # Also create reverse link for undirected graph
            self.links[(v, u)] = EONLink(
                length=length,
                fiber_type="SMF-28"
            )
        
    def _validate_path(self, path: List[str]) -> bool:
        """
        Validate if a path is valid in the graph.
        
        Args:
            path: List of nodes representing the path
            
        Returns:
            bool: True if path is valid, False otherwise
        """
        if not path or len(path) < 2:
            return False
            
        # Check if all edges in path exist
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.G.has_edge(u, v):
                logger.error(f"Invalid edge {u}-{v} in path {path}")
                return False
                
        return True
        
    def _find_nearest_regeneration_point(self, node: str, target_node: str, current_segment: List[str], num_slots: int, modulation: ModulationFormat) -> Optional[str]:
        """
        Find the optimal regeneration point considering QoT and path feasibility.
        
        Args:
            node: Current node
            target_node: Target node to reach
            current_segment: Current path segment
            num_slots: Number of spectrum slots required
            modulation: Modulation format to use
            
        Returns:
            Optimal regeneration point or None if not found
        """
        if not self.G.has_node(node):
            return None
            
        # Check if current node can regenerate
        if self.G.nodes[node].get('can_regenerate', False):
            return node
            
        # Search for regeneration points using Dijkstra's algorithm
        distances = {node: 0}
        previous: Dict[str, Optional[str]] = {node: None}
        regeneration_points = []
        visited = set()
        
        # Priority queue for Dijkstra's algorithm
        import heapq
        queue = [(0, node)]
        
        while queue:
            current_dist, current = heapq.heappop(queue)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Check if current node can regenerate
            if self.G.nodes[current].get('can_regenerate', False):
                # Test if this regeneration point would create feasible segments
                test_segment1 = current_segment + [current]
                test_segment2 = [current, target_node]
                
                qot1, conf1 = self._calculate_segment_qot(test_segment1, num_slots, modulation)
                qot2, conf2 = self._calculate_segment_qot(test_segment2, num_slots, modulation)
                
                if (qot1 >= self.qot_thresholds[modulation] and 
                    qot2 >= self.qot_thresholds[modulation]):
                    regeneration_points.append((current, current_dist))
            
            # Add neighbors to queue
            for neighbor in self.G.neighbors(current):
                if neighbor not in visited:
                    new_dist = current_dist + self.G[current][neighbor]['length']
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(queue, (new_dist, neighbor))
        
        if not regeneration_points:
            return None
            
        # Return the closest regeneration point that creates feasible segments
        return min(regeneration_points, key=lambda x: x[1])[0]
        
    def _calculate_segment_qot(self, segment: List[str], num_slots: int, modulation: ModulationFormat) -> Tuple[float, float]:
        """Calculate QoT for a path segment using ML model."""
        try:
            # Calculate QoT using ML model
            qot, confidence = self.qot_estimator.estimate_qot(
                path=segment,
                links=self.links,
                launch_power=0.0,  # dBm
                channel_spacing=12.5,  # GHz
                num_channels=1,
                modulation=modulation
            )
            
            return qot, confidence
            
        except Exception as e:
            logger.error(f"Error calculating segment QoT: {str(e)}")
            return float('inf'), 0.0

    def find_regeneration_points(self, path: List[str], num_slots: int, modulation: ModulationFormat) -> List[List[str]]:
        """Find optimal regeneration points for a path."""
        try:
            # Get QoT threshold for modulation format
            qot_threshold = self.qot_thresholds[modulation]
            
            # Initialize segments
            segments = []
            current_segment = [path[0]]
            
            # Check each node in the path
            for i in range(1, len(path)):
                current_segment.append(path[i])
                
                # Calculate QoT for current segment using ML model
                segment_qot, confidence = self._calculate_segment_qot(
                    current_segment,
                    num_slots,
                    modulation
                )
                
                # If QoT is below threshold, add regeneration point
                if segment_qot < qot_threshold:
                    # Remove last node from current segment
                    current_segment.pop()
                    
                    # Add current segment to segments
                    segments.append(current_segment)
                    
                    # Start new segment with last node
                    current_segment = [path[i]]
            
            # Add final segment if not empty
            if current_segment:
                segments.append(current_segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error finding regeneration points: {str(e)}")
            return []

    def calculate_regeneration_cost(self, segments: List[List[str]]) -> float:
        """Calculate total regeneration cost for path segments."""
        try:
            total_cost = 0.0
            
            # Calculate cost for each segment
            for segment in segments:
                # Calculate segment length
                segment_length = calculate_path_length(self.G, segment)
                
                # Calculate regeneration cost based on length
                segment_cost = self.regeneration_params["cost"] * segment_length
                total_cost += segment_cost
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating regeneration cost: {str(e)}")
            return float('inf')
        
    def calculate_regeneration_latency(self, segments: List[List[str]]) -> float:
        """
        Calculate total latency introduced by regeneration.
        
        Args:
            segments: List of path segments
            
        Returns:
            Total latency introduced by regeneration
        """
        return (len(segments) - 1) * self.regeneration_params["latency"]
        
    def calculate_regeneration_power(self, segments: List[List[str]]) -> float:
        """
        Calculate total power consumption of regeneration.
        
        Args:
            segments: List of path segments
            
        Returns:
            Total power consumption of regeneration
        """
        return (len(segments) - 1) * self.regeneration_params["power_consumption"]

    def regenerate_path(self, path: List[str], links: Dict[Tuple[str, str], EONLink], 
                       modulation: ModulationFormat) -> Tuple[List[List[str]], List[str]]:
        """
        Regenerate a path to meet QoT requirements, consuming regenerators at each regeneration node.
        Returns (segments, regeneration_nodes) if successful, ([], []) otherwise.
        """
        try:
            segments = self.find_regeneration_points(path, 1, modulation)  # Using 1 slot for simplicity
            if not segments:
                logger.info("No feasible regeneration points found")
                return [], []
            # Identify internal regeneration nodes (exclude source and destination)
            regeneration_nodes = [seg[0] for seg in segments[1:]]  # First node of each segment except the first
            # Check regenerator availability
            unavailable = [n for n in regeneration_nodes if self.G.nodes[n]['regenerators_available'] <= 0]
            if unavailable:
                logger.info(f"Regenerator unavailable at nodes: {unavailable}")
                return [], []
            # Reserve regenerators
            for n in regeneration_nodes:
                self.G.nodes[n]['regenerators_available'] -= 1
            # Check QoT for all segments
            for segment in segments:
                qot, confidence = self._calculate_segment_qot(segment, 1, modulation)
                if qot < self.qot_thresholds[modulation]:
                    logger.info(f"Segment {' -> '.join(segment)} does not meet QoT requirements")
                    # Roll back regenerator reservations
                    for n in regeneration_nodes:
                        self.G.nodes[n]['regenerators_available'] += 1
                    return [], []
            logger.info(f"Successfully regenerated path into {len(segments)} segments with regenerators at {regeneration_nodes}")
            return segments, regeneration_nodes
        except Exception as e:
            logger.error(f"Error regenerating path: {str(e)}")
            return [], [] 
