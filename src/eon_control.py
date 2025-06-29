#eon_control.py
from typing import List, Dict, Optional, Tuple
import networkx as nx
from eon_models import EONNode, EONLink, TransceiverSpec, ModulationFormat, WSSSpec
from ml_qot import MLQoTEstimator, QoTFeatures

class EONController:
    """EON network controller for path computation and resource allocation"""
    def __init__(self, G: nx.Graph):
        self.G = G
        self.nodes: Dict[str, EONNode] = {}
        self.links: Dict[Tuple[str, str], EONLink] = {}
        self.qot_estimator = MLQoTEstimator()
        self.initialize_network()
        
    def initialize_network(self) -> None:
        """Initialize network nodes and links"""
        # Create default WSS specification
        default_wss = WSSSpec(
            num_ports=8,
            filter_bandwidth=37.5,  # GHz
            insertion_loss=5.0,  # dB
            crosstalk=-30.0  # dB
        )
        
        # Create nodes
        for node_id in self.G.nodes():
            self.nodes[node_id] = EONNode(
                node_id=node_id,
                wss_spec=default_wss  # Fixed: provide proper WSS specification
            )
            
        # Create links
        for u, v in self.G.edges():
            length = self.G[u][v]['length']
            self.links[(u, v)] = EONLink(length=length)
            self.links[(v, u)] = EONLink(length=length)
            
            # Connect links to nodes
            self.nodes[u].add_link(self.links[(u, v)], v)
            self.nodes[v].add_link(self.links[(v, u)], u)
            
    def train_qot_estimator(self, training_data: List[Tuple[QoTFeatures, float]]) -> None:
        """Train the QoT estimator with provided training data"""
        self.qot_estimator.train(training_data)
        
    def find_path(self,
                  src: str,
                  dst: str,
                  required_bandwidth: float,
                  modulation: ModulationFormat) -> Optional[List[str]]:
        """
        Find a path between source and destination
        
        Args:
            src: Source node ID
            dst: Destination node ID
            required_bandwidth: Required bandwidth in GHz
            modulation: Required modulation format
            
        Returns:
            List of nodes representing the path if found, None otherwise
        """
        # Find k-shortest paths
        paths = list(nx.shortest_simple_paths(self.G, src, dst, weight='length'))
        
        for path in paths:
            if self._check_path_feasibility(path, required_bandwidth, modulation):
                return path
                
        return None
        
    def _check_path_feasibility(self,
                               path: List[str],
                               required_bandwidth: float,
                               modulation: ModulationFormat) -> bool:
        """
        Check if a path is feasible using ML-based QoT estimation
        
        Args:
            path: List of nodes representing the path
            required_bandwidth: Required bandwidth in GHz
            modulation: Required modulation format
            
        Returns:
            True if path is feasible, False otherwise
        """
        # Calculate total path length
        total_length = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_length += self.G[u][v]['length']
            
        # Check if path length is within reach of modulation format
        max_reach = {
            ModulationFormat.QPSK: 2000,  # km
            ModulationFormat.QAM8: 1000,
            ModulationFormat.QAM16: 500,
            ModulationFormat.QAM64: 250
        }
        
        if total_length > max_reach[modulation]:
            return False
            
        # Check spectrum availability - simplified check
        # In a real implementation, this would check with a spectrum manager
        # For now, we assume spectrum is available if path is feasible
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self.links[(u, v)]
            # Simplified spectrum check - assume available if path is feasible
            # TODO: Implement proper spectrum management
            pass
                
        # Estimate QoT using ML model
        estimated_qot, confidence = self.qot_estimator.estimate_qot(
            path=path,
            links=self.links,
            launch_power=3.0,  # Default launch power
            channel_spacing=12.5,  # Default channel spacing
            num_channels=1,  # Single channel
            modulation=modulation
        )
        
        # Check if QoT meets requirements with confidence margin
        min_qot = {
            ModulationFormat.QPSK: 12.0,  # dB
            ModulationFormat.QAM8: 15.0,
            ModulationFormat.QAM16: 18.0,
            ModulationFormat.QAM64: 21.0
        }
        
        return estimated_qot >= (min_qot[modulation] + confidence)
        
    def allocate_resources(self,
                          path: List[str],
                          required_bandwidth: float,
                          modulation: ModulationFormat) -> bool:
        """
        Allocate resources for a path
        
        Args:
            path: List of nodes representing the path
            required_bandwidth: Required bandwidth in GHz
            modulation: Required modulation format
            
        Returns:
            True if resources were allocated successfully, False otherwise
        """
        # Check if path is feasible
        if not self._check_path_feasibility(path, required_bandwidth, modulation):
            return False
            
        # Allocate spectrum on each link
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self.links[(u, v)]
            
            # Simplified spectrum allocation - assume success if path is feasible
            # TODO: Implement proper spectrum management
            # For now, we assume allocation succeeds if path is feasible
            pass
                
        return True
        
    def _release_resources(self, path: List[str]) -> None:
        """
        Release allocated resources for a path
        
        Args:
            path: List of nodes representing the path
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self.links[(u, v)]
            # Release spectrum (implementation depends on how spectrum is tracked)
            # TODO: Implement proper spectrum release
            pass
            
    def calculate_path_qot(self,
                          path: List[str],
                          launch_power: float,
                          channel_spacing: float,
                          num_channels: int) -> float:
        """
        Calculate QoT for a path
        
        Args:
            path: List of nodes representing the path
            launch_power: Launch power in dBm
            channel_spacing: Channel spacing in GHz
            num_channels: Number of channels
            
        Returns:
            QoT in dB
        """
        # Calculate number of amplifiers (simplified)
        total_length = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            total_length += self.G[u][v]['length']
        num_amplifiers = int(total_length / 80)  # One amplifier every 80 km
        
        # Calculate QoT for each link
        min_qot = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self.links[(u, v)]
            qot = link.calculate_total_qot(
                launch_power,
                channel_spacing,
                num_channels
            )
            min_qot = min(min_qot, qot)
            
        return min_qot 
