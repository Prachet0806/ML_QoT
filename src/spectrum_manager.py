# spectrum_manager.py
import os
import numpy as np
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import networkx as nx
import math

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eon_models import ModulationFormat

logger = logging.getLogger(__name__)

class SpectrumManager:
    def __init__(self, graph, total_slots=320):
        """
        Initialize spectrum manager.
        
        Args:
            graph: NetworkX graph object
            total_slots: Total number of spectrum slots per link
        """
        self.total_slots = total_slots
        self.spectrum = {
            tuple(sorted((u, v))): np.zeros(total_slots, dtype=int)
            for u, v in graph.edges()
        }
        self.connections = {}  
        self.fragmentation_history = []
        
        self.slot_width = 12.5
        
        # Modulation format efficiency (bits/symbol)
        self.modulation_efficiency = {
            ModulationFormat.QPSK: 2,
            ModulationFormat.QAM8: 3,
            ModulationFormat.QAM16: 4,
            ModulationFormat.QAM64: 6
        }
        
        # Adaptive weights for spectrum allocation scoring
        self.weights = {
            'fragmentation': 0.35,    
            'block_size': 0.25,       
            'fragments': 0.20,        
            'utilization': 0.15,      
            'edge_balance': 0.05      
        }
        
        # Track network-wide metrics for adaptive weight adjustment
        self.network_metrics = {
            'avg_utilization': 0.0,
            'avg_fragmentation': 0.0,
            'total_allocations': 0
        }

    def calculate_required_slots(self, bandwidth: float, modulation_format: ModulationFormat) -> int:
        """
        Calculate the number of slots required for a given bandwidth and modulation format.
        Args:
            bandwidth: Required bandwidth in Gbps
            modulation_format: Modulation format to use
        Returns:
            Number of slots required
        """
        if modulation_format not in self.modulation_efficiency:
            raise ValueError(f"Unsupported modulation format: {modulation_format}")
            
        # Calculate spectral efficiency (bits/symbol)
        efficiency = self.modulation_efficiency[modulation_format]
        
        # Calculate required slots
        # Formula: slots = ceil(bandwidth / (slot_width * efficiency))
        required_slots = int(np.ceil(bandwidth / (self.slot_width * efficiency)))
        
        # Add guard band (1 slot on each side)
        required_slots += 2
        
        logger.debug(f"Calculated required slots: {required_slots} for BW={bandwidth}Gbps, Mod={modulation_format}")
        return required_slots

    def _calculate_fragmentation(self, edge: Tuple[str, str]) -> float:
        """Calculate fragmentation score for an edge."""
        edge = (str(edge[0]), str(edge[1]))  # Ensure tuple of exactly two strings
        spectrum = self.spectrum[edge]
        occupied_slots = np.where(spectrum != 0)[0]
        if len(occupied_slots) == 0:
            return 0.0
        # Calculate gaps between occupied slots
        gaps = np.diff(occupied_slots) - 1
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            return 0.0
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        return float(std_gap / (mean_gap + 1e-6))  
        
    def _update_adaptive_weights(self):
        """Update weights based on network state."""
        if self.network_metrics['total_allocations'] < 10:
            return
            
        # Adjust weights based on network state
        if self.network_metrics['avg_utilization'] < 0.3:
            # Low utilization: prioritize block size and utilization
            self.weights['block_size'] = 0.30
            self.weights['utilization'] = 0.25
            self.weights['fragmentation'] = 0.25
            self.weights['fragments'] = 0.15
            self.weights['edge_balance'] = 0.05
        elif self.network_metrics['avg_fragmentation'] > 0.7:
            # High fragmentation: prioritize fragmentation reduction
            self.weights['fragmentation'] = 0.40
            self.weights['block_size'] = 0.25
            self.weights['fragments'] = 0.20
            self.weights['utilization'] = 0.10
            self.weights['edge_balance'] = 0.05
        else:
            # Balanced state: use default weights
            self.weights = {
                'fragmentation': 0.35,
                'block_size': 0.25,
                'fragments': 0.20,
                'utilization': 0.15,
                'edge_balance': 0.05
            }
        
    def _find_best_slot(self, path: List[str], required_slots: int) -> Optional[int]:
        """
        Find the best slot for allocation using an enhanced best-fit approach.
        Args:
            path: List of nodes in the path
            required_slots: Number of slots required 
        Returns:
            Best slot index if found, None otherwise
        """
        # Update adaptive weights based on network state
        self._update_adaptive_weights()
        
        # Get available slots for each edge
        available_slots = []
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            spectrum = self.spectrum[edge]
            # Find continuous available slots
            current_slot = None
            current_length = 0
            slots = []
            for i, slot in enumerate(spectrum):
                if slot == 0:  # Available slot
                    if current_slot is None:
                        current_slot = i
                    current_length += 1
                else:  # Occupied slot
                    if current_slot is not None and current_length >= required_slots:
                        slots.append((current_slot, current_length))
                    current_slot = None
                    current_length = 0
            
            # Handle case where spectrum ends with available slots
            if current_slot is not None and current_length >= required_slots:
                slots.append((current_slot, current_length))
            
            available_slots.append(slots)
        
        if not available_slots:
            return None
        
        # Find common available slots across all edges
        common_slots = set()
        for slots in available_slots:
            edge_slots = set()
            for start, length in slots:
                edge_slots.update(range(start, start + length - required_slots + 1))
            if not common_slots:
                common_slots = edge_slots
            else:
                common_slots &= edge_slots
        
        if not common_slots:
            return None
        
        # Score each potential slot based on multiple factors
        slot_scores = {}
        for slot in common_slots:
            score = 0
            edge_utilizations = []
            
            for u, v in zip(path[:-1], path[1:]):
                edge = tuple(sorted((u, v)))
                spectrum = self.spectrum[edge].copy()
                
                # Simulate allocation
                spectrum[slot:slot + required_slots] = 1
                
                # Calculate metrics after allocation
                metrics = self._calculate_fragmentation_metrics(spectrum)
                
                # Factor 1: Minimize fragmentation
                frag_score = 1 - metrics['fragmentation_index']
                
                # Factor 2: Prefer slots that create larger continuous blocks
                block_score = metrics['max_slice_size'] / self.total_slots
            
                # Factor 3: Prefer slots that minimize the number of fragments
                fragment_score = 1 - (metrics['num_slices'] / self.total_slots)
                
                # Factor 4: Prefer slots that maintain balanced utilization
                util_before = np.sum(self.spectrum[edge]) / self.total_slots
                util_after = np.sum(spectrum) / self.total_slots
                util_score = 1 - abs(util_after - util_before)
                edge_utilizations.append(util_after)
                
                # Factor 5: Edge balance (new)
                # Prefer slots that maintain similar utilization across edges
                edge_balance_score = 1 - np.std(edge_utilizations) if len(edge_utilizations) > 1 else 1.0
                
                # Combine scores with adaptive weights
                combined_score = (
                    self.weights['fragmentation'] * frag_score +
                    self.weights['block_size'] * block_score +
                    self.weights['fragments'] * fragment_score +
                    self.weights['utilization'] * util_score +
                    self.weights['edge_balance'] * edge_balance_score
                )
                
                score += combined_score
            
            # Average score across all edges
            slot_scores[slot] = score / (len(path) - 1)
        
        # Update network metrics
        if slot_scores:
            best_slot = max(slot_scores.items(), key=lambda x: x[1])[0]
            self.network_metrics['total_allocations'] += 1
            self.network_metrics['avg_utilization'] = (
                (self.network_metrics['avg_utilization'] * (self.network_metrics['total_allocations'] - 1) +
                 np.mean([np.sum(self.spectrum[edge]) / self.total_slots for edge in self.spectrum])) /
                self.network_metrics['total_allocations']
            )
            self.network_metrics['avg_fragmentation'] = (
                (self.network_metrics['avg_fragmentation'] * (self.network_metrics['total_allocations'] - 1) +
                 np.mean([self._calculate_fragmentation(edge) for edge in self.spectrum])) /
                self.network_metrics['total_allocations']
            )
            return best_slot
            
        return None

    def _calculate_fragmentation_metrics(self, spectrum: np.ndarray) -> dict:
        """Calculate fragmentation metrics for a spectrum array."""
        # Find continuous available slices (0s in the spectrum)
        available_slices = []
        current_slice_start = None
        
        for i, slot in enumerate(spectrum):
            if slot == 0:  # Available slot
                if current_slice_start is None:
                    current_slice_start = i
            else:  # Occupied slot
                if current_slice_start is not None:
                    available_slices.append((current_slice_start, i - 1))
                    current_slice_start = None
        
        # Handle case where spectrum ends with available slots
        if current_slice_start is not None:
            available_slices.append((current_slice_start, len(spectrum) - 1))
        
        # Calculate metrics
        if not available_slices:
            return {
                'num_slices': 0,
                'avg_slice_size': 0,
                'max_slice_size': 0,
                'min_slice_size': 0,
                'total_available': 0,
                'fragmentation_index': 1.0
            }
        
        slice_sizes = [end - start + 1 for start, end in available_slices]
        total_available = sum(slice_sizes)
        num_slices = len(available_slices)
        
        # Calculate fragmentation index using entropy-based metric
        if total_available > 0:
            # Normalize slice sizes
            normalized_sizes = [size / total_available for size in slice_sizes]
            # Calculate entropy
            entropy = -sum(p * math.log2(p) for p in normalized_sizes)
            # Normalize entropy to [0,1] range
            max_entropy = math.log2(num_slices) if num_slices > 1 else 0
            fragmentation_index = entropy / max_entropy if max_entropy > 0 else 0
        else:
            fragmentation_index = 1.0
        
        return {
            'num_slices': num_slices,
            'avg_slice_size': sum(slice_sizes) / num_slices,
            'max_slice_size': max(slice_sizes),
            'min_slice_size': min(slice_sizes),
            'total_available': total_available,
            'fragmentation_index': fragmentation_index
        }

    def find_feasible_slots(self, path: List[str], width: int) -> List[int]:
        """Find all feasible starting slots for a given path and width."""
        # First try to find the best slot considering fragmentation
        best_slot = self._find_best_slot(path, width)
        if best_slot is not None:
            return [best_slot]
            
        # If no optimal slot found, fall back to finding any feasible slot
        feasible_slots = []
        for start_slot in range(self.total_slots - width + 1):
            if self._is_slot_available(path, start_slot, width):
                feasible_slots.append(start_slot)
        return feasible_slots
    
    def _is_slot_available(self, path: List[str], start_slot: int, width: int) -> bool:
        """Check if slots are available for the entire path."""
        if start_slot + width > self.total_slots:
            return False
            
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            if edge not in self.spectrum:
                logger.warning(f"Edge {edge} not found in spectrum")
                return False
            if np.any(self.spectrum[edge][start_slot:start_slot + width] != 0):
                return False
        return True
        
    def check_spectrum_availability(self, path: List[str], width: int) -> bool:
        """Check if spectrum is available for a path with given width."""
        # Find feasible slots
        feasible_slots = self.find_feasible_slots(path, width)
        return len(feasible_slots) > 0

    def has_available_spectrum(self, path: List[str], bandwidth: float) -> bool:
        """Check if spectrum is available for a path with given bandwidth."""
        # Calculate required slots
        required_slots = self.calculate_required_slots(bandwidth, ModulationFormat.QPSK)  # Start with QPSK
        return self.check_spectrum_availability(path, required_slots)

    def allocate_spectrum(self, path: List[str], bandwidth: float) -> Optional[List[int]]:
        """
        Allocate spectrum slots for a path.
        
        Args:
            path: List of nodes in the path
            bandwidth: Required bandwidth in Gbps
            
        Returns:
            List of allocated slot indices if successful, None otherwise
        """
        # Calculate required slots
        required_slots = self.calculate_required_slots(bandwidth, ModulationFormat.QPSK)  # Start with QPSK
        
        # Find the best slot
        start_slot = self._find_best_slot(path, required_slots)
        if start_slot is None:
            return None
            
        if not self._is_slot_available(path, start_slot, required_slots):
            return None
            
        # Allocate slots
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            self.spectrum[edge][start_slot:start_slot + required_slots] = 1
            
        # Calculate and store fragmentation
        fragmentation = sum(self._calculate_fragmentation((str(u), str(v))) 
                          for u, v in zip(path[:-1], path[1:]))
        self.fragmentation_history.append(fragmentation)
        
        # Return allocated slots
        return list(range(start_slot, start_slot + required_slots))

    def release_spectrum(self, path: List[str], start_slot: int, width: int) -> None:
        """Release allocated spectrum slots."""
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            self.spectrum[edge][start_slot:start_slot + width] = 0

    def fragmentation_penalty(self, path: List[str], width: int) -> float:
        """Calculate fragmentation penalty for a path."""
        if not self.find_feasible_slots(path, width):
            return float('inf')
            
        # Count gaps between allocated slots
        gaps = 0
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            spectrum = self.spectrum[edge]
            for i in range(1, len(spectrum)):
                if spectrum[i] == 0 and spectrum[i-1] == 1:
                    gaps += 1
                    
        return gaps / self.total_slots
    
    def get_spectrum_utilization(self) -> float:
        """Calculate overall spectrum utilization."""
        total_slots = len(self.spectrum) * self.total_slots
        used_slots = sum(np.sum(spectrum) for spectrum in self.spectrum.values())
        return (used_slots / total_slots) * 100

    def bandwidth_fragmentation_ratio(self) -> float:
        """
        Bandwidth Fragmentation Ratio (BFR) as per EON literature:
        BFR = 1 - (largest contiguous free block / total free slots), averaged over all links.
        """
        ratios = []
        for slots in self.spectrum.values():
            free_blocks = [len(block) for block in np.split(slots, np.where(slots == 1)[0]) if np.all(block == 0)]
            total_free = sum(free_blocks)
            if total_free > 0:
                largest_block = max(free_blocks)
                ratios.append(1 - (largest_block / total_free))
        return float(np.mean(ratios)) if ratios else 0.0

    def entropy_fragmentation(self) -> float:
        """
        Entropy-based fragmentation metric (EBFM):
        Entropy = -sum(p_i * log2(p_i)), where p_i is the fraction of free slots in block i.
        """
        entropies = []
        for slots in self.spectrum.values():
            free_blocks = [len(block) for block in np.split(slots, np.where(slots == 1)[0]) if np.all(block == 0)]
            total = sum(free_blocks)
            if total > 0 and len(free_blocks) > 1:
                probs = [size / total for size in free_blocks]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                entropies.append(entropy)
        return float(np.mean(entropies)) if entropies else 0.0

    def external_fragmentation(self, demand_width: int = 4) -> float:
        """
        Demand-size-aware external fragmentation metric (EFM):
        Fraction of free blocks smaller than the demand width.
        """
        total_free_blocks = 0
        small_free_blocks = 0
        
        for slots in self.spectrum.values():
            free_blocks = [len(block) for block in np.split(slots, np.where(slots == 1)[0]) if np.all(block == 0)]
            total_free_blocks += len(free_blocks)
            small_free_blocks += sum(1 for block in free_blocks if block < demand_width)
            
        return small_free_blocks / total_free_blocks if total_free_blocks > 0 else 0.0

    def get_spectrum_state(self) -> Dict[str, object]:
        """Get current spectrum state."""
        return {
            'utilization': self.get_spectrum_utilization(),
            'fragmentation': {
                'bandwidth_ratio': self.bandwidth_fragmentation_ratio(),
                'entropy': self.entropy_fragmentation(),
                'external': self.external_fragmentation()
            },
            'spectrum': {str(edge): slots.tolist() for edge, slots in self.spectrum.items()}
        }

    def find_contiguous_free_blocks(self, path: List[str], required_slots: int) -> List[List[int]]:
        """
        Find all contiguous free slot blocks of size >= required_slots available on all links of the path.
        Returns a list of slot index lists (e.g., [[0,1,2,3], [10,11,12,13], ...])
        """
        # For each edge, find available blocks
        edge_blocks = []
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            spectrum = self.spectrum[edge]
            blocks = []
            start = None
            for i, slot in enumerate(spectrum):
                if slot == 0:
                    if start is None:
                        start = i
                else:
                    if start is not None and i - start >= required_slots:
                        blocks.append((start, i - 1))
                    start = None
            if start is not None and len(spectrum) - start >= required_slots:
                blocks.append((start, len(spectrum) - 1))
            edge_blocks.append(blocks)
        # Find intersection of available blocks across all edges
        if not edge_blocks:
            return []
        # Convert each edge's blocks to set of possible start indices
        edge_starts = []
        for blocks in edge_blocks:
            starts = set()
            for start, end in blocks:
                for s in range(start, end - required_slots + 2):
                    starts.add(s)
            edge_starts.append(starts)
        # Intersection of all start indices
        common_starts = set.intersection(*edge_starts) if edge_starts else set()
        # Return as list of slot index lists
        return [[s + i for i in range(required_slots)] for s in sorted(common_starts)]

    def allocate_specific_slots(self, path: List[str], slots: List[int]) -> bool:
        """
        Allocate the given slot indices on all links of the path. Returns True if successful, False otherwise.
        """
        # Check availability first
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            spectrum = self.spectrum[edge]
            if any(spectrum[slot] != 0 for slot in slots):
                return False
        # Allocate
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            self.spectrum[edge][slots] = 1
        return True

    def bandwidth_to_slots(self, bandwidth: float, modulation_format: ModulationFormat) -> int:
        """
        Convert bandwidth and modulation format to required number of slots (with guard bands).
        """
        return self.calculate_required_slots(bandwidth, modulation_format)

    def get_link_utilization(self, u: str, v: str) -> float:
        """
        Get the spectrum utilization for a specific link.

        Args:
            u: Start node of the link
            v: End node of the link

        Returns:
            Spectrum utilization (0.0 to 1.0)
        """
        edge = tuple(sorted((u, v)))
        if edge not in self.spectrum:
            raise ValueError(f"Link ({u}, {v}) not found in spectrum database.")
        
        spectrum = self.spectrum[edge]
        return np.sum(spectrum != 0) / self.total_slots

    def get_link_fragmentation(self, u: str, v: str) -> float:
        """
        Get the fragmentation index for a specific link.

        Args:
            u: Start node of the link
            v: End node of the link

        Returns:
            Fragmentation index
        """
        edge = tuple(sorted((u, v)))
        if edge not in self.spectrum:
            raise ValueError(f"Link ({u}, {v}) not found in spectrum database.")
            
        spectrum = self.spectrum[edge]
        metrics = self._calculate_fragmentation_metrics(spectrum)
        return metrics['fragmentation_index']

    def compute_global_ssfm(self) -> float:
        """
        Compute Global Spectrum Slot Fragmentation Metric (SSFM) across the entire network.
        
        Global SSFM measures the overall fragmentation state of the network by considering
        the fragmentation across all links and their interconnections.
        
        Returns:
            Global SSFM value (0.0 = no fragmentation, 1.0 = maximum fragmentation)
        """
        if not self.spectrum:
            return 0.0
            
        # Calculate individual link fragmentations
        link_fragmentations = []
        for edge in self.spectrum:
            spectrum = self.spectrum[edge]
            metrics = self._calculate_fragmentation_metrics(spectrum)
            link_fragmentations.append(metrics['fragmentation_index'])
        
        # Calculate average fragmentation
        avg_fragmentation = float(np.mean(link_fragmentations))
        
        # Calculate fragmentation variance (indicates uneven distribution)
        frag_variance = float(np.var(link_fragmentations)) if len(link_fragmentations) > 1 else 0.0
        
        # Calculate network-wide utilization
        total_utilization = float(self.get_spectrum_utilization()) / 100.0
        
        # Global SSFM combines average fragmentation, variance, and utilization
        # Higher variance and lower utilization contribute to higher fragmentation
        global_ssfm = (
            0.6 * avg_fragmentation +      # Average fragmentation (60% weight)
            0.3 * frag_variance +          # Fragmentation variance (30% weight)
            0.1 * (1.0 - total_utilization)  # Low utilization penalty (10% weight)
        )
        
        return float(min(global_ssfm, 1.0))  # Ensure value is in [0, 1] range

    def compute_local_ssfm(self, path: List[str]) -> float:
        """
        Compute Local Spectrum Slot Fragmentation Metric (SSFM) for a specific path.
        
        Local SSFM measures the fragmentation state along the specific path
        that would be used for a connection.
        
        Args:
            path: List of nodes in the path
            
        Returns:
            Local SSFM value (0.0 = no fragmentation, 1.0 = maximum fragmentation)
        """
        if len(path) < 2:
            return 0.0
            
        # Calculate fragmentation for each link in the path
        path_fragmentations = []
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            if edge not in self.spectrum:
                logger.warning(f"Edge {edge} not found in spectrum for path {path}")
                continue
                
            spectrum = self.spectrum[edge]
            metrics = self._calculate_fragmentation_metrics(spectrum)
            path_fragmentations.append(metrics['fragmentation_index'])
        
        if not path_fragmentations:
            return 0.0
            
        # Calculate average fragmentation along the path
        avg_path_fragmentation = float(np.mean(path_fragmentations))
        
        # Calculate path utilization (average utilization across path links)
        path_utilizations = []
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            if edge in self.spectrum:
                spectrum = self.spectrum[edge]
                utilization = np.sum(spectrum != 0) / self.total_slots
                path_utilizations.append(utilization)
        
        avg_path_utilization = float(np.mean(path_utilizations)) if path_utilizations else 0.0
        
        # Local SSFM combines path fragmentation and utilization
        local_ssfm = (
            0.8 * avg_path_fragmentation +     # Path fragmentation (80% weight)
            0.2 * (1.0 - avg_path_utilization)  # Low utilization penalty (20% weight)
        )
        
        return float(min(local_ssfm, 1.0))  # Ensure value is in [0, 1] range

    def get_modulation_efficiency(self, modulation_format: ModulationFormat) -> float:
        """
        Get modulation efficiency for a given modulation format.
        
        Modulation efficiency is calculated as bits per symbol normalized by
        the maximum possible efficiency (QAM64 = 6 bits/symbol).
        
        Args:
            modulation_format: The modulation format to evaluate
            
        Returns:
            Modulation efficiency value (0.0 to 1.0, where 1.0 is most efficient)
        """
        if modulation_format not in self.modulation_efficiency:
            raise ValueError(f"Unsupported modulation format: {modulation_format}")
        
        # Get bits per symbol for the modulation format
        bits_per_symbol = self.modulation_efficiency[modulation_format]
        
        # Normalize by maximum efficiency (QAM64 = 6 bits/symbol)
        max_efficiency = max(self.modulation_efficiency.values())
        efficiency = bits_per_symbol / max_efficiency
        
        return efficiency


def calculate_mwu_reward(
    spectrum_manager: SpectrumManager,
    path: List[str],
    allocated_slots: List[int],
    modulation_format: ModulationFormat,
    allocation_successful: bool,
    ssfm_before: float,
    global_ssfm_before: float
) -> float:
    """
    Calculate reward for MWU-RSA algorithm using the new reward formula.
    
    Reward components:
    - Local SSFM: SSFM change only on the path used
    - Global SSFM: SSFM change across the entire network  
    - U: Spectrum compaction (lower slot indices)
    - S: Success bonus
    - M: Modulation efficiency
    
    Args:
        spectrum_manager: The spectrum manager instance
        path: List of nodes in the path
        allocated_slots: List of allocated slot indices
        modulation_format: Modulation format used
        allocation_successful: Whether allocation was successful
        ssfm_before: Local SSFM before allocation
        global_ssfm_before: Global SSFM before allocation
        
    Returns:
        Calculated reward value
    """
    # Calculate SSFM after allocation
    ssfm_after = spectrum_manager.compute_local_ssfm(path)
    global_ssfm_after = spectrum_manager.compute_global_ssfm()
    
    # Calculate spectrum compaction (U)
    # Lower slot indices are preferred (closer to 0)
    if allocated_slots:
        avg_slot_index = np.mean(allocated_slots)
        total_slots = spectrum_manager.total_slots
        compaction_score = -avg_slot_index / total_slots
    else:
        compaction_score = 0.0
    
    # Success bonus (S)
    success_bonus = 1.0 if allocation_successful else -1.0
    
    # Modulation efficiency (M)
    modulation_efficiency = spectrum_manager.get_modulation_efficiency(modulation_format)
    
    # Calculate reward using the specified formula
    reward = (
        0.3 * (ssfm_before - ssfm_after) +              # Local SSFM improvement
        0.3 * (global_ssfm_before - global_ssfm_after) +  # Global SSFM improvement
        0.2 * compaction_score +                        # Spectrum compaction
        0.1 * success_bonus +                           # Success bonus
        0.1 * modulation_efficiency                     # Modulation efficiency
    )
    
    return float(reward)


def calculate_mwu_reward_with_simulation(
    spectrum_manager: SpectrumManager,
    path: List[str],
    required_slots: int,
    modulation_format: ModulationFormat,
    start_slot: int
) -> Tuple[float, bool]:
    """
    Calculate reward for MWU-RSA algorithm with spectrum simulation.
    
    This function simulates the allocation, calculates the reward, and then
    reverts the changes to avoid affecting the actual spectrum state.
    
    Args:
        spectrum_manager: The spectrum manager instance
        path: List of nodes in the path
        required_slots: Number of slots required
        modulation_format: Modulation format to use
        start_slot: Starting slot index for allocation
        
    Returns:
        Tuple of (reward, allocation_successful)
    """
    # Store current spectrum state
    original_spectrum = {}
    for edge in spectrum_manager.spectrum:
        original_spectrum[edge] = spectrum_manager.spectrum[edge].copy()
    
    # Calculate SSFM before allocation
    ssfm_before = spectrum_manager.compute_local_ssfm(path)
    global_ssfm_before = spectrum_manager.compute_global_ssfm()
    
    # Check if allocation is possible
    allocation_successful = spectrum_manager._is_slot_available(path, start_slot, required_slots)
    
    if allocation_successful:
        # Simulate allocation
        for u, v in zip(path[:-1], path[1:]):
            edge = tuple(sorted((u, v)))
            spectrum_manager.spectrum[edge][start_slot:start_slot + required_slots] = 1
        
        # Calculate allocated slots
        allocated_slots = list(range(start_slot, start_slot + required_slots))
        
        # Calculate reward
        reward = calculate_mwu_reward(
            spectrum_manager=spectrum_manager,
            path=path,
            allocated_slots=allocated_slots,
            modulation_format=modulation_format,
            allocation_successful=True,
            ssfm_before=ssfm_before,
            global_ssfm_before=global_ssfm_before
        )
        
        # Restore original spectrum state
        for edge in original_spectrum:
            spectrum_manager.spectrum[edge] = original_spectrum[edge]
    else:
        # Allocation failed - calculate penalty
        reward = calculate_mwu_reward(
            spectrum_manager=spectrum_manager,
            path=path,
            allocated_slots=[],
            modulation_format=modulation_format,
            allocation_successful=False,
            ssfm_before=ssfm_before,
            global_ssfm_before=global_ssfm_before
        )
    
    return reward, allocation_successful
