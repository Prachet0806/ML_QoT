#eon_models.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy import constants

class ModulationFormat(Enum):
    """Supported modulation formats in the EON"""
    QPSK = "QPSK"
    QAM8 = "8-QAM"
    QAM16 = "16-QAM"
    QAM64 = "64-QAM"

@dataclass
class TransceiverSpec:
    """Specifications for a transceiver"""
    modulation: ModulationFormat
    baud_rate: float  # Gbaud
    fec_overhead: float  # FEC overhead ratio
    power_consumption: float  # W
    cost: float  # Cost in arbitrary units
    min_osnr: float  # Minimum required OSNR in dB
    max_reach: float  # Maximum reach in km
    
    @property
    def spectral_efficiency(self) -> float:
        """Calculate spectral efficiency in bits/s/Hz"""
        bits_per_symbol = {
            ModulationFormat.QPSK: 2,
            ModulationFormat.QAM8: 3,
            ModulationFormat.QAM16: 4,
            ModulationFormat.QAM64: 6
        }
        return bits_per_symbol[self.modulation] * (1 - self.fec_overhead)

class WSSSpec:
    """Wavelength Selective Switch specifications"""
    def __init__(self,
                 num_ports: int = 9,
                 insertion_loss: float = 5.0,  # dB
                 filter_bandwidth: float = 37.5,  # GHz
                 filter_rolloff: float = 0.1,  # Roll-off factor
                 crosstalk: float = -40.0):  # dB
        self.num_ports = num_ports
        self.insertion_loss = insertion_loss
        self.filter_bandwidth = filter_bandwidth
        self.filter_rolloff = filter_rolloff
        self.crosstalk = crosstalk

class EONNode:
    """EON node with ROADM capabilities"""
    def __init__(self,
                 node_id: str,
                 wss_spec: WSSSpec,
                 num_add_drop_ports: int = 4):
        self.node_id = node_id
        self.wss_spec = wss_spec
        self.num_add_drop_ports = num_add_drop_ports
        self.connected_links: Dict[str, 'EONLink'] = {}
        
    def add_link(self, link: 'EONLink', direction: str) -> None:
        """Add a link to the node"""
        self.connected_links[direction] = link

class FlexibleGrid:
    """Flexible grid system for EON"""
    def __init__(self, 
                 start_freq: float = 191.3,  # THz
                 end_freq: float = 196.1,    # THz
                 slot_width: float = 12.5):  # GHz
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.slot_width = slot_width
        self.total_slots = int((end_freq - start_freq) * 1000 / slot_width)
        self.occupied_slots = np.zeros(self.total_slots, dtype=bool)
        
    def allocate_spectrum(self, 
                         center_freq: float, 
                         bandwidth: float) -> Optional[Tuple[int, int]]:
        """
        Allocate spectrum slots for a channel
        
        Args:
            center_freq: Center frequency in THz
            bandwidth: Required bandwidth in GHz
            
        Returns:
            Tuple of (start_slot, end_slot) if allocation successful, None otherwise
        """
        # Convert frequency to slot numbers
        center_slot = int((center_freq - self.start_freq) * 1000 / self.slot_width)
        num_slots = int(np.ceil(bandwidth / self.slot_width))
        
        # Check if allocation is possible
        if center_slot < 0 or center_slot + num_slots > self.total_slots:
            return None
            
        # Check if slots are available
        if np.any(self.occupied_slots[center_slot:center_slot + num_slots]):
            return None
            
        # Allocate slots
        self.occupied_slots[center_slot:center_slot + num_slots] = True
        return (center_slot, center_slot + num_slots)
        
    def release_spectrum(self, start_slot: int, end_slot: int) -> None:
        """Release allocated spectrum slots"""
        self.occupied_slots[start_slot:end_slot] = False
        
    def get_available_spectrum(self) -> List[Tuple[int, int]]:
        """Get list of available spectrum blocks"""
        available_blocks = []
        start = None
        
        for i in range(self.total_slots):
            if not self.occupied_slots[i]:
                if start is None:
                    start = i
            elif start is not None:
                available_blocks.append((start, i))
                start = None
                
        if start is not None:
            available_blocks.append((start, self.total_slots))
            
        return available_blocks

class EONLink:
    """EON link with flexible grid and transceiver support"""
    def __init__(self, length: float, fiber_type: str = "SMF-28"):
        self.length = length  # km
        self.fiber_type = fiber_type
        
        # Fiber parameters
        self.fiber_params = {
            "SMF-28": {
                "attenuation": 0.2,  # dB/km
                "dispersion": 17.0,  # ps/nm/km
                "pmd_coefficient": 0.1,  # ps/sqrt(km)
                "nonlinear_index": 2.6e-20,  # m²/W
                "effective_area": 80.0,  # µm²
                "refractive_index": 1.4682
            },
            "LEAF": {
                "attenuation": 0.22,  # dB/km
                "dispersion": 4.0,  # ps/nm/km
                "pmd_coefficient": 0.08,  # ps/sqrt(km)
                "nonlinear_index": 2.6e-20,  # m²/W
                "effective_area": 72.0,  # µm²
                "refractive_index": 1.4682
            }
        }
        
        # Amplifier parameters
        self.amp_params = {
            "gain": 20.0,  # dB
            "noise_figure": 5.0,  # dB
            "saturation_power": 20.0,  # dBm
            "spacing": 80.0  # km
        }
        
        # Transceiver parameters
        self.tx_params = {
            "launch_power": 3.0,  # dBm
            "linewidth": 100.0,  # kHz
            "extinction_ratio": 10.0  # dB
        }
        
    def calculate_path_loss(self) -> float:
        """Calculate total path loss including fiber and splice losses."""
        fiber_loss = self.length * self.fiber_params[self.fiber_type]["attenuation"]
        splice_loss = 0.1 * (self.length / 2)  # 0.1 dB per splice, one every 2 km
        return fiber_loss + splice_loss
        
    def calculate_osnr(self, launch_power: float, num_amplifiers: int) -> float:
        """Calculate OSNR considering ASE noise."""
        # Calculate signal power at receiver
        path_loss = self.calculate_path_loss()
        signal_power = launch_power - path_loss
        
        # Add EDFA gain
        edfa_gain = num_amplifiers * 25.0  # 25 dB per amplifier
        signal_power += edfa_gain
        
        # Calculate noise power (thermal + ASE)
        thermal_noise = -174 + 10 * np.log10(12.5 * 1e9)  # dBm for 12.5 GHz channel
        ase_noise = num_amplifiers * 5.0  # 5 dB ASE noise per amplifier
        noise_power = thermal_noise + ase_noise
        
        # Calculate OSNR
        osnr = signal_power - noise_power
        
        return osnr
        
    def calculate_nonlinear_effects(self, launch_power: float, channel_spacing: float, num_channels: int) -> float:
        """Calculate nonlinear impairments (SPM, XPM, FWM)."""
        # Convert dBm to mW
        launch_power_mw = 10 ** (launch_power / 10)
        
        # Fiber parameters
        n2 = self.fiber_params[self.fiber_type]["nonlinear_index"]
        A_eff = self.fiber_params[self.fiber_type]["effective_area"]
        alpha = self.fiber_params[self.fiber_type]["attenuation"] / 4.343  # Convert to 1/km
        
        # Calculate effective length
        L_eff = (1 - np.exp(-alpha * self.length)) / alpha
        
        # Calculate nonlinear coefficient
        gamma = 2 * np.pi * n2 / (self.tx_params["linewidth"] * 1e3 * A_eff * 1e-12)
        
        # Calculate SPM penalty
        p_spm = 0.5 * gamma * launch_power_mw * L_eff
        
        # Calculate XPM penalty (simplified)
        p_xpm = 0.1 * p_spm * (num_channels - 1)
        
        # Calculate FWM penalty (simplified)
        p_fwm = 0.05 * p_spm * (num_channels - 1) * (num_channels - 2)
        
        # Convert to dB
        total_penalty_db = 10 * np.log10(1 + p_spm + p_xpm + p_fwm)
        
        return total_penalty_db
        
    def calculate_pmd_penalty(self) -> float:
        """Calculate PMD penalty."""
        pmd_coef = self.fiber_params[self.fiber_type]["pmd_coefficient"]
        pmd = pmd_coef * np.sqrt(self.length)
        
        # Calculate PMD penalty using simplified model
        b = 10e9  # Bit rate (10 Gbps)
        pmd_penalty = 0.5 * (pmd / (0.1 / b)) ** 2
        
        return pmd_penalty
        
    def calculate_total_qot(self,
                           launch_power: float,
                           channel_spacing: float,
                           num_channels: int) -> float:
        """Calculate total QoT considering all impairments."""
        # Calculate number of amplifiers based on link length (one every 80 km)
        num_amplifiers = max(1, int(np.ceil(self.length / 80)))
        
        # Calculate OSNR
        osnr = self.calculate_osnr(launch_power, num_amplifiers)
        
        # Calculate nonlinear impairments
        nli = self.calculate_nonlinear_effects(launch_power, channel_spacing, num_channels)
        
        # Calculate PMD penalty
        pmd = self.calculate_pmd_penalty()
        
        # Total QoT is OSNR minus all penalties
        total_qot = osnr - nli - pmd
        
        return total_qot

@dataclass
class QoTParameters:
    """QoT parameters for a lightpath"""
    osnr: float  # Optical Signal-to-Noise Ratio in dB
    ber: float  # Bit Error Rate
    q_factor: float  # Q-factor in dB
    nonlinear_penalty: float  # Nonlinear penalty in dB
    chromatic_dispersion: float  # Chromatic dispersion in ps/nm
    pmd: float  # Polarization Mode Dispersion in ps
    reach: float  # Maximum reach in km
    launch_power: float  # Launch power in dBm
    received_power: float  # Received power in dBm
