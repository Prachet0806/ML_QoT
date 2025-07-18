# generate_synthetic_data.py

import json
import numpy as np
from typing import List, Dict
import networkx as nx
from datetime import datetime
import os
from src.eon_models import ModulationFormat

def generate_network_topology(num_nodes: int = 10, 
                            avg_degree: float = 3.0) -> nx.Graph:
    """Generate a random network topology"""
    # Generate random graph
    G = nx.gnm_random_graph(num_nodes, int(num_nodes * avg_degree))
    
    # Add node properties
    for node in G.nodes():
        G.nodes[node]['type'] = np.random.choice(['ROADM', 'REGEN'])
        G.nodes[node]['has_regeneration'] = G.nodes[node]['type'] == 'REGEN'
        
    # Add edge properties
    for u, v in G.edges():
        G[u][v]['length'] = np.random.uniform(50, 200)  # km
        G[u][v]['fiber_type'] = np.random.choice(['SMF-28', 'NZDSF'])
        G[u][v]['temperature'] = np.random.normal(25, 5)  # °C
        G[u][v]['fiber_age'] = np.random.uniform(0, 10)  # years
        
    return G

def generate_training_samples(G: nx.Graph, 
                            num_samples: int = 1000) -> List[Dict]:
    """Generate synthetic training samples"""
    samples = []
    
    # Generate all possible paths up to 3 hops
    paths = []
    for src in G.nodes():
        for dst in G.nodes():
            if src != dst:
                paths.extend(list(nx.all_simple_paths(G, src, dst, cutoff=3)))
                
    # Convert paths to list of lists for numpy compatibility
    paths = [list(path) for path in paths]
    
    # Define modulation penalties
    modulation_penalties = {
        ModulationFormat.QPSK: 0,
        ModulationFormat.QAM8: 2,
        ModulationFormat.QAM16: 4,
        ModulationFormat.QAM64: 6
    }
    
    for _ in range(num_samples):
        # Randomly select a path
        path = paths[np.random.randint(0, len(paths))]
        
        # Calculate path metrics
        path_length = 0
        total_loss = 0
        total_dispersion = 0
        total_pmd = 0
        num_amplifiers = 0
        num_filters = 0
        total_wss_loss = 0
        total_node_loss = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = G[u][v]
            
            # Accumulate path metrics
            path_length += link['length']
            
            # Calculate losses
            fiber_loss = link['length'] * 0.2  # dB/km
            total_loss += fiber_loss
            
            # Calculate dispersion
            dispersion = link['length'] * 17  # ps/nm/km for SMF-28
            total_dispersion += dispersion
            
            # Calculate PMD
            pmd = link['length'] * 0.1  # ps/sqrt(km)
            total_pmd += pmd
            
            # Calculate number of amplifiers (one every 80 km, no amplifiers for paths < 80km)
            num_amplifiers += max(0, int(np.ceil(link['length'] / 80)))
            
            # Add node and filter effects
            num_filters += 2
            total_wss_loss += 5.0
            total_node_loss += 2.0
            
        # Generate random parameters
        launch_power = np.random.uniform(0, 5)  # dBm
        channel_spacing = np.random.choice([12.5, 25, 37.5, 50])  # GHz
        num_channels = np.random.randint(1, 10)
        # Convert ModulationFormat enum to list of strings for numpy compatibility
        modulation_formats = [mod.name for mod in ModulationFormat]
        modulation = ModulationFormat[np.random.choice(modulation_formats)]
        
        # Calculate QoT using the span model
        osnr = calculate_osnr_span_model(path_length, launch_power, channel_spacing)
        
        # Add modulation format penalty
        if modulation == ModulationFormat.QAM64:
            osnr -= 3.0
        elif modulation == ModulationFormat.QAM16:
            osnr -= 1.5
        elif modulation == ModulationFormat.QAM8:
            osnr -= 0.5
        
        # Add small random variation
        qot = osnr + np.random.normal(0, 0.5)
        
        # Create sample
        sample = {
            "path": [str(node) for node in path],
            "path_length": float(path_length),
            "launch_power": float(launch_power),
            "channel_spacing": float(channel_spacing),
            "num_channels": int(num_channels),
            "modulation": modulation.name,
            "fiber_type": G[path[0]][path[1]]['fiber_type'],
            "num_amplifiers": int(num_amplifiers),
            "total_loss": float(total_loss),
            "total_dispersion": float(total_dispersion),
            "total_pmd": float(total_pmd),
            "temperature": float(G[path[0]][path[1]]['temperature']),
            "fiber_age": float(G[path[0]][path[1]]['fiber_age']),
            "num_filters": int(num_filters),
            "filter_bandwidth": 37.5,
            "wss_loss": float(total_wss_loss),
            "node_loss": float(total_node_loss),
            "measured_qot": float(qot)
        }
        
        samples.append(sample)
        
    return samples

def save_data(G: nx.Graph, samples: List[Dict], output_dir: str) -> None:
    """Save network topology and training data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save network topology
    network_data = {
        "nodes": [
            {
                "id": str(node),
                "properties": {
                    "type": G.nodes[node]['type'],
                    "has_regeneration": G.nodes[node]['has_regeneration']
                }
            }
            for node in G.nodes()
        ],
        "links": [
            {
                "source": str(u),
                "target": str(v),
                "length": G[u][v]['length'],
                "properties": {
                    "fiber_type": G[u][v]['fiber_type'],
                    "temperature": G[u][v]['temperature'],
                    "fiber_age": G[u][v]['fiber_age']
                }
            }
            for u, v in G.edges()
        ]
    }
    
    with open(os.path.join(output_dir, "network_topology.json"), 'w') as f:
        json.dump(network_data, f, indent=2)
        
    # Save training data
    training_data = {"samples": samples}
    with open(os.path.join(output_dir, "training_data.json"), 'w') as f:
        json.dump(training_data, f, indent=2)

def generate_data(num_nodes: int = 10,
                 avg_degree: float = 3.0,
                 num_samples: int = 1000,
                 output_dir: str = "data/synthetic") -> None:
    """Generate synthetic data for training"""
    print("Generating network topology...")
    G = generate_network_topology(num_nodes, avg_degree)
    
    print("Generating training samples...")
    samples = generate_training_samples(G, num_samples)
    
    print("Saving data...")
    save_data(G, samples, output_dir)
    
    print(f"\nData generation completed!")
    print(f"Network topology: {output_dir}/network_topology.json")
    print(f"Training data: {output_dir}/training_data.json")
    print(f"Number of samples: {len(samples)}")

def calculate_osnr_span_model(length_km, launch_power_dbm=3.0, channel_spacing_ghz=12.5):
    """
    Calculate OSNR using industry-standard span model.
    This is the most accurate method for multi-span optical links.
    """
    # Physical constants
    h = 6.626e-34
    c = 3e8
    wavelength_nm = 1550
    freq = c / (wavelength_nm * 1e-9)
    bandwidth_hz = channel_spacing_ghz * 1e9

    # Span and amplifier setup
    span_length_km = 80  # Standard span length
    num_spans = int(np.ceil(length_km / span_length_km))
    fiber_loss_db = span_length_km * 0.2  # dB/km
    amp_gain_db = fiber_loss_db  # Each amp compensates for span loss
    amp_gain_linear = 10**(amp_gain_db / 10)
    noise_figure_db = 5.0
    noise_figure_linear = 10**(noise_figure_db / 10)
    n_sp = noise_figure_linear / 2

    # ASE noise per amp (in Watts)
    ase_per_amp_w = 2 * n_sp * h * freq * bandwidth_hz * (amp_gain_linear - 1)

    # Signal power at receiver (in mW)
    signal_power_dbm = launch_power_dbm  # Each span resets signal to launch power
    signal_power_mw = 10**(signal_power_dbm / 10)

    # Total ASE noise (in mW) after N spans
    total_ase_w = num_spans * ase_per_amp_w
    total_ase_mw = total_ase_w * 1e3

    # OSNR (linear)
    osnr_linear = signal_power_mw / total_ase_mw
    osnr_db = 10 * np.log10(osnr_linear)

    return osnr_db

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--degree", type=float, default=3.0, help="Average node degree")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    
    args = parser.parse_args()
    
    generate_data(args.nodes, args.degree, args.samples, args.output) 
