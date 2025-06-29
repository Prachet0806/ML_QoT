import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from ml_qot import MLQoTEstimator, QoTFeatures
from eon_models import EONLink, ModulationFormat
from eon_control import EONController
import networkx as nx

def load_network_data(network_file: str) -> nx.Graph:
    """Load network topology from file"""
    with open(network_file, 'r') as f:
        network_data = json.load(f)
        
    G = nx.Graph()
    
    
    for node in network_data['nodes']:
        G.add_node(node['id'], **node['properties'])
        
    for link in network_data['links']:
        G.add_edge(
            link['source'],
            link['target'],
            length=link['length'],
            **link['properties']
        )  
    return G

def load_training_data(data_file: str) -> List[Dict]:
    """Load real QoT measurements from file"""
    with open(data_file, 'r') as f:
        data = json.load(f)
        return data['samples']  #

def prepare_training_data(network_data: List[Dict], 
                         links: Dict[Tuple[str, str], EONLink]) -> List[Tuple[QoTFeatures, float]]:
    """Convert network data to training samples"""
    training_data = []
    
    for sample in network_data:
        path = sample['path']
        
        launch_power = sample['launch_power']
        channel_spacing = sample['channel_spacing']
        num_channels = sample['num_channels']
        modulation = ModulationFormat[sample['modulation']]
        
        # Create features
        features = QoTFeatures(
            path_length=sample['path_length'],
            num_hops=len(path) - 1,
            num_channels=num_channels,
            channel_spacing=channel_spacing,
            launch_power=launch_power,
            modulation=modulation,
            fiber_type=sample['fiber_type'],
            num_amplifiers=sample['num_amplifiers'],
            total_loss=sample['total_loss'],
            total_dispersion=sample['total_dispersion'],
            total_pmd=sample['total_pmd'],
            temperature=sample['temperature'],
            fiber_age=sample['fiber_age'],
            num_filters=sample['num_filters'],
            filter_bandwidth=sample['filter_bandwidth'],
            wss_loss=sample['wss_loss'],
            node_loss=sample['node_loss']
        )
        
        qot = sample['measured_qot']
        
        training_data.append((features, qot))
        
    return training_data

def train_model(network_file: str, 
                training_file: str,
                model_dir: str = "models") -> None:
    """Train QoT estimator on real network data"""
    print("Loading network topology...")
    G = load_network_data(network_file)
    
    print("Initializing EON controller...")
    controller = EONController(G)
    
    print("Loading training data...")
    network_data = load_training_data(training_file)
    
    print("Preparing training samples...")
    training_data = prepare_training_data(network_data, controller.links)
    
    print("Training QoT estimator...")
    controller.train_qot_estimator(training_data)
    
    print("Saving trained model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    controller.qot_estimator.save_model(f"qot_model_real_{timestamp}")
    
    print("\nTraining completed successfully!")
    print(f"Model saved in: {model_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train QoT estimator on real network data")
    parser.add_argument("--network", required=True, help="Network topology file (JSON)")
    parser.add_argument("--data", required=True, help="Training data file (JSON)")
    parser.add_argument("--model-dir", default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    train_model(args.network, args.data, args.model_dir) 
