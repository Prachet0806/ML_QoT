# train_new_qot_model.py
import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import dataclasses as dataclass


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_qot_model import train_model, load_network_data, load_training_data, prepare_training_data
from eon_control import EONController
from generate_synthetic_data import generate_data, generate_training_samples, save_data
from lightpath_reader import LightpathReader
from ml_qot import MLQoTEstimator, QoTFeatures
import networkx as nx
from process_lightpath_data import process_lightpath_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_required_attributes(G: nx.Graph) -> nx.Graph:
    """Add required node and edge attributes to the graph"""
    
    for node in G.nodes():
        G.nodes[node]['type'] = 'ROADM'
    
    
    for u, v, data in G.edges(data=True):
        if 'length' not in data:
            if 'dist' in data:
                G[u][v]['length'] = data['dist']
            else:
                G[u][v]['length'] = 100.0  # fallback default
        if 'fiber_type' not in data:
            G[u][v]['fiber_type'] = 'SMF-28'
        if 'temperature' not in data:
            G[u][v]['temperature'] = 25.0
        if 'fiber_age' not in data:
            G[u][v]['fiber_age'] = 1.0
    
    return G

def convert_lightpath_data_to_qot_features(lightpath_data: List[Tuple], G: nx.Graph) -> List[Tuple[QoTFeatures, float]]:
    """Convert lightpath dataset output to QoTFeatures format"""
    from eon_models import ModulationFormat
    
    qot_features_data = []
    
    for link, qot_value in lightpath_data:
        path = ["node1", "node2"]  # Placeholder path
        
        # Ensure link.length is float to avoid type errors
        link_length = float(link.length) if not isinstance(link.length, (int, float)) else link.length
        
        features = QoTFeatures(
            path_length=link_length,
            num_hops=1,
            num_channels=1,
            channel_spacing=12.5,
            launch_power=0.0,
            modulation=ModulationFormat.QPSK,
            fiber_type=link.fiber_type,
            num_amplifiers=max(1, int(link_length / 80)),
            total_loss=link_length * 0.2,  # 0.2 dB/km
            total_dispersion=link_length * 17.0,  # 17 ps/nm/km
            total_pmd=link_length * 0.1,  # 0.1 ps/sqrt(km)
            temperature=25.0,
            fiber_age=1.0,
            num_filters=2,
            filter_bandwidth=37.5,
            wss_loss=5.0,
            node_loss=2.0
        )
        
        qot_features_data.append((features, qot_value))
    
    return qot_features_data

def main():
    # Create models directory inside src
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Initialize a single QoT estimator with the new models directory
    logger.info("Initializing QoT estimator...")
    qot_estimator = MLQoTEstimator(model_dir=str(models_dir))

    #Generate synthetic data using Germany50 topology
    logger.info("Generating synthetic data using Germany50 topology...")
    germany50_gml = "data/germany50.gml"
    G_germany50 = nx.read_gml(germany50_gml)
    G_germany50 = add_required_attributes(G_germany50)
    
    # Generate more samples for better training using the correct function
    germany50_samples = generate_training_samples(G_germany50, num_samples=50000)
    save_data(G_germany50, germany50_samples, output_dir="data/synthetic_germany50")

    # Train on synthetic data (Germany50 topology)
    logger.info("Training on synthetic data (Germany50 topology)...")
    G_germany50 = load_network_data("data/synthetic_germany50/network_topology.json")
    controller_germany50 = EONController(G_germany50)
    controller_germany50.qot_estimator = qot_estimator
    training_data_germany50 = load_training_data("data/synthetic_germany50/training_data.json")
    training_samples_germany50 = prepare_training_data(training_data_germany50, controller_germany50.links)
    controller_germany50.train_qot_estimator(training_samples_germany50)

    # Train on real-world data (lightpath756)
    logger.info("Training on real-world data (lightpath756)...")
    topology_file = "data/germany50.gml"
    G = None
    try:
        G = nx.read_gml(topology_file)
        G = add_required_attributes(G)
    except Exception as e:
        logger.error(f"Failed to load topology: {e}")
        return
    controller = EONController(G)
    controller.qot_estimator = qot_estimator  # Use our single estimator
    
    # Process lightpath dataset
    lightpath_file = "data/Lightpath_756_label_4_QoT_dataset_train_900.txt"
    if os.path.exists(lightpath_file):
        lightpath_data = process_lightpath_dataset(lightpath_file, max_samples=10000)
        if lightpath_data:
            # Convert lightpath data to QoTFeatures format
            qot_features_data = convert_lightpath_data_to_qot_features(lightpath_data, G)
            controller.train_qot_estimator(qot_features_data)
            logger.info("Trained on lightpath data")
    else:
        logger.warning(f"Lightpath dataset file not found: {lightpath_file}")

    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"qot_model_germany50_lightpath_{timestamp}"  # Descriptive name indicating data sources
    logger.info(f"Saving final model to {models_dir / f'{model_name}.joblib'}")
    qot_estimator.save_model(model_name)
    
    # Print model performance metrics
    if qot_estimator.model_metrics:
        logger.info("\nFinal Model Performance:")
        logger.info(f"R² Score: {qot_estimator.model_metrics.r2_score:.4f}")
        logger.info(f"RMSE: {qot_estimator.model_metrics.rmse:.4f}")
        logger.info(f"Training Time: {qot_estimator.model_metrics.training_time:.2f} seconds")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 
