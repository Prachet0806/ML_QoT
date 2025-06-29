import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
from eon_models import ModulationFormat
from ml_qot import QoTFeatures

@dataclass
class LightpathData:
    """Lightpath measurement data"""
    path: List[str]
    path_length: float
    launch_power: float
    channel_spacing: float
    num_channels: int
    modulation: ModulationFormat
    fiber_type: str
    num_amplifiers: int
    total_loss: float
    total_dispersion: float
    total_pmd: float
    temperature: float
    fiber_age: float
    num_filters: int
    filter_bandwidth: float
    wss_loss: float
    node_loss: float
    measured_qot: float
    timestamp: datetime

class LightpathReader:
    """Reader for lightpath measurement data"""
    def __init__(self, data_dir: str = "data/lightpath_dataset"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def read_csv(self, file_path: str) -> List[LightpathData]:
        """Read lightpath data from CSV file"""
        df = pd.read_csv(file_path)
        lightpaths = []
        
        for _, row in df.iterrows():
            # Convert path string to list
            path_str = str(row['path'])
            path = path_str.strip('[]').split(',')
            path = [node.strip() for node in path]
            
            # Convert modulation string to enum
            modulation_str = str(row['modulation'])
            modulation = ModulationFormat[modulation_str]
            
            # Convert timestamp string to datetime
            timestamp = pd.to_datetime(row['timestamp']).item()
            
            lightpath = LightpathData(
                path=path,
                path_length=float(row['path_length']),
                launch_power=float(row['launch_power']),
                channel_spacing=float(row['channel_spacing']),
                num_channels=int(row['num_channels']),
                modulation=modulation,
                fiber_type=str(row['fiber_type']),
                num_amplifiers=int(row['num_amplifiers']),
                total_loss=float(row['total_loss']),
                total_dispersion=float(row['total_dispersion']),
                total_pmd=float(row['total_pmd']),
                temperature=float(row['temperature']),
                fiber_age=float(row['fiber_age']),
                num_filters=int(row['num_filters']),
                filter_bandwidth=float(row['filter_bandwidth']),
                wss_loss=float(row['wss_loss']),
                node_loss=float(row['node_loss']),
                measured_qot=float(row['measured_qot']),
                timestamp=timestamp
            )
            
            lightpaths.append(lightpath)
            
        return lightpaths
        
    def read_json(self, file_path: str) -> List[LightpathData]:
        """Read lightpath data from JSON file"""
        import json
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        lightpaths = []
        for item in data['lightpaths']:
            # Convert path string to list
            path = item['path']
            
            # Convert modulation string to enum
            modulation = ModulationFormat[item['modulation']]
            
            # Convert timestamp string to datetime
            timestamp = pd.to_datetime(item['timestamp']).item()
            
            lightpath = LightpathData(
                path=path,
                path_length=float(item['path_length']),
                launch_power=float(item['launch_power']),
                channel_spacing=float(item['channel_spacing']),
                num_channels=int(item['num_channels']),
                modulation=modulation,
                fiber_type=item['fiber_type'],
                num_amplifiers=int(item['num_amplifiers']),
                total_loss=float(item['total_loss']),
                total_dispersion=float(item['total_dispersion']),
                total_pmd=float(item['total_pmd']),
                temperature=float(item['temperature']),
                fiber_age=float(item['fiber_age']),
                num_filters=int(item['num_filters']),
                filter_bandwidth=float(item['filter_bandwidth']),
                wss_loss=float(item['wss_loss']),
                node_loss=float(item['node_loss']),
                measured_qot=float(item['measured_qot']),
                timestamp=timestamp
            )
            
            lightpaths.append(lightpath)
            
        return lightpaths
        
    def convert_to_training_data(self, lightpaths: List[LightpathData]) -> List[Tuple[QoTFeatures, float]]:
        """Convert lightpath data to training samples"""
        training_data = []
        
        for lightpath in lightpaths:
            features = QoTFeatures(
                path_length=lightpath.path_length,
                num_hops=len(lightpath.path) - 1,
                num_channels=lightpath.num_channels,
                channel_spacing=lightpath.channel_spacing,
                launch_power=lightpath.launch_power,
                modulation=lightpath.modulation,
                fiber_type=lightpath.fiber_type,
                num_amplifiers=lightpath.num_amplifiers,
                total_loss=lightpath.total_loss,
                total_dispersion=lightpath.total_dispersion,
                total_pmd=lightpath.total_pmd,
                temperature=lightpath.temperature,
                fiber_age=lightpath.fiber_age,
                num_filters=lightpath.num_filters,
                filter_bandwidth=lightpath.filter_bandwidth,
                wss_loss=lightpath.wss_loss,
                node_loss=lightpath.node_loss
            )
            
            training_data.append((features, lightpath.measured_qot))
            
        return training_data
        
    def analyze_data(self, lightpaths: List[LightpathData]) -> Dict:
        """Analyze lightpath data"""
        df = pd.DataFrame([vars(lp) for lp in lightpaths])
        
        analysis = {
            'num_samples': len(lightpaths),
            'qot_stats': {
                'mean': df['measured_qot'].mean(),
                'std': df['measured_qot'].std(),
                'min': df['measured_qot'].min(),
                'max': df['measured_qot'].max()
            },
            'modulation_counts': df['modulation'].value_counts().to_dict(),
            'fiber_type_counts': df['fiber_type'].value_counts().to_dict(),
            'path_length_stats': {
                'mean': df['path_length'].mean(),
                'std': df['path_length'].std(),
                'min': df['path_length'].min(),
                'max': df['path_length'].max()
            }
        }
        
        return analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Read and analyze lightpath data")
    parser.add_argument("--file", required=True, help="Path to data file (CSV or JSON)")
    parser.add_argument("--format", choices=['csv', 'json'], required=True, help="File format")
    
    args = parser.parse_args()
    
    reader = LightpathReader()
    
    if args.format == 'csv':
        lightpaths = reader.read_csv(args.file)
    else:
        lightpaths = reader.read_json(args.file)
        
    analysis = reader.analyze_data(lightpaths)
    
    print("\nData Analysis:")
    print(f"Number of samples: {analysis['num_samples']}")
    print("\nQoT Statistics (dB):")
    print(f"Mean: {analysis['qot_stats']['mean']:.2f}")
    print(f"Std: {analysis['qot_stats']['std']:.2f}")
    print(f"Min: {analysis['qot_stats']['min']:.2f}")
    print(f"Max: {analysis['qot_stats']['max']:.2f}")
    print("\nModulation Format Distribution:")
    for mod, count in analysis['modulation_counts'].items():
        print(f"{mod}: {count}")
    print("\nFiber Type Distribution:")
    for fiber, count in analysis['fiber_type_counts'].items():
        print(f"{fiber}: {count}")
    print("\nPath Length Statistics (km):")
    print(f"Mean: {analysis['path_length_stats']['mean']:.2f}")
    print(f"Std: {analysis['path_length_stats']['std']:.2f}")
    print(f"Min: {analysis['path_length_stats']['min']:.2f}")
    print(f"Max: {analysis['path_length_stats']['max']:.2f}") 
