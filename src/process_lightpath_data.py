# process_lightpath_data.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from eon_models import ModulationFormat, EONLink
import os
from pathlib import Path
from eon_models import ModulationFormat
from ml_qot import QoTFeatures, MLQoTEstimator
from lightpath_reader import LightpathReader

logger = logging.getLogger(__name__)

def process_lightpath_dataset(file_path: str, max_samples: int = 10000) -> List[Tuple[EONLink, float]]:
    """Process lightpath dataset and convert to training format."""
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        return []

    logger.info(f"Processing dataset: {file_path}")
    
    # Define column names
    column_names = ['path_length', 'laser_current', 'launch_power', 'osnr', 'ber', 'failure_type']
    
    # Process in chunks to handle large files
    chunk_size = 2000
    total_rows = 0
    all_data = []
    
    logger.info("Loading data in chunks...")
    for chunk in pd.read_csv(file_path, sep=r'\s+', names=column_names, skiprows=1, chunksize=chunk_size):
        all_data.append(chunk)
        total_rows += len(chunk)
        logger.info(f"Loaded {total_rows} rows...")
    
    logger.info(f"Loaded {total_rows} rows total")
    
    # Combine all chunks
    df = pd.concat(all_data, ignore_index=True)
    
    # Sample if dataset is too large
    if len(df) > max_samples:
        logger.info(f"Dataset too large ({len(df)} samples). Sampling {max_samples} samples...")
        df = df.sample(n=max_samples, random_state=42)
    
    logger.info("\nDataset Overview:")
    logger.info(f"Number of samples: {len(df)}")
    
    # Print column statistics
    logger.info("\nColumn Statistics:")
    logger.info(df.describe())
    
    # Convert to training format
    logger.info("\nConverting to training format...")
    training_data = []
    
    # Process in batches
    batch_size = 1000
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            # Create EONLink object - ensure length is float
            link = EONLink(
                length=float(row['path_length']),  # Convert to float to avoid type issues
                fiber_type="SMF-28"  # Default fiber type
            )
            
            # Use OSNR as QoT metric
            qot = float(row['osnr'])  # Also ensure qot is float
            
            training_data.append((link, qot))
        
        logger.info(f"Processed batch {(i//batch_size)+1}/{num_batches} ({((i//batch_size)+1)/num_batches*100:.1f}%)")
    
    logger.info(f"Processed {len(training_data)} samples")
    
    # Print failure type distribution
    failure_counts = df['failure_type'].value_counts()
    logger.info("\nFailure Type Distribution:")
    failure_name_map = {
        0: "No failure",
        1: "ECL failure",
        2: "EDFA failure",
        3: "NLI failure"
    }
    
    for failure_type, count in failure_counts.items():
        percentage = (count / len(df)) * 100
        # Convert failure_type to string first, then handle
        failure_type_str = str(failure_type)
        if failure_type_str.isdigit():
            failure_type_int = int(failure_type_str)
            failure_name = failure_name_map.get(failure_type_int, f"Unknown ({failure_type})")
        else:
            failure_name = f"Unknown ({failure_type})"
        logger.info(f"{failure_name}: {count} samples ({percentage:.1f}%)")
    
    return training_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process lightpath dataset and train QoT model")
    parser.add_argument("--file", required=True, help="Path to lightpath dataset file")
    parser.add_argument("--model-dir", default="models", help="Directory to save trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum number of samples to use")
    
    args = parser.parse_args()
    
    process_lightpath_dataset(args.file, args.max_samples) 
