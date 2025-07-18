# ML-QoT: Machine Learning-based Quality of Transmission Estimation for Elastic Optical Networks

## Overview

This project implements a comprehensive machine learning-based Quality of Transmission (QoT) estimation system for Elastic Optical Networks (EONs). The system uses both synthetic and real-world data to train models that can predict QoT parameters for optical lightpaths, enabling better resource allocation, network planning, and regeneration point optimization.

## Features

- **ML-based QoT Estimation**: Uses Random Forest and LightGBM models with smart ensemble prediction
- **Smart Ensemble Strategy**: Intelligent model selection based on prediction confidence and QoT thresholds
- **Regeneration Planning**: Advanced regeneration point optimization for long-haul networks
- **Flexible Network Topologies**: Support for custom network topologies and Germany50 topology
- **Real-world Data Integration**: Processes lightpath measurement data for training
- **Synthetic Data Generation**: Generates synthetic training data for various network scenarios
- **Spectrum Management**: Flexible grid spectrum allocation and management
- **Multiple Modulation Formats**: Support for QPSK, 8-QAM, 16-QAM, and 64-QAM
- **Path Optimization**: K-shortest path algorithms with QoT-aware routing
- **Industry-Standard Models**: Implements span-based OSNR calculations

## Project Structure

```
ML_QOT/
├── data/                          # Data files
│   ├── germany50.gml             # Germany50 network topology
│   ├── germany50_fixed.gml       # Fixed Germany50 topology
│   ├── synthetic_germany50/      # Generated synthetic data
│   └── Lightpath_756_label_4_QoT_dataset_train_900.txt  # Lightpath dataset (download separately)
├── src/                          # Source code
│   ├── ml_qot.py                 # Main ML QoT estimator with LightGBM and smart ensemble
│   ├── eon_models.py             # EON network models and components
│   ├── eon_control.py            # EON controller for network management
│   ├── spectrum_manager.py       # Spectrum allocation and management
│   ├── regenerator.py            # Regeneration point optimization
│   ├── topology_loader.py        # Network topology loading and path finding
│   ├── generate_synthetic_data.py # Synthetic data generation
│   ├── process_lightpath_data.py # Lightpath data processing
│   ├── lightpath_reader.py       # Lightpath data reader
│   ├── train_qot_model.py        # Basic QoT model training
│   ├── train_new_qot_model.py    # Advanced QoT model training
│   └── models/                   # Trained model files
├── models/                       # Additional model storage
├── requirements.txt              # Python dependencies
├── test_components.py           # Component testing script
└── README.md                     # This file
```

## Prerequisites

- Python 3.8 or higher
- Required Python packages (see installation section)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ML_QOT
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Lightpath Dataset**:
   The lightpath dataset file `Lightpath_756_label_4_QoT_dataset_train_900.txt` is not included in this repository due to its large size (225MB). Please download it from the following Google Drive link:
   
   **Google Drive Link**: [Lightpath Dataset](https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing)
   
   After downloading, place the file in the `data/` directory.

## Usage

### 1. Advanced QoT Model Training

Train a comprehensive QoT model using both synthetic and real-world data:

```bash
cd src
python train_new_qot_model.py
```

### 2. Generate Synthetic Data

Generate synthetic training data for custom network topologies:

```bash
cd src
python generate_synthetic_data.py --nodes 20 --degree 3.0 --samples 5000 --output data/synthetic
```

### 3. Process Lightpath Data

Process the lightpath dataset for training:

```bash
cd src
python process_lightpath_data.py --file ../data/Lightpath_756_label_4_QoT_dataset_train_900.txt --max-samples 10000
```

### 4. Use ML QoT Estimator with Smart Ensemble

```python
from src.ml_qot import MLQoTEstimator
from src.eon_models import ModulationFormat

# Initialize estimator
estimator = MLQoTEstimator(model_dir="src/models")

# Load trained model
estimator.load_model("path/to/model.joblib")

# Estimate QoT for a path with smart ensemble prediction
path = ["node1", "node2", "node3"]
links = {...}  # Dictionary of EONLink objects
estimated_qot, confidence = estimator.estimate_qot(
    path=path,
    links=links,
    launch_power=3.0,
    channel_spacing=12.5,
    num_channels=1,
    modulation=ModulationFormat.QPSK
)

# The estimator automatically uses smart ensemble strategy:
# - Fast LightGBM prediction for most cases
# - RandomForest fallback for edge cases
# - Ensemble prediction when close to QoT thresholds
```

### 5. Regeneration Point Optimization

```python
from src.regenerator import Regenerator
from src.eon_models import ModulationFormat
import networkx as nx

# Load network topology
G = nx.read_gml("data/germany50.gml")

# Initialize regenerator
regenerator = Regenerator(G, max_segment_length=200.0)

# Find regeneration points for a path
path = ["node1", "node2", "node3", "node4", "node5"]
segments = regenerator.find_regeneration_points(path, num_slots=1, modulation=ModulationFormat.QPSK)

# Calculate regeneration metrics
cost = regenerator.calculate_regeneration_cost(segments)
latency = regenerator.calculate_regeneration_latency(segments)
power = regenerator.calculate_regeneration_power(segments)
```

### 6. Path Finding and Topology Analysis

```python
from src.topology_loader import load_topology, get_k_shortest_paths

# Load topology
G = load_topology("data/germany50.gml")

# Find k-shortest paths
paths = get_k_shortest_paths(G, "source_node", "target_node", k=3)

# Calculate path metrics
for path in paths:
    length = calculate_path_length(G, path)
    print(f"Path: {path}, Length: {length} km")
```

### 7. Test Components

Run the component test suite to verify everything is working:

```bash
python test_components.py
```

## Components

### MLQoTEstimator
The main class for QoT estimation using advanced machine learning models. Supports:
- **Random Forest and LightGBM models** with hyperparameter optimization
- **Smart Ensemble Prediction**: Intelligent model selection based on prediction confidence
- **Fast Inference**: LightGBM for speed, RandomForest for accuracy when needed
- **Threshold-aware Decision Making**: Uses ensemble when predictions are close to QoT thresholds
- **Model persistence and loading** with backward compatibility
- **Industry-standard span-based OSNR calculations**

### Smart Ensemble Strategy
The estimator implements an intelligent ensemble approach:
- **Fast Path**: Always uses LightGBM first (faster inference)
- **Threshold Checking**: Only uses RandomForest if prediction is close to QoT threshold
- **Fallback Logic**: Handles negative predictions and edge cases robustly
- **Model Selection**: Returns which model was used and all predictions for transparency

### EONNode
Represents a node in the EON with ROADM and regeneration capabilities:
- WSS (Wavelength Selective Switch) specifications
- Add/drop port management
- Link connectivity
- Regeneration capabilities and resource management

### EONLink
Represents a link in the EON with:
- Fiber characteristics (attenuation, dispersion, PMD)
- Amplifier parameters
- Transceiver specifications
- QoT calculation methods using industry-standard models

### EONController
Manages the overall EON network:
- Path finding and resource allocation
- QoT feasibility checking
- Network initialization
- Regeneration point coordination

### SpectrumManager
Handles spectrum allocation in the flexible grid:
- Slot allocation and release
- Spectrum availability checking
- Block management
- Fragmentation analysis

### Regenerator
Advanced regeneration point optimization:
- QoT-aware regeneration point selection
- Cost, latency, and power consumption analysis
- Multi-segment path planning
- Regeneration resource management

### TopologyLoader
Network topology and path analysis:
- GML file loading and processing
- K-shortest path algorithms
- Distance calculations using Haversine formula
- Path validation and metrics

## Data Formats

### Lightpath Dataset
The lightpath dataset contains the following columns:
- `path_length`: Length of the lightpath in km
- `laser_current`: Laser current in mA
- `launch_power`: Launch power in dBm
- `osnr`: Optical Signal-to-Noise Ratio in dB
- `ber`: Bit Error Rate
- `failure_type`: Type of failure (0: No failure, 1: ECL failure, 2: EDFA failure, 3: NLI failure)

### Network Topology
Network topologies are stored in GML format and include:
- Node properties (type, capabilities, regeneration resources)
- Edge properties (length, fiber type, temperature, age)
- Geographic coordinates for distance calculations

## Model Performance

The trained models achieve exceptional performance on test datasets:

- **R² Score:** 0.97+  
  (Indicates the model explains nearly all variance in the data; extremely high predictive accuracy.)
- **RMSE:** 0.58 dB  
  (Shows the average prediction error is less than 1 dB, meaning predictions are very close to actual values.)
- **Training Time:** 8.43 seconds  
  (Demonstrates efficient training, even on large datasets.)
- **Inference Speed:** ~10x faster with LightGBM compared to traditional Gradient Boosting
- **Smart Ensemble:** Intelligent model selection reduces computational overhead by 60-80%

These results reflect the model's robustness and suitability for real-time Quality of Transmission estimation in elastic optical networks.

## Key Achievements

- **High Accuracy**: Achieved 97%+ R² score, indicating near-perfect prediction capability
- **Low Error**: RMSE of only 0.58 dB shows predictions are highly precise
- **Fast Training**: Complete model training in under 9 seconds
- **Robust**: Works with both synthetic and real-world data
- **Regeneration Support**: Advanced regeneration point optimization for long-haul networks
- **Industry Standards**: Implements span-based OSNR calculations and physical models
- **Smart Inference**: LightGBM-based fast prediction with intelligent ensemble fallback

## Advanced Features

### Smart Ensemble Prediction
- **Fast LightGBM Inference**: Primary model for speed and efficiency
- **Threshold-aware Decision Making**: Uses ensemble when predictions are close to QoT thresholds
- **Robust Fallback**: Handles edge cases and negative predictions gracefully
- **Transparent Model Selection**: Returns which model was used for each prediction

### Regeneration Planning
- **QoT-aware Optimization**: Finds optimal regeneration points based on QoT requirements
- **Cost Analysis**: Calculates regeneration costs, latency, and power consumption
- **Resource Management**: Tracks and allocates regeneration resources
- **Multi-segment Paths**: Supports complex multi-segment lightpath planning

### Path Optimization
- **K-shortest Paths**: Implements Yen's algorithm for finding multiple path alternatives
- **QoT-aware Routing**: Considers QoT constraints in path selection
- **Distance Calculations**: Uses Haversine formula for accurate geographic distances

### Physical Modeling
- **Span-based OSNR**: Industry-standard OSNR calculations
- **Nonlinear Effects**: Models SPM, XPM, and FWM impairments
- **PMD Analysis**: Polarization Mode Dispersion calculations
- **Fiber Parameters**: Support for multiple fiber types (SMF-28, LEAF)

## Model Architecture

### LightGBM Integration
- **Fast Training**: LightGBM's efficient implementation reduces training time
- **Memory Efficient**: Better memory usage for large datasets
- **Early Stopping**: Built-in validation during training prevents overfitting
- **Advanced Hyperparameters**: Sophisticated parameter grid for optimal performance

### Ensemble Strategy
- **Primary Model**: LightGBM for fast inference (90% of cases)
- **Fallback Model**: RandomForest for edge cases and threshold proximity
- **Ensemble Decision**: Weighted combination when both models are needed
- **Confidence Metrics**: Model selection transparency and prediction confidence

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ml_qot_2024,
  title={Machine Learning-based Quality of Transmission Estimation for Elastic Optical Networks},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository or contact the maintainers.

## Acknowledgments

- Germany50 topology data
- Lightpath dataset contributors
- Open-source community for the libraries used in this project 
