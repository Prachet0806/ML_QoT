# ML-QoT: Machine Learning-based Quality of Transmission Estimation for Elastic Optical Networks

## Overview

This project implements a machine learning-based Quality of Transmission (QoT) estimation system for Elastic Optical Networks (EONs). The system uses both synthetic and real-world data to train models that can predict QoT parameters for optical lightpaths, enabling better resource allocation and network planning.

## Features

- **ML-based QoT Estimation**: Uses Random Forest and Gradient Boosting models for QoT prediction
- **Flexible Network Topologies**: Support for custom network topologies and Germany50 topology
- **Real-world Data Integration**: Processes lightpath measurement data for training
- **Synthetic Data Generation**: Generates synthetic training data for various network scenarios
- **Spectrum Management**: Flexible grid spectrum allocation and management
- **Multiple Modulation Formats**: Support for QPSK, 8-QAM, 16-QAM, and 64-QAM
- **Comprehensive QoT Metrics**: OSNR, BER, Q-factor, nonlinear penalties, dispersion, and PMD

## Project Structure

```
ML_QOT/
├── data/                          # Data files
│   ├── germany50.gml             # Germany50 network topology
│   ├── germany50_fixed.gml       # Fixed Germany50 topology
│   └── Lightpath_756_label_4_QoT_dataset_train_900.txt  # Lightpath dataset (download separately)
├── src/                          # Source code
│   ├── ml_qot.py                 # Main ML QoT estimator
│   ├── eon_models.py             # EON network models and components
│   ├── eon_control.py            # EON controller for network management
│   ├── spectrum_manager.py       # Spectrum allocation and management
│   ├── generate_synthetic_data.py # Synthetic data generation
│   ├── process_lightpath_data.py # Lightpath data processing
│   ├── lightpath_reader.py       # Lightpath data reader
│   ├── train_qot_model.py        # Basic QoT model training
│   ├── train_new_qot_model.py    # Advanced QoT model training
│   └── models/                   # Trained model files
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
   pip install numpy pandas scikit-learn networkx matplotlib seaborn joblib
   ```

3. **Download the Lightpath Dataset**:
   The lightpath dataset file `Lightpath_756_label_4_QoT_dataset_train_900.txt` is not included in this repository due to its large size (225MB). Please download it from the following Google Drive link or the original Mendeley Link:
   
   **Google Drive Link**: [Lightpath Dataset](https://drive.google.com/file/d/1_cxhKVVeqCLElU-m4XksmpVQ_RvC9KbF/view?usp=sharing)
   **Mendeley Website Link**: [Lightpath Dataset](https://data.mendeley.com/datasets/y3pspy7j83/1)
   
   
   After downloading, place the file in the `data/` directory.

## Usage

### 1. Basic QoT Model Training

Train a basic QoT model using synthetic data:

```bash
cd src
python train_qot_model.py
```

### 2. Advanced QoT Model Training

Train a comprehensive QoT model using both synthetic and real-world data:

```bash
cd src
python train_new_qot_model.py
```

### 3. Generate Synthetic Data

Generate synthetic training data for custom network topologies:

```bash
cd src
python generate_synthetic_data.py --nodes 20 --degree 3.0 --samples 5000 --output data/synthetic
```

### 4. Process Lightpath Data

Process the lightpath dataset for training:

```bash
cd src
python process_lightpath_data.py --file ../data/Lightpath_756_label_4_QoT_dataset_train_900.txt --max-samples 10000
```

### 5. Use ML QoT Estimator

```python
from src.ml_qot import MLQoTEstimator
from src.eon_models import ModulationFormat

# Initialize estimator
estimator = MLQoTEstimator(model_dir="src/models")

# Load trained model
estimator.load_model("path/to/model.joblib")

# Estimate QoT for a path
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
```

## Components

### MLQoTEstimator
The main class for QoT estimation using machine learning models. Supports:
- Random Forest and Gradient Boosting models
- Hyperparameter tuning
- Model persistence and loading
- Confidence estimation

### EONNode
Represents a node in the EON with ROADM capabilities:
- WSS (Wavelength Selective Switch) specifications
- Add/drop port management
- Link connectivity

### EONLink
Represents a link in the EON with:
- Fiber characteristics (attenuation, dispersion, PMD)
- Amplifier parameters
- Transceiver specifications
- QoT calculation methods

### EONController
Manages the overall EON network:
- Path finding and resource allocation
- QoT feasibility checking
- Network initialization

### SpectrumManager
Handles spectrum allocation in the flexible grid:
- Slot allocation and release
- Spectrum availability checking
- Block management

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
- Node properties (type, capabilities)
- Edge properties (length, fiber type, temperature, age)

## Model Performance

The trained models typically achieve:
- R² Score: 0.85-0.95
- RMSE: 1.5-3.0 dB
- Training time: 30-120 seconds (depending on dataset size)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Contact

For questions or issues, please open an issue on the GitHub repository

## Acknowledgments

- Germany50 topology data from https://www.topohub.org/?topology=sndlib/germany50
- Lightpath dataset from Mendeley (https://data.mendeley.com/datasets/y3pspy7j83/1)
- Open-source community for the libraries used in this project 
