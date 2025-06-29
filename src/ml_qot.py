#ml_qot.py
import os
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src directory
from eon_models import EONLink, ModulationFormat
from spectrum_manager import SpectrumManager

logger = logging.getLogger(__name__)

@dataclass
class QoTFeatures:
    """Features for QoT estimation"""
    path_length: float  # km
    num_hops: int
    num_channels: int
    channel_spacing: float  # GHz
    launch_power: float  # dBm
    modulation: ModulationFormat
    fiber_type: str
    num_amplifiers: int
    total_loss: float  # dB
    total_dispersion: float  # ps/nm
    total_pmd: float  # ps
    temperature: float  # °C
    fiber_age: float  # years
    num_filters: int
    filter_bandwidth: float  # GHz
    wss_loss: float  # dB
    node_loss: float  # dB

@dataclass
class ModelMetrics:
    """Metrics for model performance"""
    r2_score: float
    mse: float
    rmse: float
    cv_scores: List[float]
    best_params: Dict
    training_time: float

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle module name changes."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # Remove 'src.' prefix if present
            if module.startswith('src.'):
                module = module[4:]
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                # Try without any prefix
                if '.' in module:
                    module = module.split('.')[-1]
                return super().find_class(module, name)

class MLQoTEstimator:
    """Machine Learning based QoT estimator"""
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Define parameter grids for tuning - balanced search space
        self.rf_param_grid = {
            'n_estimators': [100, 200, 300],  # 3 options
            'max_depth': [10, 15, 20],        # 3 options
            'min_samples_split': [2, 5, 10],  # 3 options
            'min_samples_leaf': [1, 2, 4],    # 3 options
            'max_features': ['sqrt', 'log2']  # 2 options
        }  # Total: 3 * 3 * 3 * 3 * 2 = 162 combinations
        
        self.gb_param_grid = {
            'n_estimators': [100, 200, 300],  # 3 options
            'max_depth': [5, 10, 15],         # 3 options
            'learning_rate': [0.01, 0.1, 0.2], # 3 options
            'min_samples_split': [2, 5, 10],  # 3 options
            'subsample': [0.8, 0.9, 1.0]      # 3 options
        }  # Total: 3 * 3 * 3 * 3 * 3 = 243 combinations
        
        # Initialize models with default parameters
        self.rf_model = RandomForestRegressor(random_state=42)
        self.gb_model = GradientBoostingRegressor(random_state=42)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time = None
        self.training_history = []
        self.model_metrics = None
        
        # EDFA parameters
        self.edfa_gain = 25.0  # dB per amplifier
        self.ase_noise = 5.0   # dB per amplifier
        
    def _calculate_osnr(self, features: QoTFeatures, launch_power: float, channel_spacing: float) -> float:
        """Calculate OSNR based on physical parameters"""
        # Calculate signal power
        signal_power = launch_power - features.total_loss
        
        # Add EDFA gain
        edfa_gain = features.num_amplifiers * self.edfa_gain
        signal_power += edfa_gain
        
        # Calculate noise power (thermal + ASE)
        thermal_noise = -174 + 10 * np.log10(channel_spacing * 1e9)  # dBm
        ase_noise = features.num_amplifiers * self.ase_noise
        noise_power = thermal_noise + ase_noise
        
        # Calculate OSNR
        osnr = signal_power - noise_power
        
        # Add modulation format penalty
        if features.modulation == ModulationFormat.QAM64:
            osnr -= 3.0  # Higher penalty for QAM64
        elif features.modulation == ModulationFormat.QAM16:
            osnr -= 1.5  # Medium penalty for QAM16
        elif features.modulation == ModulationFormat.QAM8:
            osnr -= 0.5  # Lower penalty for QAM8
        
        return osnr
        
    def _extract_features(self, path: List[str], links: Dict[Tuple[str, str], EONLink],
                         launch_power: float, channel_spacing: float,
                         num_channels: int, modulation: ModulationFormat) -> QoTFeatures:
        """Extract features from path for QoT estimation"""
        total_length = 0
        total_loss = 0
        total_dispersion = 0
        total_pmd = 0
        num_amplifiers = 0
        num_filters = 0
        total_wss_loss = 0
        total_node_loss = 0
        
        # Get fiber type from first link
        fiber_type = links[(path[0], path[1])].fiber_type
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = links[(u, v)]
            
            # Accumulate path metrics
            total_length += link.length
            total_loss += link.calculate_path_loss()
            total_dispersion += (link.length * 
                               link.fiber_params[link.fiber_type]["dispersion"])
            total_pmd += (link.length * 
                         link.fiber_params[link.fiber_type]["pmd_coefficient"])
            
            # Calculate number of amplifiers (one every 80 km)
            num_amplifiers += int(link.length / 80)
            
            # Add node and filter effects
            num_filters += 2  # Each node has filters
            total_wss_loss += 5.0  # Typical WSS loss
            total_node_loss += 2.0  # Typical node loss
            
        # Simulate temperature and aging effects
        temperature = 25.0 + np.random.normal(0, 5)  # Room temperature with variation
        fiber_age = np.random.uniform(0, 10)  # Fiber age in years
        
        # Adjust launch power based on modulation format
        adjusted_launch_power = launch_power
        if modulation == ModulationFormat.QAM64:
            adjusted_launch_power += 3.0  # Higher power for QAM64
        elif modulation == ModulationFormat.QAM16:
            adjusted_launch_power += 1.5  # Medium power for QAM16
        elif modulation == ModulationFormat.QAM8:
            adjusted_launch_power += 0.5  # Slightly higher for QAM8
        
        return QoTFeatures(
            path_length=total_length,
            num_hops=len(path) - 1,
            num_channels=num_channels,
            channel_spacing=channel_spacing,
            launch_power=adjusted_launch_power,
            modulation=modulation,
            fiber_type=fiber_type,
            num_amplifiers=num_amplifiers,
            total_loss=total_loss,
            total_dispersion=total_dispersion,
            total_pmd=total_pmd,
            temperature=temperature,
            fiber_age=fiber_age,
            num_filters=num_filters,
            filter_bandwidth=37.5,  # Typical filter bandwidth
            wss_loss=total_wss_loss,
            node_loss=total_node_loss
        )
        
    def _features_to_array(self, features: QoTFeatures) -> np.ndarray:
        """Convert features to numpy array for model input"""
        modulation_value = {
            ModulationFormat.QPSK: 0,
            ModulationFormat.QAM8: 1,
            ModulationFormat.QAM16: 2,
            ModulationFormat.QAM64: 3
        }[features.modulation]
        
        fiber_value = 0 if features.fiber_type == "SMF-28" else 1
        
        return np.array([[
            features.path_length,
            features.num_hops,
            features.num_channels,
            features.channel_spacing,
            features.launch_power,
            modulation_value,
            fiber_value,
            features.num_amplifiers,
            features.total_loss,
            features.total_dispersion,
            features.total_pmd,
            features.temperature,
            features.fiber_age,
            features.num_filters,
            features.filter_bandwidth,
            features.wss_loss,
            features.node_loss
        ]])
        
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestRegressor, GradientBoostingRegressor, Dict]:
        """
        Tune hyperparameters for both models using GridSearchCV with early stopping
        """
        # Tune Random Forest with early stopping
        rf_grid = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=self.rf_param_grid,
            cv=4,  # Using 4 folds
            scoring='r2',  # Use single metric to avoid refit issues
            refit=True,
            n_jobs=-1,
            verbose=1,
            error_score=0.0
        )
        rf_grid.fit(X, y)
        
        # Tune Gradient Boosting with early stopping
        gb_grid = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid=self.gb_param_grid,
            cv=4,  # Using 4 folds
            scoring='r2',  # Use single metric to avoid refit issues
            refit=True,
            n_jobs=-1,
            verbose=1,
            error_score=0.0
        )
        gb_grid.fit(X, y)
        
        # Get best parameters
        best_params = {
            'rf': rf_grid.best_params_,
            'gb': gb_grid.best_params_
        }
        
        return rf_grid.best_estimator_, gb_grid.best_estimator_, best_params
        
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate model performance using cross-validation"""
        start_time = datetime.now()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=5, 
            scoring='r2',
            n_jobs=-1
        )
        
        # Calculate metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        return ModelMetrics(
            r2_score=r2,
            mse=mse,
            rmse=rmse,
            cv_scores=cv_scores.tolist(),
            best_params=model.get_params(),
            training_time=training_time
        )
        
    def train(self, training_data: List[Tuple[QoTFeatures, float]], 
              retrain: bool = False) -> None:
        """Train the QoT estimator with hyperparameter tuning"""
        if self.is_trained and not retrain:
            return
            
        X = np.array([self._features_to_array(f)[0] for f, _ in training_data])
        y = np.array([qot for _, qot in training_data])
        
        # Scale features - ensure scaler is initialized
        if self.scaler is None:
            self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Tune hyperparameters with early stopping
        print("Tuning hyperparameters...")
        self.rf_model, self.gb_model, best_params = self._tune_hyperparameters(
            np.asarray(X_train), np.asarray(y_train)
        )
        
        # Train both models with best parameters
        print("Training models with best parameters...")
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        
        # Evaluate models
        rf_metrics = self._evaluate_model(self.rf_model, np.asarray(X_val), np.asarray(y_val))
        gb_metrics = self._evaluate_model(self.gb_model, np.asarray(X_val), np.asarray(y_val))
        
        # Select best model based on R² score
        if rf_metrics.r2_score > gb_metrics.r2_score:
            self.model = self.rf_model
            self.model_metrics = rf_metrics
            selected_model = 'rf'
        else:
            self.model = self.gb_model
            self.model_metrics = gb_metrics
            selected_model = 'gb'
            
        # Record training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'rf_metrics': rf_metrics.__dict__,
            'gb_metrics': gb_metrics.__dict__,
            'selected_model': selected_model,
            'best_params': best_params,
            'num_samples': len(training_data)
        })
        
        print(f"\nTraining Results:")
        print(f"Selected Model: {selected_model}")
        print(f"R² Score: {self.model_metrics.r2_score:.3f}")
        print(f"RMSE: {self.model_metrics.rmse:.3f}")
        print(f"CV Scores: {np.mean(self.model_metrics.cv_scores):.3f} ± {np.std(self.model_metrics.cv_scores):.3f}")
        print(f"Training Time: {self.model_metrics.training_time:.2f} seconds")
        
        self.is_trained = True
        self.last_training_time = datetime.now()
        
    def estimate_qot(self,
                    path: List[str],
                    links: Dict[Tuple[str, str], EONLink],
                    launch_power: float,
                    channel_spacing: float,
                    num_channels: int,
                    modulation: ModulationFormat) -> Tuple[float, float]:
        """Estimate QoT (OSNR) for a given path."""
        try:
            features = self._extract_features(path, links, launch_power, channel_spacing, num_channels, modulation)
            logger.info(f"QoT Features for path {' -> '.join(path)}:")
            logger.info(f"  Path Length: {features.path_length:.2f} km")
            logger.info(f"  Number of Hops: {features.num_hops}")
            logger.info(f"  Launch Power: {features.launch_power:.2f} dBm")
            logger.info(f"  Modulation: {features.modulation}")
            logger.info(f"  Total Loss: {features.total_loss:.2f} dB")
            logger.info(f"  Total Dispersion: {features.total_dispersion:.2f} ps/nm")
            logger.info(f"  Total PMD: {features.total_pmd:.2f} ps")
            logger.info(f"  Number of Amplifiers: {features.num_amplifiers}")
            
            # Use ML models to predict OSNR directly
            X = self._features_to_array(features)
            if self.scaler is None:
                logger.error("Scaler not initialized. Please train the model first.")
                return 0.0, 0.0
            X_scaled = self.scaler.transform(X)
            rf_pred = self.rf_model.predict(X_scaled)[0] if self.rf_model is not None else None
            gb_pred = self.gb_model.predict(X_scaled)[0] if self.gb_model is not None else None
            
            if rf_pred is not None and gb_pred is not None:
                # Ensemble prediction
                osnr = 0.7 * rf_pred + 0.3 * gb_pred
                confidence = 1.0 - abs(rf_pred - gb_pred) / max(abs(rf_pred), abs(gb_pred))
            elif rf_pred is not None:
                osnr = rf_pred
                confidence = 1.0
            elif gb_pred is not None:
                osnr = gb_pred
                confidence = 1.0
            else:
                logger.error("No trained model available for prediction.")
                return 0.0, 0.0
            
            logger.info(f"QoT Estimation Results:")
            logger.info(f"  Random Forest Prediction: {rf_pred if rf_pred is not None else 'N/A'}")
            logger.info(f"  Gradient Boosting Prediction: {gb_pred if gb_pred is not None else 'N/A'}")
            logger.info(f"  Final OSNR: {osnr:.2f} dB")
            logger.info(f"  Confidence: {confidence:.2f}")
            
            return osnr, confidence
            
        except Exception as e:
            logger.error(f"Error estimating QoT: {str(e)}")
            return 0.0, 0.0
        
    def save_model(self, name: str = "qot_model") -> None:
        """Save trained model, metrics, and scaler to file."""
        try:
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.model_dir, f"{name}_{timestamp}.joblib")
            
            # Save all components in a dictionary
            model_data = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'metrics': self.model_metrics
            }
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
        
    def load_model(self, model_path: str) -> None:
        """Load a trained model from file."""
        try:
            logger.info(f"Loading model from {model_path}")
            loaded_data = joblib.load(model_path)
            if isinstance(loaded_data, dict):
                self.rf_model = loaded_data.get('rf_model', None)
                self.gb_model = loaded_data.get('gb_model', None)
                self.scaler = loaded_data.get('scaler', None)
                self.model_metrics = loaded_data.get('metrics', None)
                # Fallback for old format
                if self.rf_model is None and 'model' in loaded_data:
                    self.rf_model = loaded_data['model']
                if self.scaler is None and 'scaler' in loaded_data:
                    self.scaler = loaded_data['scaler']
                if self.model_metrics is None and 'metrics' in loaded_data:
                    self.model_metrics = loaded_data['metrics']
            else:
                raise ValueError("Invalid model format")
            self.is_trained = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
    def generate_training_data(self,
                             paths: List[List[str]],
                             links: Dict[Tuple[str, str], EONLink],
                             num_samples: int = 1000) -> List[Tuple[QoTFeatures, float]]:
        """Generate training data for the model"""
        training_data = []
        
        # Convert paths to list of lists for numpy compatibility
        paths = [list(path) for path in paths]
        
        # Create list of modulation formats for random choice
        modulation_formats = [ModulationFormat.QPSK, ModulationFormat.QAM8, 
                             ModulationFormat.QAM16, ModulationFormat.QAM64]
        
        for _ in range(num_samples):
            # Randomly select a path
            path = paths[np.random.randint(0, len(paths))]
            
            # Generate random parameters with realistic ranges
            launch_power = np.random.uniform(0, 5)  # dBm
            channel_spacing = np.random.choice([12.5, 25, 37.5, 50])  # GHz
            num_channels = np.random.randint(1, 10)
            modulation = random.choice(modulation_formats)  # Fixed: use random.choice instead of np.random.choice
            
            features = self._extract_features(
                path, links, launch_power, channel_spacing,
                num_channels, modulation
            )
            
            # Calculate OSNR using physical model
            osnr = self._calculate_osnr(features, features.launch_power, features.channel_spacing)
            
            # Add noise to simulate real-world variations
            osnr += np.random.normal(0, 0.5)  # 0.5 dB standard deviation
            
            training_data.append((features, osnr))
            
        return training_data
        
    def should_retrain(self, min_samples: int = 100) -> bool:
        """Check if model should be retrained"""
        if not self.is_trained or not self.last_training_time:
            return True
            
        # Retrain if last training was more than a day ago
        time_since_training = datetime.now() - self.last_training_time
        if time_since_training.days >= 1:
            return True
            
        # Retrain if we have enough new samples
        if len(self.training_history) >= min_samples:
            return True
            
        return False 
