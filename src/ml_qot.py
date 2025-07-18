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
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import logging
from scipy.sparse import issparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from src directory
from src.eon_models import EONLink, ModulationFormat
from src.spectrum_manager import SpectrumManager

if TYPE_CHECKING:
    from src.regenerator import Regenerator

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
        
        self.lgbm_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.1, 1.0],
            'reg_lambda': [0.1, 1.0]
        }
        
        # Initialize models with default parameters
        self.rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.5,  # Use 50% of features at each split
            random_state=42
        )
        self.lgbm_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42
        ) if lgb is not None else None
        self.ensemble_margin = 1.0  # dB
        self.qot_threshold = 15.0   # Example OSNR threshold for "good" QoT
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_time = None
        self.training_history = []
        self.model_metrics = None
        
        # EDFA parameters - Updated to match physics-based calculation
        self.edfa_gain = 20.0  # dB per amplifier (reduced from 25.0)
        self.ase_noise = -36.0  # dBm per amplifier (physics-based, was 5.0 dB)
        
    def _calculate_osnr_span_model(self, features: QoTFeatures, launch_power: float, channel_spacing: float) -> float:
        """
        Calculate OSNR using industry-standard span model.
        This is the most accurate method for multi-span optical links.
        """
        # Physical constants
        h = 6.626e-34
        c = 3e8
        wavelength_nm = 1550
        freq = c / (wavelength_nm * 1e-9)
        bandwidth_hz = channel_spacing * 1e9

        # Span and amplifier setup
        span_length_km = 80  # Standard span length
        num_spans = int(np.ceil(features.path_length / span_length_km))
        fiber_loss_db = span_length_km * 0.2  # dB/km
        amp_gain_db = fiber_loss_db  # Each amp compensates for span loss
        amp_gain_linear = 10**(amp_gain_db / 10)
        noise_figure_db = 5.0
        noise_figure_linear = 10**(noise_figure_db / 10)
        n_sp = noise_figure_linear / 2

        # ASE noise per amp (in Watts)
        ase_per_amp_w = 2 * n_sp * h * freq * bandwidth_hz * (amp_gain_linear - 1)

        # Signal power at receiver (in mW)
        signal_power_dbm = launch_power  # Each span resets signal to launch power
        signal_power_mw = 10**(signal_power_dbm / 10)

        # Total ASE noise (in mW) after N spans
        total_ase_w = num_spans * ase_per_amp_w
        total_ase_mw = total_ase_w * 1e3

        # OSNR (linear)
        osnr_linear = signal_power_mw / total_ase_mw
        osnr_db = 10 * np.log10(osnr_linear)

        # Add modulation format penalty
        if features.modulation == ModulationFormat.QAM64:
            osnr_db -= 3.0
        elif features.modulation == ModulationFormat.QAM16:
            osnr_db -= 1.5
        elif features.modulation == ModulationFormat.QAM8:
            osnr_db -= 0.5

        return osnr_db
        
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
            
            # Add node and filter effects
            num_filters += 2  # Each node has filters
            total_wss_loss += 5.0  # Typical WSS loss
            total_node_loss += 2.0  # Typical node loss
        
        # Calculate number of amplifiers based on total path length (one every 80 km span)
        # No amplifiers for paths < 80km, then one per 80km span
        # For paths >= 80km: number of amplifiers = ceil(path_length / 80) - 1
        # This accounts for the fact that we don't need an amp at the start
        if total_length < 80:
            num_amplifiers = 0
        else:
            num_amplifiers = int(np.ceil(total_length / 80)) - 1
        
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
        
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """
        Tune hyperparameters for both models using RandomizedSearchCV with early stopping
        """
        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error'
        }
        
        # Tune Random Forest with early stopping
        rf_search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_distributions=self.rf_param_grid,
            n_iter=60,
            cv=5,
            scoring='r2',
            refit=True,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            error_score=0.0
        )
        rf_search.fit(X, y)
        
        # Tune LightGBM with early stopping
        lgbm_search = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(random_state=42),
            param_distributions=self.lgbm_param_grid,
            n_iter=60,
            cv=5,
            scoring='r2',
            refit=True,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            error_score=0.0
        )
        lgbm_search.fit(X, y)
        
        # Get best parameters
        best_params = {
            'rf': rf_search.best_params_,
            'lgbm': lgbm_search.best_params_
        }
        
        return rf_search.best_estimator_, lgbm_search.best_estimator_, best_params
        
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
        self.rf_model, self.lgbm_model, best_params = self._tune_hyperparameters(
            np.asarray(X_train), np.asarray(y_train)
        )
        
        # Train both models with best parameters
        print("Training models with best parameters...")
        # Ensure dense arrays for LightGBM
        if issparse(X_train) and not isinstance(X_train, (list, tuple, str)):
            X_train = X_train.toarray()
        if issparse(X_val) and not isinstance(X_val, (list, tuple, str)):
            X_val = X_val.toarray()
        self.rf_model.fit(X_train, y_train)
        best_lgbm = lgb.LGBMRegressor(**best_params['lgbm'])
        best_lgbm.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse'
        )
        
        # Evaluate models
        rf_metrics = self._evaluate_model(self.rf_model, np.asarray(X_val), np.asarray(y_val))
        lgbm_metrics = self._evaluate_model(best_lgbm, np.asarray(X_val), np.asarray(y_val))
        
        # Select best model based on R² score
        if rf_metrics.r2_score > lgbm_metrics.r2_score:
            self.model = self.rf_model
            self.model_metrics = rf_metrics
            selected_model = 'rf'
        else:
            self.model = best_lgbm
            self.model_metrics = lgbm_metrics
            selected_model = 'lgbm'
            
        # Record training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'rf_metrics': rf_metrics.__dict__,
            'lgbm_metrics': lgbm_metrics.__dict__,
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
        
    def smart_ensemble_predict(self, X: np.ndarray) -> Tuple[float, str, Dict]:
        """Smart ensemble: use fast LightGBM, only call RF if prediction is close to threshold.
        Returns (prediction, model_used, predictions_dict)"""
        if self.lgbm_model is None:
            raise RuntimeError("LightGBM model is not available.")
        
        # Ensure dense arrays for LightGBM
        if issparse(X):
            X = X.toarray()
        
        # Always get LightGBM prediction first
        fast_pred = float(self.lgbm_model.predict(X)[0])
        predictions = {"LightGBM": fast_pred}
        
        # Check if prediction is reasonable (positive OSNR)
        if fast_pred < 0:
            # If LightGBM predicts negative, try RandomForest
            if self.rf_model is not None:
                slow_pred = float(self.rf_model.predict(X)[0])
                predictions["RandomForest"] = slow_pred
                # Use the better prediction (less negative or positive)
                if slow_pred > fast_pred:
                    return slow_pred, "RandomForest", predictions
                else:
                    # Both are negative, return the average but cap at 0
                    ensemble_pred = max((fast_pred + slow_pred) / 2, 0.0)
                    return ensemble_pred, "Ensemble", predictions
            else:
                # No RF model, cap at 0
                return max(fast_pred, 0.0), "LightGBM_capped", predictions
        
        # For reasonable predictions, check if we need ensemble
        # Only use ensemble if prediction is close to threshold (within margin)
        if abs(fast_pred - self.qot_threshold) <= self.ensemble_margin:
            # Prediction is close to threshold, use ensemble for better accuracy
            if self.rf_model is not None:
                slow_pred = float(self.rf_model.predict(X)[0])
                predictions["RandomForest"] = slow_pred
                ensemble_pred = (fast_pred + slow_pred) / 2
                return ensemble_pred, "Ensemble", predictions
            else:
                return fast_pred, "LightGBM", predictions
        else:
            # Prediction is far from threshold, LightGBM is sufficient
            return fast_pred, "LightGBM", predictions

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
            
            if self.lgbm_model is not None and self.rf_model is not None:
                # Ensure X_scaled is a numpy ndarray
                if issparse(X_scaled):
                    X_scaled = X_scaled.toarray()
                osnr, model_used, predictions = self.smart_ensemble_predict(X_scaled)
                confidence = 1.0  # Optionally, you can define a confidence metric
            else:
                logger.error("No trained model available for prediction.")
                return 0.0, 0.0
            
            logger.info(f"QoT Estimation Results:")
            # Log individual model predictions
            for model_name, pred in predictions.items():
                logger.info(f"  {model_name} Prediction: {pred}")
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
                'lgbm_model': self.lgbm_model,
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
                self.lgbm_model = loaded_data.get('lgbm_model', None)
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
            osnr = self._calculate_osnr_span_model(features, features.launch_power, features.channel_spacing)
            
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
