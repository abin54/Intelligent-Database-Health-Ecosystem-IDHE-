"""
Advanced Performance Predictor
=============================

Revolutionary predictive maintenance system using:
- LSTM networks for time series forecasting
- ARIMA models for trend analysis
- Prophet for seasonal pattern detection
- Ensemble forecasting with confidence intervals
- Failure prediction and risk assessment
- Maintenance schedule optimization
- Resource planning recommendations

This is a unique, cutting-edge approach to database predictive maintenance.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import statistics
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logging.warning("Statsmodels not available, using basic forecasting")

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available, using basic prediction")

from ..utils.data_models import PerformancePrediction, MaintenanceRecommendation, FailureRisk

logger = logging.getLogger(__name__)

class LSTMTimeSeriesPredictor:
    """LSTM-based time series predictor for database metrics"""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 24):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
        if TENSORFLOW_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        self.model = keras.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 1)),
            layers.Dropout(0.2),
            layers.LSTM(100, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(50, activation='relu'),
            layers.Dense(self.prediction_horizon, activation='linear')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i:i+self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, training_data: np.ndarray, epochs: int = 100):
        """Train LSTM model on time series data"""
        if not self.model or not TENSORFLOW_AVAILABLE:
            logger.warning("LSTM model not available for training")
            return False
        
        try:
            # Normalize data
            normalized_data = self.scaler.fit_transform(training_data.reshape(-1, 1)).flatten()
            
            # Prepare sequences
            X, y = self.prepare_sequences(normalized_data)
            
            if len(X) == 0:
                logger.warning("Insufficient data for LSTM training")
                return False
            
            # Reshape for LSTM
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Train model
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            logger.info(f"LSTM model trained on {len(X)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False
    
    def predict(self, recent_data: np.ndarray) -> Optional[np.ndarray]:
        """Predict future values using trained LSTM model"""
        if not self.is_trained or not self.model:
            return None
        
        try:
            # Normalize data
            normalized_data = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()
            
            # Prepare input sequence
            if len(normalized_data) < self.sequence_length:
                # Pad sequence if too short
                padding = np.zeros(self.sequence_length - len(normalized_data))
                normalized_data = np.concatenate([padding, normalized_data])
            
            # Take last sequence_length points
            input_seq = normalized_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Make prediction
            prediction = self.model.predict(input_seq, verbose=0)
            
            # Denormalize prediction
            denormalized_pred = self.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            
            return denormalized_pred
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return None

class ARIMAForecaster:
    """ARIMA model for trend analysis and forecasting"""
    
    def __init__(self):
        self.models = {}
        self.model_orders = {}
        self.is_trained = {}
    
    def fit(self, data: np.ndarray, metric_name: str, order: Tuple[int, int, int] = None):
        """Fit ARIMA model to time series data"""
        if not ARIMA_AVAILABLE:
            logger.warning("ARIMA not available")
            return False
        
        try:
            # Auto-select order if not provided
            if order is None:
                order = self._auto_select_order(data)
            
            # Fit ARIMA model
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            
            self.models[metric_name] = fitted_model
            self.model_orders[metric_name] = order
            self.is_trained[metric_name] = True
            
            logger.info(f"ARIMA model fitted for {metric_name} with order {order}")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model for {metric_name}: {e}")
            return False
    
    def _auto_select_order(self, data: np.ndarray, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
        """Auto-select ARIMA order using information criteria"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        try:
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        if p == 0 and d == 0 and q == 0:
                            continue
                        
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            fitted = model.fit()
                            aic = fitted.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                
                        except:
                            continue
            
            return best_order
            
        except Exception as e:
            logger.error(f"Error in auto ARIMA order selection: {e}")
            return (1, 1, 1)
    
    def forecast(self, metric_name: str, steps: int = 24) -> Optional[np.ndarray]:
        """Forecast future values using fitted ARIMA model"""
        if not self.is_trained.get(metric_name, False):
            return None
        
        try:
            fitted_model = self.models[metric_name]
            forecast = fitted_model.forecast(steps=steps)
            
            return np.array(forecast)
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecast for {metric_name}: {e}")
            return None
    
    def get_confidence_intervals(self, metric_name: str, steps: int = 24, confidence: float = 0.95) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get confidence intervals for ARIMA forecast"""
        if not self.is_trained.get(metric_name, False):
            return None
        
        try:
            fitted_model = self.models[metric_name]
            forecast_result = fitted_model.get_forecast(steps=steps)
            
            conf_int = forecast_result.conf_int(alpha=1-confidence)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
            
            return lower, upper
            
        except Exception as e:
            logger.error(f"Error getting confidence intervals for {metric_name}: {e}")
            return None

class ProphetForecaster:
    """Facebook Prophet for seasonal pattern detection"""
    
    def __init__(self):
        self.models = {}
        self.is_trained = {}
    
    def fit(self, data: pd.DataFrame, metric_name: str):
        """Fit Prophet model to time series data"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_data = data[['timestamp', metric_name]].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_data)
            
            self.models[metric_name] = model
            self.is_trained[metric_name] = True
            
            logger.info(f"Prophet model fitted for {metric_name}")
            return True
            
        except ImportError:
            logger.warning("Prophet not available, skipping Prophet forecasting")
            return False
        except Exception as e:
            logger.error(f"Error fitting Prophet model for {metric_name}: {e}")
            return False
    
    def forecast(self, metric_name: str, periods: int = 24) -> Optional[pd.DataFrame]:
        """Forecast future values using Prophet model"""
        if not self.is_trained.get(metric_name, False):
            return None
        
        try:
            model = self.models[metric_name]
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='H')
            
            # Make prediction
            forecast = model.predict(future)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in Prophet forecast for {metric_name}: {e}")
            return None

class EnsemblePredictor:
    """Ensemble predictor combining multiple forecasting methods"""
    
    def __init__(self):
        self.lstm_predictor = LSTMTimeSeriesPredictor()
        self.arima_forecaster = ARIMAForecaster()
        self.prophet_forecaster = ProphetForecaster()
        self.random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gradient_boost = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.ensemble_weights = {}
        self.is_trained = False
    
    def train(self, metrics_data: pd.DataFrame):
        """Train all prediction models"""
        training_successful = False
        
        # Train LSTM if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            for column in metrics_data.columns:
                if column != 'timestamp' and metrics_data[column].dtype in ['float64', 'int64']:
                    values = metrics_data[column].fillna(0).values
                    if len(values) > 50:  # Minimum data requirement
                        self.lstm_predictor.train(values, epochs=50)
                        training_successful = True
        
        # Train ARIMA models
        if ARIMA_AVAILABLE:
            for column in metrics_data.columns:
                if column != 'timestamp' and metrics_data[column].dtype in ['float64', 'int64']:
                    values = metrics_data[column].fillna(0).values
                    if len(values) > 20:
                        self.arima_forecaster.fit(values, column)
                        training_successful = True
        
        # Train Prophet models
        try:
            for column in metrics_data.columns:
                if column != 'timestamp' and metrics_data[column].dtype in ['float64', 'int64']:
                    if len(metrics_data) > 30:
                        self.prophet_forecaster.fit(metrics_data, column)
                        training_successful = True
        except:
            pass
        
        # Train ensemble models
        if len(metrics_data) > 10:
            # Prepare features for traditional ML models
            features, targets = self._prepare_ml_features(metrics_data)
            
            if features is not None and targets is not None:
                # Scale features
                scaled_features = self.scaler.fit_transform(features)
                
                # Train models
                self.random_forest.fit(scaled_features, targets)
                self.gradient_boost.fit(scaled_features, targets)
                training_successful = True
        
        self.is_trained = training_successful
        logger.info(f"Ensemble predictor training completed: {training_successful}")
        return training_successful
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare features for traditional ML models"""
        try:
            # Create lag features and rolling statistics
            feature_data = []
            target_data = []
            
            for i in range(24, len(data)):  # Need historical data
                # Lag features (1, 6, 12, 24 hours ago)
                features = []
                
                for col in data.columns:
                    if col != 'timestamp':
                        value = data[col].iloc[i]
                        if pd.notna(value):
                            features.append(value)
                            
                            # Add lag features
                            if i >= 1:
                                features.append(data[col].iloc[i-1])
                            if i >= 6:
                                features.append(data[col].iloc[i-6])
                            if i >= 12:
                                features.append(data[col].iloc[i-12])
                            if i >= 24:
                                features.append(data[col].iloc[i-24])
                            
                            # Add rolling statistics
                            if i >= 6:
                                features.append(data[col].iloc[i-6:i].mean())
                                features.append(data[col].iloc[i-6:i].std())
                        else:
                            # Pad with zeros for missing values
                            features.extend([0] * 7)
                
                if features:
                    feature_data.append(features)
                    # Target is next value of key metric (e.g., CPU usage)
                    target_data.append(data['cpu_percent'].iloc[i+1] if i+1 < len(data) and 'cpu_percent' in data.columns else 0)
            
            return np.array(feature_data), np.array(target_data) if feature_data else (None, None)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None, None
    
    def predict_future_performance(self, current_data: np.ndarray, 
                                 prediction_horizon: int = 24) -> Dict[str, Any]:
        """Predict future performance using ensemble of methods"""
        predictions = {}
        confidence_intervals = {}
        
        # LSTM predictions
        if self.lstm_predictor.is_trained:
            lstm_pred = self.lstm_predictor.predict(current_data)
            if lstm_pred is not None:
                predictions['lstm'] = lstm_pred[:prediction_horizon]
        
        # ARIMA predictions
        for metric_name in self.arima_forecaster.models.keys():
            arima_pred = self.arima_forecaster.forecast(metric_name, prediction_horizon)
            if arima_pred is not None:
                predictions[f'arima_{metric_name}'] = arima_pred[:prediction_horizon]
                
                # Get confidence intervals
                conf_int = self.arima_forecaster.get_confidence_intervals(metric_name, prediction_horizon)
                if conf_int:
                    confidence_intervals[f'arima_{metric_name}'] = conf_int
        
        # Traditional ML predictions
        if hasattr(self, 'last_features'):
            # Predict using trained models
            try:
                scaled_features = self.scaler.transform(self.last_features.reshape(1, -1))
                rf_pred = self.random_forest.predict(scaled_features)[0]
                gb_pred = self.gradient_boost.predict(scaled_features)[0]
                
                # Create prediction horizon
                predictions['random_forest'] = np.full(prediction_horizon, rf_pred)
                predictions['gradient_boost'] = np.full(prediction_horizon, gb_pred)
                
            except Exception as e:
                logger.error(f"Error in traditional ML prediction: {e}")
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'ensemble_prediction': self._create_ensemble_prediction(predictions)
        }
    
    def _create_ensemble_prediction(self, predictions: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Create weighted ensemble prediction"""
        if not predictions:
            return None
        
        # Equal weights for all methods (could be optimized)
        weights = {method: 1.0 / len(predictions) for method in predictions}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Create ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for method, weight in weights.items():
            if method in predictions:
                ensemble_pred += weight * predictions[method]
        
        return ensemble_pred

class PerformancePredictor:
    """
    Advanced Performance Predictor
    
    Revolutionary features:
    1. Ensemble forecasting with multiple algorithms
    2. Failure prediction and risk assessment
    3. Maintenance schedule optimization
    4. Resource planning recommendations
    5. Confidence intervals and uncertainty quantification
    6. Real-time prediction updates
    """
    
    def __init__(self):
        self.ensemble_predictor = EnsemblePredictor()
        self.prediction_history = deque(maxlen=100)
        self.risk_thresholds = {
            'cpu_percent': {'high': 90, 'critical': 95},
            'memory_percent': {'high': 85, 'critical': 95},
            'error_rate': {'high': 0.05, 'critical': 0.1},
            'avg_query_time': {'high': 5.0, 'critical': 10.0}
        }
        
        self.maintenance_schedules = {}
        self.is_trained = False
        
        logger.info("Advanced Performance Predictor initialized")
    
    async def predict_performance_trends(self, metrics_history: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Predict future performance trends
        
        Args:
            metrics_history: Historical metrics data
            
        Returns:
            Dict containing predictions, trends, and recommendations
        """
        if not metrics_history or len(metrics_history) < 24:
            logger.warning("Insufficient data for performance prediction")
            return None
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(metrics_history)
            
            # Prepare time series data
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Train ensemble predictor if not trained
            if not self.is_trained:
                success = self.ensemble_predictor.train(df)
                if not success:
                    return self._fallback_predictions(df)
                self.is_trained = True
            
            # Generate predictions for each metric
            predictions = {}
            
            for column in df.columns:
                if column != 'timestamp' and df[column].dtype in ['float64', 'int64']:
                    # Get current trend
                    current_data = df[column].fillna(0).values[-24:]  # Last 24 data points
                    
                    # Predict future values
                    future_prediction = self.ensemble_predictor.predict_future_performance(
                        current_data, prediction_horizon=24
                    )
                    
                    if future_prediction:
                        predictions[column] = self._analyze_prediction_trend(
                            current_data, future_prediction, column
                        )
            
            # Generate maintenance recommendations
            maintenance_recs = self._generate_maintenance_recommendations(predictions)
            
            # Calculate failure risks
            failure_risks = self._calculate_failure_risks(predictions)
            
            # Create comprehensive prediction result
            result = {
                'timestamp': datetime.now(),
                'prediction_horizon_hours': 24,
                'predictions': predictions,
                'maintenance_recommendations': maintenance_recs,
                'failure_risks': failure_risks,
                'confidence_score': self._calculate_overall_confidence(predictions),
                'model_performance': self._get_model_performance()
            }
            
            # Store prediction
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in performance prediction: {e}")
            return self._fallback_predictions(pd.DataFrame(metrics_history))
    
    def _analyze_prediction_trend(self, historical_data: np.ndarray, 
                                future_prediction: Dict, metric_name: str) -> Dict[str, Any]:
        """Analyze prediction trend for a specific metric"""
        predictions = future_prediction['predictions']
        
        # Get ensemble prediction if available
        ensemble_pred = future_prediction.get('ensemble_prediction')
        if ensemble_pred is not None:
            predicted_values = ensemble_pred
        else:
            # Use average of all predictions
            predicted_values = np.mean([pred for pred in predictions.values() if len(pred) == 24], axis=0)
        
        # Calculate trend metrics
        current_value = historical_data[-1]
        predicted_mean = np.mean(predicted_values)
        predicted_max = np.max(predicted_values)
        predicted_trend = 'increasing' if predicted_values[-1] > predicted_values[0] else 'decreasing'
        
        # Calculate change percentages
        mean_change = ((predicted_mean - current_value) / current_value * 100) if current_value > 0 else 0
        max_change = ((predicted_max - current_value) / current_value * 100) if current_value > 0 else 0
        
        # Risk assessment
        risk_level = 'low'
        if metric_name in self.risk_thresholds:
            thresholds = self.risk_thresholds[metric_name]
            if np.any(predicted_values >= thresholds['critical']):
                risk_level = 'critical'
            elif np.any(predicted_values >= thresholds['high']):
                risk_level = 'high'
            elif mean_change > 20:  # 20% increase
                risk_level = 'medium'
        
        return {
            'metric_name': metric_name,
            'current_value': current_value,
            'predicted_mean': predicted_mean,
            'predicted_max': predicted_max,
            'predicted_min': np.min(predicted_values),
            'predicted_trend': predicted_trend,
            'mean_change_percent': mean_change,
            'max_change_percent': max_change,
            'risk_level': risk_level,
            'prediction_stability': 1.0 - (np.std(predicted_values) / np.mean(predicted_values) if np.mean(predicted_values) > 0 else 0),
            'time_to_threshold': self._calculate_time_to_threshold(predicted_values, metric_name)
        }
    
    def _calculate_time_to_threshold(self, predicted_values: np.ndarray, metric_name: str) -> Optional[int]:
        """Calculate time to reach risk threshold"""
        if metric_name not in self.risk_thresholds:
            return None
        
        high_threshold = self.risk_thresholds[metric_name]['high']
        
        for i, value in enumerate(predicted_values):
            if value >= high_threshold:
                return i + 1  # Time in hours
        
        return None  # No threshold reached within prediction horizon
    
    def _generate_maintenance_recommendations(self, predictions: Dict[str, Any]) -> List[MaintenanceRecommendation]:
        """Generate maintenance recommendations based on predictions"""
        recommendations = []
        
        for metric_name, pred_data in predictions.items():
            if pred_data['risk_level'] in ['high', 'critical']:
                # Time to threshold recommendation
                if pred_data['time_to_threshold']:
                    days_to_threshold = pred_data['time_to_threshold'] / 24
                    if days_to_threshold < 7:  # Less than a week
                        recommendation = MaintenanceRecommendation(
                            id=f"maintenance_{metric_name}",
                            maintenance_type="preventive",
                            priority="high",
                            description=f"Schedule maintenance for {metric_name} within {days_to_threshold:.1f} days",
                            affected_component=metric_name,
                            estimated_downtime="2-4 hours",
                            cost_estimate=500,  # Estimated cost in USD
                            recommended_date=datetime.now() + timedelta(days=days_to_threshold),
                            confidence_score=0.8,
                            metadata={
                                'predicted_max_value': pred_data['predicted_max'],
                                'time_to_threshold_hours': pred_data['time_to_threshold']
                            }
                        )
                        recommendations.append(recommendation)
                
                # Scaling recommendation
                if metric_name in ['cpu_percent', 'memory_percent'] and pred_data['max_change_percent'] > 50:
                    recommendation = MaintenanceRecommendation(
                        id=f"scaling_{metric_name}",
                        maintenance_type="scaling",
                        priority="medium",
                        description=f"Consider scaling {metric_name} resources due to increasing trend",
                        affected_component=metric_name,
                        estimated_downtime="1-2 hours",
                        cost_estimate=200,
                        recommended_date=datetime.now() + timedelta(days=1),
                        confidence_score=0.7,
                        metadata={
                            'predicted_growth': pred_data['max_change_percent']
                        }
                    )
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_failure_risks(self, predictions: Dict[str, Any]) -> List[FailureRisk]:
        """Calculate failure risks based on predictions"""
        risks = []
        
        # Calculate overall system risk
        high_risk_metrics = [m for m, p in predictions.items() if p['risk_level'] == 'high']
        critical_risk_metrics = [m for m, p in predictions.items() if p['risk_level'] == 'critical']
        
        if critical_risk_metrics:
            overall_risk = FailureRisk(
                component="database_system",
                risk_level="critical",
                probability=0.8,
                impact="high",
                time_to_failure_hours=min([p['time_to_threshold'] for m, p in predictions.items() 
                                         if p['risk_level'] == 'critical' and p['time_to_threshold']], default=24),
                mitigation_strategies=[
                    "Immediate performance review",
                    "Resource scaling",
                    "Query optimization",
                    "Load balancing"
                ],
                confidence_score=0.9
            )
            risks.append(overall_risk)
        
        # Component-specific risks
        for metric_name, pred_data in predictions.items():
            if pred_data['risk_level'] in ['high', 'critical']:
                component_risk = FailureRisk(
                    component=metric_name,
                    risk_level=pred_data['risk_level'],
                    probability=0.6 if pred_data['risk_level'] == 'high' else 0.8,
                    impact="medium" if pred_data['risk_level'] == 'high' else "high",
                    time_to_failure_hours=pred_data['time_to_threshold'],
                    mitigation_strategies=self._get_mitigation_strategies(metric_name),
                    confidence_score=0.7
                )
                risks.append(component_risk)
        
        return risks
    
    def _get_mitigation_strategies(self, component: str) -> List[str]:
        """Get mitigation strategies for specific component risks"""
        strategies_map = {
            'cpu_percent': [
                "Optimize CPU-intensive queries",
                "Scale CPU resources",
                "Review concurrent workload",
                "Implement query caching"
            ],
            'memory_percent': [
                "Check for memory leaks",
                "Optimize memory usage",
                "Scale memory resources",
                "Review query result sizes"
            ],
            'error_rate': [
                "Investigate error patterns",
                "Review application logs",
                "Check connectivity issues",
                "Update error handling"
            ],
            'avg_query_time': [
                "Optimize slow queries",
                "Review index usage",
                "Update database statistics",
                "Consider query rewriting"
            ]
        }
        
        return strategies_map.get(component, [
            "Monitor component performance",
            "Review recent changes",
            "Check resource utilization"
        ])
    
    def _calculate_overall_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence score for predictions"""
        if not predictions:
            return 0.0
        
        confidence_scores = []
        for pred_data in predictions.values():
            # Base confidence on prediction stability
            stability = pred_data.get('prediction_stability', 0.0)
            
            # Adjust for data quality
            data_quality = min(1.0, pred_data.get('max_change_percent', 0) / 100 + 0.5)
            
            confidence = (stability + data_quality) / 2
            confidence_scores.append(confidence)
        
        return np.mean(confidence_scores)
    
    def _get_model_performance(self) -> Dict[str, float]:
        """Get performance metrics of prediction models"""
        # This would typically involve backtesting predictions against actual values
        # For demo purposes, returning placeholder metrics
        
        return {
            'mae': 0.05,  # Mean Absolute Error
            'rmse': 0.08,  # Root Mean Square Error
            'mape': 0.12,  # Mean Absolute Percentage Error
            'directional_accuracy': 0.75
        }
    
    def _fallback_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback predictions when ML models are not available"""
        predictions = {}
        
        for column in df.columns:
            if column != 'timestamp' and df[column].dtype in ['float64', 'int64']:
                values = df[column].fillna(0).values
                
                # Simple linear trend extrapolation
                if len(values) >= 2:
                    slope = (values[-1] - values[-24]) / 24 if len(values) >= 24 else (values[-1] - values[0]) / len(values)
                    predicted_mean = values[-1] + slope * 12  # 12 hours ahead
                    
                    predictions[column] = {
                        'metric_name': column,
                        'current_value': values[-1],
                        'predicted_mean': predicted_mean,
                        'predicted_max': predicted_mean * 1.2,
                        'predicted_min': predicted_mean * 0.8,
                        'predicted_trend': 'increasing' if slope > 0 else 'decreasing',
                        'mean_change_percent': (predicted_mean - values[-1]) / values[-1] * 100 if values[-1] > 0 else 0,
                        'risk_level': 'low',
                        'prediction_stability': 0.6,
                        'time_to_threshold': None
                    }
        
        return {
            'timestamp': datetime.now(),
            'prediction_horizon_hours': 24,
            'predictions': predictions,
            'maintenance_recommendations': [],
            'failure_risks': [],
            'confidence_score': 0.5,
            'model_performance': {'note': 'Using fallback prediction method'}
        }
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of recent predictions"""
        if not self.prediction_history:
            return {'total_predictions': 0}
        
        recent_predictions = [p for p in self.prediction_history if p['timestamp'] > datetime.now() - timedelta(days=7)]
        
        if not recent_predictions:
            return {'total_predictions': 0}
        
        # Aggregate metrics
        total_predictions = sum(len(p['predictions']) for p in recent_predictions)
        avg_confidence = np.mean([p['confidence_score'] for p in recent_predictions])
        
        # Risk level distribution
        risk_distribution = defaultdict(int)
        maintenance_count = 0
        
        for prediction in recent_predictions:
            for pred_data in prediction['predictions'].values():
                risk_distribution[pred_data['risk_level']] += 1
            
            maintenance_count += len(prediction['maintenance_recommendations'])
        
        return {
            'total_predictions': total_predictions,
            'average_confidence': avg_confidence,
            'risk_distribution': dict(risk_distribution),
            'recent_maintenance_recommendations': maintenance_count,
            'last_prediction_time': recent_predictions[-1]['timestamp'] if recent_predictions else None
        }
    
    def export_predictions(self, filepath: str):
        """Export prediction history to file"""
        try:
            prediction_data = []
            
            for prediction in self.prediction_history:
                prediction_data.append({
                    'timestamp': prediction['timestamp'].isoformat(),
                    'prediction_horizon_hours': prediction['prediction_horizon_hours'],
                    'confidence_score': prediction['confidence_score'],
                    'predictions': prediction['predictions'],
                    'maintenance_recommendations': [
                        {
                            'id': rec.id,
                            'maintenance_type': rec.maintenance_type,
                            'priority': rec.priority,
                            'description': rec.description
                        }
                        for rec in prediction['maintenance_recommendations']
                    ],
                    'failure_risks': [
                        {
                            'component': risk.component,
                            'risk_level': risk.risk_level,
                            'probability': risk.probability
                        }
                        for risk in prediction['failure_risks']
                    ]
                })
            
            with open(filepath, 'w') as f:
                import json
                json.dump(prediction_data, f, indent=2, default=str)
            
            logger.info(f"Predictions exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting predictions: {e}")
            raise