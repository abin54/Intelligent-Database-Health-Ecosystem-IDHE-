#!/usr/bin/env python3
"""
Intelligent Database Health Ecosystem (IDHE)
===========================================

A revolutionary system that combines:
- Real-time SQL performance analytics
- Machine learning-based query optimization
- Predictive database maintenance
- Intelligent indexing recommendations
- Anomaly detection and alerting
- Automated performance tuning

This is a unique, advanced project not available anywhere else.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from pathlib import Path

# Import our custom modules
from core.database_monitor import DatabaseMonitor
from core.ml_optimizer import MLQueryOptimizer
from core.anomaly_detector import AnomalyDetector
from core.performance_predictor import PerformancePredictor
from core.security_scanner import SecurityScanner
from core.capacity_planner import CapacityPlanner
from utils.logger_config import setup_logging
from utils.config_manager import ConfigManager
from utils.data_models import DatabaseMetrics, QueryAnalysis, HealthScore
from api.rest_api import IDHERESTAPI
from dashboard.web_dashboard import IDHEDashboard

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

@dataclass
class IDHEConfig:
    """Configuration for the IDHE system"""
    database_url: str = "sqlite:///idhe_health.db"
    redis_url: str = "redis://localhost:6379/0"
    update_interval: int = 30  # seconds
    retention_days: int = 90
    enable_ml_optimization: bool = True
    enable_predictive_maintenance: bool = True
    enable_security_scanning: bool = True
    performance_threshold: float = 0.8
    anomaly_sensitivity: float = 0.95

class IntelligentDatabaseHealthEcosystem:
    """
    The main IDHE system that orchestrates all components
    
    This is a revolutionary approach to database health management that combines:
    - Real-time monitoring with ML-powered analytics
    - Predictive maintenance using time series forecasting
    - Intelligent query optimization with neural networks
    - Automated security scanning and threat detection
    """
    
    def __init__(self, config: IDHEConfig):
        self.config = config
        self.is_running = False
        self.start_time = datetime.now()
        
        # Initialize core components
        self.database_monitor = DatabaseMonitor(config.database_url)
        self.ml_optimizer = MLQueryOptimizer() if config.enable_ml_optimization else None
        self.anomaly_detector = AnomalyDetector(config.anomaly_sensitivity)
        self.performance_predictor = PerformancePredictor() if config.enable_predictive_maintenance else None
        self.security_scanner = SecurityScanner() if config.enable_security_scanning else None
        self.capacity_planner = CapacityPlanner()
        
        # Initialize API and dashboard
        self.api = IDHERESTAPI(self)
        self.dashboard = IDHEDashboard(self)
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.metrics_history = []
        self.health_scores = []
        self.query_performance = {}
        
        logger.info("IDHE System initialized successfully")
    
    async def start(self):
        """Start the IDHE system"""
        try:
            logger.info("Starting Intelligent Database Health Ecosystem...")
            self.is_running = True
            
            # Start background tasks
            await asyncio.gather(
                self._monitoring_loop(),
                self._optimization_loop() if self.ml_optimizer else asyncio.sleep(0),
                self._anomaly_detection_loop(),
                self._predictive_maintenance_loop() if self.performance_predictor else asyncio.sleep(0),
                self._security_scanning_loop() if self.security_scanner else asyncio.sleep(0),
                self._capacity_planning_loop()
            )
            
        except Exception as e:
            logger.error(f"Error starting IDHE: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Continuous database monitoring loop"""
        while self.is_running:
            try:
                # Collect database metrics
                metrics = await self.database_monitor.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim old data based on retention policy
                cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_date
                ]
                
                # Calculate health score
                health_score = self._calculate_health_score(metrics)
                self.health_scores.append(health_score)
                
                # Log performance summary
                logger.info(f"DB Health Score: {health_score.overall_score:.2f} | "
                          f"Query Performance: {metrics.avg_query_time:.3f}s | "
                          f"Connections: {metrics.active_connections}")
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.update_interval)
    
    async def _optimization_loop(self):
        """ML-based query optimization loop"""
        while self.is_running and self.ml_optimizer:
            try:
                # Analyze slow queries
                slow_queries = self._identify_slow_queries()
                
                for query_info in slow_queries:
                    # Get ML-powered optimization recommendations
                    optimization = await self.ml_optimizer.optimize_query(
                        query_info['sql'],
                        query_info['execution_time'],
                        query_info['frequency']
                    )
                    
                    if optimization:
                        logger.info(f"ML Optimization: {optimization.recommendation}")
                        # Store optimization for analysis
                        self.query_performance[query_info['sql']] = optimization
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)
    
    async def _anomaly_detection_loop(self):
        """Anomaly detection and alerting"""
        while self.is_running:
            try:
                if len(self.metrics_history) > 10:  # Need sufficient data
                    # Detect anomalies in recent data
                    recent_metrics = self.metrics_history[-50:]  # Last 50 data points
                    anomalies = self.anomaly_detector.detect_anomalies(recent_metrics)
                    
                    if anomalies:
                        await self._handle_anomalies(anomalies)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(60)
    
    async def _predictive_maintenance_loop(self):
        """Predictive maintenance using time series forecasting"""
        while self.is_running and self.performance_predictor:
            try:
                if len(self.metrics_history) > 100:  # Need sufficient historical data
                    # Predict future performance trends
                    predictions = await self.performance_predictor.predict_performance_trends(
                        self.metrics_history
                    )
                    
                    if predictions:
                        # Generate maintenance recommendations
                        maintenance_items = self._generate_maintenance_recommendations(predictions)
                        
                        for item in maintenance_items:
                            if item.priority == 'HIGH':
                                logger.warning(f"Predictive Maintenance: {item.description}")
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                logger.error(f"Predictive maintenance error: {e}")
                await asyncio.sleep(1800)
    
    async def _security_scanning_loop(self):
        """Security scanning and threat detection"""
        while self.is_running and self.security_scanner:
            try:
                # Perform security scan
                security_issues = await self.security_scanner.scan_database()
                
                if security_issues:
                    await self._handle_security_issues(security_issues)
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Security scanning error: {e}")
                await asyncio.sleep(3600)
    
    async def _capacity_planning_loop(self):
        """Database capacity planning and forecasting"""
        while self.is_running:
            try:
                if len(self.metrics_history) > 50:
                    # Analyze capacity trends
                    capacity_analysis = await self.capacity_planner.analyze_capacity_trends(
                        self.metrics_history
                    )
                    
                    if capacity_analysis.growth_rate > 0.8:  # 80% capacity warning
                        logger.warning(f"Capacity Planning: Growth rate {capacity_analysis.growth_rate:.1%}. "
                                     f"Consider scaling at {capacity_analysis.scaling_recommendation}")
                
                await asyncio.sleep(7200)  # Run every 2 hours
                
            except Exception as e:
                logger.error(f"Capacity planning error: {e}")
                await asyncio.sleep(7200)
    
    def _calculate_health_score(self, metrics: DatabaseMetrics) -> HealthScore:
        """Calculate comprehensive database health score"""
        # Performance score (0-100)
        perf_score = max(0, 100 - (metrics.avg_query_time * 10))
        
        # Connection utilization score
        conn_score = max(0, 100 - (metrics.active_connections / metrics.max_connections * 100))
        
        # Storage utilization score
        storage_score = max(0, 100 - (metrics.storage_used / metrics.storage_total * 100))
        
        # Error rate score
        error_score = max(0, 100 - (metrics.error_rate * 1000))
        
        # Weighted overall score
        overall_score = (
            perf_score * 0.3 +
            conn_score * 0.2 +
            storage_score * 0.3 +
            error_score * 0.2
        )
        
        return HealthScore(
            overall_score=overall_score,
            performance_score=perf_score,
            connection_score=conn_score,
            storage_score=storage_score,
            error_score=error_score,
            timestamp=metrics.timestamp
        )
    
    def _identify_slow_queries(self) -> List[Dict]:
        """Identify slow queries that need optimization"""
        # This would typically query a query log table
        # For demo purposes, returning sample data
        return [
            {
                'sql': 'SELECT * FROM users WHERE age > 30',
                'execution_time': 2.5,
                'frequency': 150
            },
            {
                'sql': 'SELECT COUNT(*) FROM orders WHERE date > NOW() - INTERVAL 1 DAY',
                'execution_time': 1.8,
                'frequency': 200
            }
        ]
    
    async def _handle_anomalies(self, anomalies: List[Dict]):
        """Handle detected anomalies"""
        for anomaly in anomalies:
            logger.warning(f"Anomaly detected: {anomaly['type']} - {anomaly['description']}")
            
            # Here you could send alerts, create tickets, etc.
            if anomaly['severity'] == 'CRITICAL':
                # Take immediate action
                pass
    
    async def _handle_security_issues(self, security_issues: List[Dict]):
        """Handle security issues"""
        for issue in security_issues:
            logger.critical(f"Security Issue: {issue['type']} - {issue['description']}")
            # Here you could send alerts, take preventive actions, etc.
    
    def _generate_maintenance_recommendations(self, predictions: Dict) -> List[Dict]:
        """Generate maintenance recommendations based on predictions"""
        recommendations = []
        
        if predictions.get('storage_forecast', {}).get('days_to_full', 30) < 90:
            recommendations.append({
                'type': 'storage',
                'priority': 'HIGH',
                'description': 'Storage capacity will be exhausted in less than 90 days',
                'action': 'Consider adding storage or archiving old data'
            })
        
        if predictions.get('performance_forecast', {}).get('degradation_rate', 0) > 0.1:
            recommendations.append({
                'type': 'performance',
                'priority': 'MEDIUM',
                'description': 'Performance degradation trend detected',
                'action': 'Consider query optimization and index maintenance'
            })
        
        return recommendations
    
    def get_current_health(self) -> Dict:
        """Get current database health status"""
        if not self.health_scores:
            return {'status': 'unknown', 'score': 0}
        
        latest_score = self.health_scores[-1]
        return {
            'status': 'healthy' if latest_score.overall_score > 80 else 'warning' if latest_score.overall_score > 60 else 'critical',
            'score': latest_score.overall_score,
            'components': {
                'performance': latest_score.performance_score,
                'connections': latest_score.connection_score,
                'storage': latest_score.storage_score,
                'errors': latest_score.error_score
            }
        }
    
    def get_recent_metrics(self, hours: int = 24) -> List[DatabaseMetrics]:
        """Get recent database metrics"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff]
    
    def stop(self):
        """Stop the IDHE system"""
        logger.info("Stopping IDHE System...")
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("IDHE System stopped")

async def main():
    """Main entry point"""
    # Load configuration
    config = ConfigManager.load_config()
    
    # Create and start IDHE system
    idhe = IntelligentDatabaseHealthEcosystem(config)
    
    try:
        # Start the system
        await idhe.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        idhe.stop()

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())