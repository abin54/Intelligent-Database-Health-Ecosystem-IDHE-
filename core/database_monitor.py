"""
Core Database Monitor
====================

Advanced real-time database monitoring with performance analytics,
query performance tracking, and system health assessment.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sqlite3
import psutil
import threading
from contextlib import contextmanager
import statistics
from collections import deque, defaultdict

from ..utils.data_models import DatabaseMetrics, QueryPerformance, ConnectionMetrics

logger = logging.getLogger(__name__)

class DatabaseMonitor:
    """
    Advanced Database Monitor
    
    Features:
    - Real-time performance tracking
    - Query performance analysis
    - Connection pool monitoring
    - Storage utilization tracking
    - Error rate monitoring
    - Custom metrics collection
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.monitoring_active = False
        self.query_history = deque(maxlen=1000)
        self.performance_samples = deque(maxlen=100)
        self.connection_history = deque(maxlen=100)
        self.error_history = deque(maxlen=200)
        
        # Performance thresholds
        self.thresholds = {
            'query_time': 1.0,  # seconds
            'error_rate': 0.05,  # 5%
            'connection_usage': 0.8,  # 80%
            'storage_usage': 0.85  # 85%
        }
        
        # Start monitoring background task
        self.monitor_thread = None
        self.start_background_monitoring()
    
    def start_background_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._background_monitor,
                daemon=True
            )
            self.monitor_thread.start()
            logger.info("Background database monitoring started")
    
    def _background_monitor(self):
        """Background monitoring loop"""
        while True:
            try:
                # Collect basic system metrics
                self._collect_system_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(30)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Store in performance samples
            sample = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
            
            self.performance_samples.append(sample)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    @contextmanager
    def track_query(self, query: str, query_type: str = "unknown"):
        """Context manager to track query performance"""
        start_time = time.time()
        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Record query performance
            query_record = QueryPerformance(
                query=query,
                query_type=query_type,
                execution_time=execution_time,
                success=success,
                error=error,
                timestamp=datetime.now()
            )
            
            self.query_history.append(query_record)
            self._analyze_query_performance(query_record)
    
    def _analyze_query_performance(self, query_record: QueryPerformance):
        """Analyze query performance and detect issues"""
        if query_record.execution_time > self.thresholds['query_time']:
            logger.warning(f"Slow query detected: {query_record.execution_time:.3f}s - {query_record.query[:50]}...")
        
        if not query_record.success:
            self.error_history.append(query_record)
            
            # Calculate error rate in last 100 queries
            recent_queries = list(self.query_history)[-100:]
            if recent_queries:
                error_count = sum(1 for q in recent_queries if not q.success)
                error_rate = error_count / len(recent_queries)
                
                if error_rate > self.thresholds['error_rate']:
                    logger.critical(f"High error rate detected: {error_rate:.1%}")
    
    async def collect_metrics(self) -> DatabaseMetrics:
        """
        Collect comprehensive database metrics
        
        Returns:
            DatabaseMetrics: Complete database health metrics
        """
        try:
            # Get current system metrics
            current_metrics = self._get_current_system_metrics()
            
            # Analyze query performance
            query_metrics = self._analyze_query_metrics()
            
            # Get connection information
            connection_metrics = self._get_connection_metrics()
            
            # Calculate storage utilization
            storage_metrics = self._get_storage_metrics()
            
            # Calculate error rate
            error_rate = self._calculate_error_rate()
            
            # Create comprehensive metrics object
            db_metrics = DatabaseMetrics(
                timestamp=datetime.now(),
                cpu_percent=current_metrics['cpu_percent'],
                memory_percent=current_metrics['memory_percent'],
                active_connections=connection_metrics['active'],
                max_connections=connection_metrics['max'],
                avg_query_time=query_metrics['avg_time'],
                max_query_time=query_metrics['max_time'],
                query_count_1min=query_metrics['count_1min'],
                storage_used=storage_metrics['used'],
                storage_total=storage_metrics['total'],
                storage_utilization=storage_metrics['utilization'],
                error_rate=error_rate,
                throughput_per_sec=query_metrics['throughput'],
                cache_hit_ratio=query_metrics['cache_hit_ratio']
            )
            
            return db_metrics
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
            # Return default metrics on error
            return DatabaseMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                max_connections=100,
                avg_query_time=0.0,
                max_query_time=0.0,
                query_count_1min=0,
                storage_used=0,
                storage_total=1,
                storage_utilization=0.0,
                error_rate=0.0,
                throughput_per_sec=0.0,
                cache_hit_ratio=0.0
            )
    
    def _get_current_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'cpu_percent': 0.0, 'memory_percent': 0.0}
    
    def _analyze_query_metrics(self) -> Dict[str, float]:
        """Analyze query performance metrics"""
        if not self.query_history:
            return {
                'avg_time': 0.0,
                'max_time': 0.0,
                'count_1min': 0,
                'throughput': 0.0,
                'cache_hit_ratio': 0.0
            }
        
        recent_queries = [
            q for q in self.query_history 
            if q.timestamp > datetime.now() - timedelta(minutes=1)
        ]
        
        if not recent_queries:
            return {
                'avg_time': 0.0,
                'max_time': 0.0,
                'count_1min': 0,
                'throughput': 0.0,
                'cache_hit_ratio': 0.0
            }
        
        execution_times = [q.execution_time for q in recent_queries]
        
        return {
            'avg_time': statistics.mean(execution_times),
            'max_time': max(execution_times),
            'count_1min': len(recent_queries),
            'throughput': len(recent_queries) / 60.0,  # queries per second
            'cache_hit_ratio': 0.85 + (hash(str(recent_queries)) % 15) / 100  # Simulated cache hit ratio
        }
    
    def _get_connection_metrics(self) -> Dict[str, int]:
        """Get database connection metrics"""
        # For SQLite, connections are mostly for demonstration
        # In real implementation, this would query the database for connection info
        return {
            'active': min(len(self.query_history), 10),  # Simulated active connections
            'max': 100  # Default max connections
        }
    
    def _get_storage_metrics(self) -> Dict[str, int]:
        """Get storage utilization metrics"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'used': disk.used,
                'total': disk.total,
                'utilization': (disk.used / disk.total) * 100
            }
        except Exception as e:
            logger.error(f"Error getting storage metrics: {e}")
            return {'used': 0, 'total': 1, 'utilization': 0.0}
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent queries"""
        if not self.query_history:
            return 0.0
        
        recent_queries = list(self.query_history)[-100:]  # Last 100 queries
        if not recent_queries:
            return 0.0
        
        error_count = sum(1 for q in recent_queries if not q.success)
        return error_count / len(recent_queries)
    
    def get_slow_queries(self, threshold: float = None) -> List[QueryPerformance]:
        """Get slow queries above threshold"""
        if threshold is None:
            threshold = self.thresholds['query_time']
        
        return [
            q for q in self.query_history 
            if q.execution_time > threshold and q.success
        ]
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get comprehensive query statistics"""
        if not self.query_history:
            return {}
        
        # Group queries by type
        query_types = defaultdict(list)
        for q in self.query_history:
            query_types[q.query_type].append(q)
        
        stats = {
            'total_queries': len(self.query_history),
            'successful_queries': sum(1 for q in self.query_history if q.success),
            'failed_queries': sum(1 for q in self.query_history if not q.success),
            'avg_execution_time': statistics.mean([q.execution_time for q in self.query_history]),
            'query_types': {}
        }
        
        # Statistics by query type
        for query_type, queries in query_types.items():
            execution_times = [q.execution_time for q in queries]
            stats['query_types'][query_type] = {
                'count': len(queries),
                'avg_time': statistics.mean(execution_times),
                'max_time': max(execution_times),
                'success_rate': sum(1 for q in queries if q.success) / len(queries)
            }
        
        return stats
    
    def get_performance_trend(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_samples = [
            s for s in self.performance_samples 
            if s['timestamp'] > cutoff
        ]
        
        if not recent_samples:
            return {'cpu': [], 'memory': [], 'disk': []}
        
        return {
            'cpu': [s['cpu_percent'] for s in recent_samples],
            'memory': [s['memory_percent'] for s in recent_samples],
            'disk': [s['disk_percent'] for s in recent_samples]
        }
    
    def set_performance_threshold(self, metric: str, threshold: float):
        """Set performance threshold for alerting"""
        if metric in self.thresholds:
            self.thresholds[metric] = threshold
            logger.info(f"Set threshold for {metric}: {threshold}")
        else:
            logger.warning(f"Unknown threshold metric: {metric}")
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'query_history': [
                    {
                        'query': q.query,
                        'execution_time': q.execution_time,
                        'success': q.success,
                        'timestamp': q.timestamp.isoformat()
                    }
                    for q in self.query_history
                ],
                'performance_samples': [
                    {
                        'cpu_percent': s['cpu_percent'],
                        'memory_percent': s['memory_percent'],
                        'timestamp': s['timestamp'].isoformat()
                    }
                    for s in self.performance_samples
                ],
                'statistics': self.get_query_statistics()
            }
            
            with open(filepath, 'w') as f:
                import json
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise