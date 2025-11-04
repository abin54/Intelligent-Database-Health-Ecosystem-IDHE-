"""
Data Models for IDHE (Intelligent Database Health Ecosystem)
==========================================================

This module defines all data structures and models used throughout the IDHE system.
These models provide type safety and structured data representation.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

# ================================
# Core Database Models
# ================================

@dataclass
class DatabaseMetrics:
    """Comprehensive database metrics model"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    active_connections: int
    max_connections: int
    avg_query_time: float
    max_query_time: float
    query_count_1min: int
    storage_used: int
    storage_total: int
    storage_utilization: float
    error_rate: float
    throughput_per_sec: float
    cache_hit_ratio: float = 0.0
    io_reads_per_sec: float = 0.0
    io_writes_per_sec: float = 0.0
    lock_waits_per_min: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'avg_query_time': self.avg_query_time,
            'max_query_time': self.max_query_time,
            'query_count_1min': self.query_count_1min,
            'storage_used': self.storage_used,
            'storage_total': self.storage_total,
            'storage_utilization': self.storage_utilization,
            'error_rate': self.error_rate,
            'throughput_per_sec': self.throughput_per_sec,
            'cache_hit_ratio': self.cache_hit_ratio,
            'io_reads_per_sec': self.io_reads_per_sec,
            'io_writes_per_sec': self.io_writes_per_sec,
            'lock_waits_per_min': self.lock_waits_per_min
        }

@dataclass
class QueryPerformance:
    """Individual query performance tracking"""
    query: str
    query_type: str
    execution_time: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    user: str = "unknown"
    database: str = "default"
    rows_affected: int = 0
    rows_examined: int = 0
    execution_plan: str = ""

@dataclass
class ConnectionMetrics:
    """Database connection metrics"""
    timestamp: datetime
    active_connections: int
    idle_connections: int
    max_connections: int
    connection_rate_per_sec: float
    average_connection_lifetime: float
    connection_pool_utilization: float
    failed_connections_per_min: int = 0

# ================================
# Health and Anomaly Models
# ================================

@dataclass
class HealthScore:
    """Overall database health score breakdown"""
    overall_score: float
    performance_score: float
    connection_score: float
    storage_score: float
    error_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    health_level: str = field(init=False)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.overall_score >= 90:
            self.health_level = "excellent"
        elif self.overall_score >= 80:
            self.health_level = "good"
        elif self.overall_score >= 70:
            self.health_level = "fair"
        elif self.overall_score >= 60:
            self.health_level = "poor"
        else:
            self.health_level = "critical"
        
        # Generate recommendations based on low scores
        self.recommendations = []
        if self.performance_score < 70:
            self.recommendations.append("Optimize slow queries and improve indexing")
        if self.connection_score < 70:
            self.recommendations.append("Review connection pool settings and connection leaks")
        if self.storage_score < 70:
            self.recommendations.append("Implement data archiving and optimize storage")
        if self.error_score < 70:
            self.recommendations.append("Investigate error patterns and improve error handling")

@dataclass
class AnomalyAlert:
    """Security and performance anomaly alert"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    anomaly_type: str = ""
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    affected_metrics: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'description': self.description,
            'affected_metrics': self.affected_metrics,
            'confidence_score': self.confidence_score,
            'recommended_actions': self.recommended_actions,
            'metadata': self.metadata
        }

@dataclass
class MetricAnomaly:
    """Specific metric anomaly detection result"""
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    severity: str
    detection_method: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QueryAnomaly:
    """Query pattern anomaly"""
    query_hash: str
    query_pattern: str
    expected_frequency: float
    actual_frequency: float
    frequency_deviation: float
    severity: str

@dataclass
class ConnectionAnomaly:
    """Connection pattern anomaly"""
    connection_type: str
    expected_pattern: str
    actual_pattern: str
    severity: str
    affected_users: List[str] = field(default_factory=list)

# ================================
# Security Models
# ================================

@dataclass
class SecurityAlert:
    """Security threat and vulnerability alert"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    alert_type: str = ""  # sql_injection, privilege_escalation, data_breach, etc.
    severity: str = "medium"
    description: str = ""
    affected_queries: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'description': self.description,
            'affected_queries': self.affected_queries,
            'affected_users': self.affected_users,
            'confidence_score': self.confidence_score,
            'recommended_actions': self.recommended_actions,
            'metadata': self.metadata
        }

@dataclass
class VulnerabilityAssessment:
    """Security vulnerability assessment result"""
    vulnerability_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    vulnerability_type: str = ""
    severity: str = "low"
    description: str = ""
    affected_components: List[str] = field(default_factory=list)
    cvss_score: float = 0.0
    remediation_priority: str = "low"
    estimated_fix_time: str = "unknown"
    recommended_actions: List[str] = field(default_factory=list)

@dataclass
class AccessPattern:
    """User access pattern analysis"""
    user: str
    access_frequency: float
    typical_access_hours: List[int]
    typical_tables_accessed: List[str]
    privilege_level: str
    risk_score: float
    last_access: datetime
    access_anomalies: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    threat_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    threat_type: str = ""
    threat_level: str = "low"
    ioc_indicators: List[str] = field(default_factory=list)
    attribution: str = "unknown"
    ttp_tactics: List[str] = field(default_factory=list)
    confidence_level: float = 0.5
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

# ================================
# Machine Learning Models
# ================================

@dataclass
class QueryAnalysis:
    """Comprehensive query analysis for optimization"""
    query: str
    query_type: str
    complexity_score: float
    tables_involved: List[str]
    columns_involved: List[str]
    join_complexity: Dict[str, Any]
    filtering_complexity: Dict[str, Any]
    aggregation_level: Dict[str, Any]
    sorting_requirements: Dict[str, Any]
    subquery_depth: int
    estimated_execution_time: float = 0.0
    optimization_suggestions: List[str] = field(default_factory=list)

@dataclass
class QueryOptimization:
    """ML-powered query optimization result"""
    original_query: str
    analysis: QueryAnalysis
    predicted_execution_time: float
    baseline_time: float
    improvement_potential: float
    index_recommendations: List['IndexRecommendation'] = field(default_factory=list)
    optimization_strategies: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.5
    estimated_cost: float = 0.0
    potential_savings: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class IndexRecommendation:
    """Database index optimization recommendation"""
    table: str
    columns: List[str]
    index_type: str = "B-TREE"  # B-TREE, HASH, GIN, GIST, etc.
    estimated_improvement: float = 0.0
    confidence: float = 0.5
    maintenance_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'table': self.table,
            'columns': self.columns,
            'index_type': self.index_type,
            'estimated_improvement': self.estimated_improvement,
            'confidence': self.confidence,
            'maintenance_cost': self.maintenance_cost,
            'created_at': self.created_at.isoformat()
        }

# ================================
# Performance Prediction Models
# ================================

@dataclass
class PerformancePrediction:
    """Performance forecasting and prediction result"""
    metric_name: str
    current_value: float
    predicted_values: List[float]
    confidence_intervals: Dict[str, List[float]] = field(default_factory=dict)
    prediction_horizon_hours: int = 24
    trend_direction: str = "stable"
    growth_rate: float = 0.0
    risk_level: str = "low"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class MaintenanceRecommendation:
    """Predictive maintenance recommendation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    maintenance_type: str = ""  # preventive, corrective, predictive
    priority: str = "medium"  # low, medium, high, critical
    description: str = ""
    affected_component: str = ""
    estimated_downtime: str = "unknown"
    cost_estimate: float = 0.0
    recommended_date: Optional[datetime] = None
    confidence_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FailureRisk:
    """Component failure risk assessment"""
    component: str
    risk_level: str = "low"  # low, medium, high, critical
    probability: float = 0.0
    impact: str = "low"
    time_to_failure_hours: Optional[int] = None
    mitigation_strategies: List[str] = field(default_factory=list)
    confidence_score: float = 0.5

# ================================
# Capacity Planning Models
# ================================

@dataclass
class CapacityForecast:
    """Resource capacity forecasting result"""
    resource_type: str  # cpu, memory, storage, connections
    current_capacity: float
    forecasted_demand: List[float]
    optimal_capacity: float
    scaling_recommendation: str = "maintain"  # scale_up, scale_down, maintain
    scaling_timeline: str = "unknown"
    cost_estimate: float = 0.0
    confidence_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ScalingRecommendation:
    """Resource scaling recommendation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    resource_type: str = ""
    scaling_type: str = "scale_up"  # scale_up, scale_down
    current_capacity: float = 0.0
    recommended_capacity: float = 0.0
    scaling_factor: float = 1.0
    estimated_cost: float = 0.0
    estimated_benefit: float = 0.0
    confidence_score: float = 0.5
    implementation_timeline: Optional[datetime] = None
    risk_level: str = "low"
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'resource_type': self.resource_type,
            'scaling_type': self.scaling_type,
            'current_capacity': self.current_capacity,
            'recommended_capacity': self.recommended_capacity,
            'scaling_factor': self.scaling_factor,
            'estimated_cost': self.estimated_cost,
            'estimated_benefit': self.estimated_benefit,
            'confidence_score': self.confidence_score,
            'implementation_timeline': self.implementation_timeline.isoformat() if self.implementation_timeline else None,
            'risk_level': self.risk_level,
            'recommended_actions': self.recommended_actions
        }

@dataclass
class ResourceAllocation:
    """Optimized resource allocation recommendation"""
    resource_type: str
    current_allocation: float
    recommended_allocation: float
    allocation_efficiency: float
    optimization_score: float
    cost_impact: float
    performance_impact: float
    risk_assessment: str = "low"

# ================================
# Enums for Better Type Safety
# ================================

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HealthLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    PATTERN = "pattern"
    BEHAVIORAL = "behavioral"
    THRESHOLD = "threshold"

class SecurityThreatType(Enum):
    SQL_INJECTION = "sql_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

class QueryType(Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"

class MaintenanceType(Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"

class ScalingType(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

# ================================
# Configuration Models
# ================================

@dataclass
class IDHEConfig:
    """Main IDHE system configuration"""
    database_url: str = "sqlite:///idhe_health.db"
    redis_url: str = "redis://localhost:6379/0"
    update_interval: int = 30  # seconds
    retention_days: int = 90
    enable_ml_optimization: bool = True
    enable_predictive_maintenance: bool = True
    enable_security_scanning: bool = True
    performance_threshold: float = 0.8
    anomaly_sensitivity: float = 0.95
    log_level: str = "INFO"
    max_concurrent_monitors: int = 4
    alert_cooldown_minutes: int = 15
    
    # Security settings
    security_scan_interval: int = 3600  # seconds
    max_failed_attempts: int = 10
    sql_injection_threshold: float = 0.7
    
    # Capacity planning
    forecast_horizon_days: int = 30
    capacity_warning_threshold: float = 0.8
    auto_scaling_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'database_url': self.database_url,
            'redis_url': self.redis_url,
            'update_interval': self.update_interval,
            'retention_days': self.retention_days,
            'enable_ml_optimization': self.enable_ml_optimization,
            'enable_predictive_maintenance': self.enable_predictive_maintenance,
            'enable_security_scanning': self.enable_security_scanning,
            'performance_threshold': self.performance_threshold,
            'anomaly_sensitivity': self.anomaly_sensitivity,
            'log_level': self.log_level,
            'max_concurrent_monitors': self.max_concurrent_monitors,
            'alert_cooldown_minutes': self.alert_cooldown_minutes,
            'security_scan_interval': self.security_scan_interval,
            'max_failed_attempts': self.max_failed_attempts,
            'sql_injection_threshold': self.sql_injection_threshold,
            'forecast_horizon_days': self.forecast_horizon_days,
            'capacity_warning_threshold': self.capacity_warning_threshold,
            'auto_scaling_enabled': self.auto_scaling_enabled
        }

# ================================
# API Response Models
# ================================

@dataclass
class APIResponse:
    """Standard API response model"""
    success: bool
    message: str
    data: Any = None
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class DashboardMetrics:
    """Dashboard metrics summary"""
    overall_health_score: float
    active_alerts: int
    total_queries_monitored: int
    security_threats_detected: int
    scaling_recommendations: int
    predicted_maintenance_items: int
    capacity_utilization: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dashboard"""
        return {
            'overall_health_score': self.overall_health_score,
            'active_alerts': self.active_alerts,
            'total_queries_monitored': self.total_queries_monitored,
            'security_threats_detected': self.security_threats_detected,
            'scaling_recommendations': self.scaling_recommendations,
            'predicted_maintenance_items': self.predicted_maintenance_items,
            'capacity_utilization': self.capacity_utilization,
            'performance_trends': self.performance_trends,
            'last_updated': self.last_updated.isoformat()
        }