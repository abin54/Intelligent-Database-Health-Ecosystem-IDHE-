#!/usr/bin/env python3
"""
IDHE Installation and Setup Script
=================================

This script automatically installs and configures the IDHE (Intelligent Database Health Ecosystem)
system with all its advanced components.
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   IDHE - Intelligent Database Health Ecosystem               ║
    ║                                                               ║
    ║   Revolutionary SQL-Python Database Management System        ║
    ║   • Machine Learning Query Optimization                      ║
    ║   • Advanced Anomaly Detection                               ║
    ║   • Predictive Performance Analysis                          ║
    ║   • AI-Powered Security Scanning                             ║
    ║   • Intelligent Capacity Planning                            ║
    ║   • Real-time Dashboard                                       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        print("   Please upgrade your Python version.")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Core dependencies
    core_packages = [
        'pandas>=2.1.0',
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        'scikit-learn>=1.3.0',
        'psutil>=5.9.0'
    ]
    
    # Optional ML packages
    ml_packages = [
        'tensorflow>=2.13.0',
        'statsmodels>=0.14.0',
        'prophet>=1.1.0'
    ]
    
    # Web framework packages
    web_packages = [
        'fastapi>=0.100.0',
        'uvicorn[standard]>=0.23.0',
        'dash>=2.14.0',
        'plotly>=5.16.0',
        'dash-bootstrap-components>=1.4.0'
    ]
    
    # Utility packages
    utility_packages = [
        'loguru>=0.7.0',
        'python-dotenv>=1.0.0',
        'schedule>=1.2.0',
        'tqdm>=4.66.0'
    ]
    
    # Security packages
    security_packages = [
        'cryptography>=41.0.0',
        'bcrypt>=4.0.0'
    ]
    
    all_packages = core_packages + ml_packages + web_packages + utility_packages + security_packages
    
    print("   Installing core packages...")
    for package in core_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Failed to install")
    
    print("\n   Installing ML packages...")
    for package in ml_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Not available (optional)")
    
    print("\n   Installing web framework packages...")
    for package in web_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Failed to install")
    
    print("\n   Installing utility packages...")
    for package in utility_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Failed to install")
    
    print("\n   Installing security packages...")
    for package in security_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Failed to install")

def create_project_structure():
    """Create project directory structure"""
    print("\n📁 Creating project structure...")
    
    directories = [
        'logs',
        'data',
        'config',
        'exports',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created directory: {directory}")

def create_configuration():
    """Create default configuration files"""
    print("\n⚙️  Creating configuration files...")
    
    # Create main configuration
    config = {
        "database_url": "sqlite:///idhe_health.db",
        "redis_url": "redis://localhost:6379/0",
        "update_interval": 30,
        "retention_days": 90,
        "enable_ml_optimization": True,
        "enable_predictive_maintenance": True,
        "enable_security_scanning": True,
        "performance_threshold": 0.8,
        "anomaly_sensitivity": 0.95,
        "log_level": "INFO",
        "max_concurrent_monitors": 4,
        "alert_cooldown_minutes": 15,
        "security_scan_interval": 3600,
        "max_failed_attempts": 10,
        "sql_injection_threshold": 0.7,
        "forecast_horizon_days": 30,
        "capacity_warning_threshold": 0.8,
        "auto_scaling_enabled": False,
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "dashboard_host": "0.0.0.0",
        "dashboard_port": 8050
    }
    
    with open('idhe_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("   ✅ Created idhe_config.json")
    
    # Create environment file
    env_content = """# IDHE Environment Configuration
# Database Configuration
IDHE_DATABASE_URL=sqlite:///idhe_health.db
IDHE_REDIS_URL=redis://localhost:6379/0

# API Configuration
IDHE_API_HOST=0.0.0.0
IDHE_API_PORT=8000
IDHE_DASHBOARD_HOST=0.0.0.0
IDHE_DASHBOARD_PORT=8050

# System Configuration
IDHE_UPDATE_INTERVAL=30
IDHE_RETENTION_DAYS=90
IDHE_LOG_LEVEL=INFO

# Feature Toggles
IDHE_ENABLE_ML_OPTIMIZATION=true
IDHE_ENABLE_PREDICTIVE_MAINTENANCE=true
IDHE_ENABLE_SECURITY_SCANNING=true

# Thresholds
IDHE_PERFORMANCE_THRESHOLD=0.8
IDHE_ANOMALY_SENSITIVITY=0.95
IDHE_CAPACITY_WARNING_THRESHOLD=0.8

# Security
IDHE_SECURITY_SCAN_INTERVAL=3600
IDHE_MAX_FAILED_ATTEMPTS=10
IDHE_SQL_INJECTION_THRESHOLD=0.7

# Capacity Planning
IDHE_FORECAST_HORIZON_DAYS=30
IDHE_AUTO_SCALING_ENABLED=false
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("   ✅ Created .env file")

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    print("\n🚀 Creating startup scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting IDHE - Intelligent Database Health Ecosystem...
echo.
echo Features:
echo • Machine Learning Query Optimization
echo • Advanced Anomaly Detection
echo • Predictive Performance Analysis
echo • AI-Powered Security Scanning
echo • Intelligent Capacity Planning
echo • Real-time Dashboard
echo.
echo Starting system...
python main_idhe.py
pause
"""
    
    with open('start_idhe.bat', 'w') as f:
        f.write(windows_script)
    print("   ✅ Created start_idhe.bat (Windows)")
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting IDHE - Intelligent Database Health Ecosystem..."
echo ""
echo "Features:"
echo "• Machine Learning Query Optimization"
echo "• Advanced Anomaly Detection"
echo "• Predictive Performance Analysis"
echo "• AI-Powered Security Scanning"
echo "• Intelligent Capacity Planning"
echo "• Real-time Dashboard"
echo ""
echo "Starting system..."
python main_idhe.py
"""
    
    with open('start_idhe.sh', 'w') as f:
        f.write(unix_script)
    
    # Make executable on Unix systems
    if platform.system() != 'Windows':
        os.chmod('start_idhe.sh', 0o755)
    print("   ✅ Created start_idhe.sh (Unix/Linux/macOS)")

def create_demo_data():
    """Create demo data for testing"""
    print("\n📊 Creating demo data...")
    
    demo_queries = [
        "SELECT * FROM users WHERE age > 30 AND status = 'active'",
        "SELECT COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL 1 DAY",
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
        "UPDATE products SET price = price * 1.1 WHERE category = 'electronics'",
        "DELETE FROM logs WHERE created_at < NOW() - INTERVAL 30 DAYS",
        "SELECT * FROM products WHERE name LIKE '%laptop%' AND price < 1000",
        "SELECT AVG(amount) FROM transactions WHERE status = 'completed'",
        "INSERT INTO audit_log (action, table_name, timestamp) VALUES ('UPDATE', 'users', NOW())"
    ]
    
    # Save demo queries
    with open('demo_queries.json', 'w') as f:
        json.dump(demo_queries, f, indent=2)
    print("   ✅ Created demo_queries.json")
    
    # Create sample metrics data
    import datetime
    sample_metrics = []
    for i in range(100):
        sample_metrics.append({
            "timestamp": (datetime.datetime.now() - datetime.timedelta(hours=i)).isoformat(),
            "cpu_percent": 45 + (i % 20) * 2,
            "memory_percent": 60 + (i % 15) * 1.5,
            "active_connections": 20 + (i % 8) * 3,
            "avg_query_time": 0.1 + (i % 10) * 0.02,
            "query_count_1min": 100 + (i % 5) * 20,
            "storage_used": 1000000 + i * 1000,
            "storage_total": 2000000,
            "error_rate": 0.001 + (i % 20) * 0.0001
        })
    
    with open('sample_metrics.json', 'w') as f:
        json.dump(sample_metrics, f, indent=2)
    print("   ✅ Created sample_metrics.json")

def validate_installation():
    """Validate the installation"""
    print("\n🔍 Validating installation...")
    
    # Check Python packages
    critical_packages = ['pandas', 'numpy', 'scikit-learn', 'psutil']
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Not installed")
            missing_packages.append(package)
    
    # Check project files
    required_files = [
        'main_idhe.py',
        'idhe_config.json',
        'requirements.txt'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Warning: {len(missing_packages)} critical packages are missing")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ Installation validation completed successfully!")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     INSTALLATION COMPLETE!                    ║
╚═══════════════════════════════════════════════════════════════╝

🚀 Next Steps:

1. Start the system:
   • Windows: Double-click start_idhe.bat
   • Linux/macOS: ./start_idhe.sh
   • Manual: python main_idhe.py

2. Access the dashboard:
   • Open your browser to: http://localhost:8050

3. Use the API:
   • API Documentation: http://localhost:8000/api/docs
   • Base URL: http://localhost:8000/api

4. Key Features to Explore:
   • Real-time Database Monitoring
   • ML Query Optimization
   • Anomaly Detection
   • Security Scanning
   • Capacity Planning
   • Performance Predictions

5. Configuration:
   • Edit idhe_config.json for settings
   • Modify .env for environment variables

6. Demo Data:
   • Sample queries in demo_queries.json
   • Historical metrics in sample_metrics.json

📚 Documentation:
   • README.md - Complete documentation
   • http://localhost:8000/api/docs - API reference

🆘 Troubleshooting:
   • Check logs/ directory for error logs
   • Verify Python 3.8+ and required packages
   • Check port availability (8000, 8050)

⚠️  Note: This is a unique, advanced system not available elsewhere.
    Features like ML query optimization and predictive analytics
    represent cutting-edge approaches to database management.
    """)

def main():
    """Main installation function"""
    print_banner()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    install_dependencies()
    
    # Create project structure
    create_project_structure()
    
    # Create configuration
    create_configuration()
    
    # Create startup scripts
    create_startup_scripts()
    
    # Create demo data
    create_demo_data()
    
    # Validate installation
    if validate_installation():
        print_next_steps()
    else:
        print("\n❌ Installation completed with warnings. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()