#!/usr/bin/env python3
"""
IDHE Quick Start Script
======================

Run this script to start the comprehensive IDHE demo
"""

import subprocess
import sys
import os

def print_header():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   IDHE - Intelligent Database Health Ecosystem               ║
    ║                                                               ║
    ║   🧠 Revolutionary SQL-Python Database System                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝

    🎯 QUICK START OPTIONS:
    1. Run Comprehensive Demo (Recommended)
    2. Install Dependencies
    3. Start Full System
    4. View Documentation
    """)

def run_demo():
    """Run the comprehensive IDHE demo"""
    print("\n🚀 Running IDHE Comprehensive Demo...")
    print("   This will showcase all revolutionary features!")
    
    try:
        # Run the demo
        result = subprocess.run([sys.executable, "demo_idhe.py"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("❌ Demo failed. Trying to install dependencies...")
        install_dependencies()
        return run_demo()

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing IDHE dependencies...")
    
    # Core packages
    packages = [
        "pandas", "numpy", "scipy", "scikit-learn", "psutil",
        "loguru", "python-dotenv", "schedule", "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Failed")
    
    # Try optional packages
    optional_packages = [
        "fastapi", "uvicorn[standard]", "dash", "plotly", 
        "dash-bootstrap-components", "tensorflow", "statsmodels"
    ]
    
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - Optional (not available)")

def start_full_system():
    """Start the full IDHE system"""
    print("\n🖥️  Starting IDHE Full System...")
    
    try:
        result = subprocess.run([sys.executable, "main_idhe.py"], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("❌ System failed to start")
        return False

def show_docs():
    """Show documentation"""
    print("\n📚 IDHE Documentation:")
    print("   • README.md - Complete system documentation")
    print("   • Core Components:")
    print("     - ML Optimizer: Neural networks + Genetic algorithms")
    print("     - Anomaly Detector: LSTM + Statistical methods")
    print("     - Security Scanner: AI-powered threat detection")
    print("     - Capacity Planner: Mathematical optimization")
    print("     - Performance Predictor: Time series forecasting")
    
    # Try to show README
    try:
        if os.path.exists("README.md"):
            print("\n   Opening README.md...")
            if sys.platform.startswith('win'):
                os.startfile("README.md")
            elif sys.platform.startswith('darwin'):
                subprocess.run(["open", "README.md"])
            else:
                subprocess.run(["xdg-open", "README.md"])
    except:
        print("   Could not open README.md automatically")

def main():
    """Main quick start function"""
    print_header()
    
    print("🎯 Choose an option:")
    print("   1. Demo (Comprehensive feature showcase)")
    print("   2. Install dependencies only")
    print("   3. Start full system")
    print("   4. Show documentation")
    print("   5. Exit")
    
    while True:
        try:
            choice = input("\n   Enter choice (1-5): ").strip()
            
            if choice == "1":
                success = run_demo()
                if success:
                    print("\n🎉 Demo completed successfully!")
                else:
                    print("\n❌ Demo encountered issues")
                break
                
            elif choice == "2":
                install_dependencies()
                print("\n✅ Dependencies installed!")
                break
                
            elif choice == "3":
                success = start_full_system()
                if success:
                    print("\n🚀 IDHE system started!")
                else:
                    print("\n❌ System failed to start")
                break
                
            elif choice == "4":
                show_docs()
                break
                
            elif choice == "5":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("   Please enter 1-5")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()