import sys
import subprocess
import os

def install_dependencies():
    print("====================================================")
    print(" Installing Used Car Price Prediction Dependencies ")
    print("====================================================")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} file not found.")
        sys.exit(1)
        
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    print(f"Installing all dependencies from {requirements_file}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    
    print("\n====================================================")
    print(" All dependencies installed successfully!")
    print(" To start the server, run:")
    print("   uvicorn app.main:app --reload")
    print("====================================================")

if __name__ == "__main__":
    install_dependencies()
