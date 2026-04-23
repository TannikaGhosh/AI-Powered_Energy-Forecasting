"""
Run this once to generate data, preprocess, train, and start API (optional).
"""
import subprocess
import os

def run_script(script_path):
    print(f"\n🚀 Running {script_path} ...")
    result = subprocess.run(['python', os.path.basename(script_path)], capture_output=True, text=True, cwd=os.path.dirname(script_path))
    if result.returncode != 0:
        print(f"❌ Error in {script_path}:\n{result.stderr}")
        return False
    else:
        print(result.stdout)
        return True

if __name__ == '__main__':
    # Create necessary folders
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Generate synthetic data
    if not run_script('src/data_generation.py'):
        exit(1)
    
    # Step 2: Preprocess
    if not run_script('src/preprocess.py'):
        exit(1)
    
    # Step 3: Train model
    if not run_script('src/train_model.py'):
        exit(1)
    
    print("\n✅ Full pipeline completed successfully!")
    print("To start the prediction API, run: python app.py")
