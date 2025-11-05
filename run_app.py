"""
Launcher script for FinLlama Sentiment Analysis App
Ensures PyTorch loads BEFORE Streamlit to avoid DLL conflicts on Windows
"""

import sys
import os

print("="*80)
print("FinLlama: Advanced Financial Sentiment Analysis")
print("="*80)

# Step 1: Preload PyTorch to initialize DLLs before Streamlit
print("\n[1/3] Loading PyTorch and FinBERT dependencies...")
try:
    import torch
    print(f"   PyTorch {torch.__version__} loaded successfully")
    _torch_loaded = True
except Exception as e:
    print(f"   Warning: PyTorch failed to load: {e}")
    _torch_loaded = False

# Step 2: Import transformers to ensure all dependencies are ready
if _torch_loaded:
    try:
        import transformers
        print(f"   Transformers {transformers.__version__} loaded successfully")
    except Exception as e:
        print(f"   Warning: Transformers failed to load: {e}")

# Step 3: Now it's safe to run Streamlit
print("\n[2/3] Starting Streamlit application...")
print("   Please wait while the UI loads...")

# Import and run Streamlit
from streamlit.web import cli as stcli

if __name__ == '__main__':
    # Set the main app file
    sys.argv = ["streamlit", "run", "main_app.py"]

    # Launch Streamlit
    print("\n[3/3] Launching FinLlama interface...")
    print("="*80)
    print("\nThe app should open in your browser automatically.")
    print("If not, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    print("="*80)

    sys.exit(stcli.main())
