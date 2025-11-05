# How to Run FinLlama Sentiment Analysis App

## Quick Start (RECOMMENDED)

### Windows:
**Double-click `run_app.bat`**

OR from command line:
```bash
python run_app.py
```

### Mac/Linux:
```bash
python run_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## Why Use run_app.py Instead of Streamlit Directly?

**IMPORTANT**: Due to Windows DLL loading conflicts between PyTorch and Streamlit, you **MUST** use `run_app.py` or `run_app.bat` to launch the application.

### What Happens If You Run Directly:
```bash
streamlit run main_app.py  # ❌ DON'T DO THIS
```

**Result**: You'll get `WinError 1114: DLL initialization failed` and FinBERT will not load.

### Correct Way:
```bash
python run_app.py  # ✅ DO THIS
```
OR
```bash
run_app.bat  # ✅ DO THIS (Windows)
```

**Result**: PyTorch loads first, then Streamlit, and FinBERT works perfectly!

---

##  What the Launcher Does

`run_app.py` performs these steps:

1. **Preloads PyTorch** - Ensures DLLs are initialized before Streamlit
2. **Preloads Transformers** - Loads FinBERT dependencies
3. **Launches Streamlit** - Starts the web interface programmatically

This ensures proper DLL loading order and prevents conflicts.

---

## Verifying FinBERT is Working

When the app starts, you should see in the UI:
- ✅ **"FinBERT loaded successfully"** (green banner)
- ✅ **"NER Entity Recognition loaded"** (green banner)
- ✅ **"LightGBM Meta-Learner loaded"** (green banner)
- ✅ **"Confidence Learner loaded"** (green banner)

If you see:
- ℹ️ "FinBERT not available" (blue banner) - The launcher didn't work correctly

---

## Troubleshooting

### Problem: "Module not found" errors

**Solution**:
```bash
pip install -r requirements_modern.txt
```

### Problem: Still getting DLL errors

**Solution**:
1. Make sure you're using `run_app.py`, NOT `streamlit run main_app.py`
2. Reinstall PyTorch:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch --no-cache-dir
   ```

### Problem: App doesn't open in browser

**Solution**:
Manually navigate to: `http://localhost:8501`

### Problem: Port 8501 already in use

**Solution**:
```bash
# Kill existing Streamlit processes
taskkill /F /IM streamlit.exe  # Windows
pkill -9 streamlit  # Mac/Linux

# Or use a different port
python run_app.py --server.port 8502
```

---

## For Developers

### Running Tests (No UI needed)
```bash
# Test FinBERT installation
python verify_finbert_working.py

# Test confidence scores
python test_enhanced_confidence.py

# Test FinBERT standalone
python test_finbert.py
```

### Using FinLlamaAnalyzer Programmatically
```python
from main_app import FinLlamaAnalyzer

# Initialize (API key from .env file)
analyzer = FinLlamaAnalyzer(api_key="your_api_key_here")

# Analyze text
result = analyzer.finllama_ensemble_analysis("Apple reported strong earnings")

# Check results
print(f"Sentiment: {result['ensemble_score']}")
print(f"Confidence: {result['confidence']}")
print(f"FinBERT Score: {result['components']['finbert']}")  # Should be non-zero!
```

---

## System Requirements

- **Python**: 3.10+ (tested on 3.13)
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 2GB for models (first run downloads ~500MB)
- **Internet**: Required for first-time model download

---

## First Run

On first run, the app will download:
- **FinBERT model** (~440MB) from Hugging Face
- Cached in: `C:\Users\<username>\.cache\huggingface\`
- Takes 5-15 minutes depending on internet speed
- Subsequent runs load instantly from cache

---

## Files Overview

| File | Purpose |
|------|---------|
| `run_app.py` | **Main launcher** - Use this to start the app |
| `run_app.bat` | Windows shortcut for `run_app.py` |
| `main_app.py` | Core Streamlit application |
| `confidence_learner.py` | Learned confidence model |
| `lightgbm_meta_learner.py` | Sentiment ensemble model |
| `ner_processor.py` | Named entity recognition |
| `verify_finbert_working.py` | Test script to verify FinBERT |

---

## Success Indicators

When everything is working correctly:

1. **Console Output**:
   ```
   ================================================================================
   FinLlama: Advanced Financial Sentiment Analysis
   ================================================================================

   [1/3] Loading PyTorch and FinBERT dependencies...
      PyTorch 2.9.0+cpu loaded successfully
      Transformers 4.57.1 loaded successfully

   [2/3] Starting Streamlit application...
      Please wait while the UI loads...

   [3/3] Launching FinLlama interface...
   ```

2. **Browser UI**:
   - Green success messages for FinBERT, NER, Meta-Learner, Confidence Learner
   - Analysis produces non-zero FinBERT scores
   - Confidence scores in 0.4-0.9 range (not all zeros)

3. **Test Script**:
   ```bash
   python verify_finbert_working.py
   ```
   Should show:
   ```
   FinBERT Available:            YES [OK]
   FinBERT Score:      +0.9526  [WORKING!]
   ```

---

## Additional Help

- **Documentation**: See `FINBERT_FIX_SUMMARY.md` for technical details
- **Issues**: Check GitHub issues or create new one
- **API Keys**: Configure in `.env` file (see `.env.example`)

---

**Remember**: Always use `run_app.py` or `run_app.bat` to launch the app!
