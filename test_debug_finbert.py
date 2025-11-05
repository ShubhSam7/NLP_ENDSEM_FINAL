"""
Debug FinBERT loading to see exact error
"""

import sys
import os
import traceback

# Suppress streamlit
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Mock streamlit to capture messages
class MockStreamlit:
    def success(self, msg):
        print(f"[SUCCESS] {msg}")

    def info(self, msg):
        print(f"[INFO] {msg}")

sys.modules['streamlit'] = MockStreamlit()
import streamlit as st

print("="*80)
print("DEBUGGING FINBERT LOADING")
print("="*80)

print("\nAttempting to load FinBERT...")
finbert_available = False
finbert_pipeline = None

try:
    print("  Step 1: Importing transformers...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    print("  [OK] Imports successful")

    print("  Step 2: Loading tokenizer...")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    print("  [OK] Tokenizer loaded")

    print("  Step 3: Loading model...")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("  [OK] Model loaded")

    print("  Step 4: Creating pipeline...")
    finbert_pipeline = pipeline("sentiment-analysis",
                               model=finbert_model,
                               tokenizer=finbert_tokenizer)
    print("  [OK] Pipeline created")

    finbert_available = True
    st.success("FinBERT loaded successfully")

except Exception as e:
    print(f"\n[ERROR] FinBERT loading failed!")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    st.info(f"FinBERT not available, using enhanced TextBlob: {str(e)}")
    finbert_pipeline = None

print("\n" + "="*80)
print(f"Final status: finbert_available = {finbert_available}")
print("="*80)
