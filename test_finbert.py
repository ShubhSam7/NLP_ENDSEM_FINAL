"""
Test FinBERT loading and inference
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print("="*80)
print("TESTING FINBERT INSTALLATION")
print("="*80)

# Load tokenizer
print("\n1. Loading FinBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
print("   [OK] Tokenizer loaded")

# Load model
print("\n2. Loading FinBERT model...")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("   [OK] Model loaded")

# Create pipeline
print("\n3. Creating sentiment analysis pipeline...")
finbert_pipeline = pipeline("sentiment-analysis",
                           model=model,
                           tokenizer=tokenizer)
print("   [OK] Pipeline created")

# Test inference
print("\n4. Testing inference...")
test_texts = [
    "The company reported record profits and strong growth.",
    "Stocks crashed amid recession fears.",
    "The meeting was held today."
]

for text in test_texts:
    result = finbert_pipeline(text)[0]
    label = result['label']
    score = result['score']

    # Convert to sentiment score (-1 to +1)
    if label == 'positive':
        sentiment = score
    elif label == 'negative':
        sentiment = -score
    else:
        sentiment = 0.0

    print(f"\n   Text: {text}")
    print(f"   Label: {label}, Confidence: {score:.4f}, Sentiment: {sentiment:+.4f}")

print("\n" + "="*80)
print("FINBERT IS WORKING CORRECTLY!")
print("="*80)
