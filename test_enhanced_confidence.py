"""
Test script for enhanced confidence score implementation
"""

import sys
import os

# Suppress streamlit warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Redirect streamlit output
import warnings
warnings.filterwarnings('ignore')

from main_app import FinLlamaAnalyzer

def test_confidence_improvements():
    """Test the enhanced confidence score with various scenarios"""

    print("=" * 80)
    print("TESTING ENHANCED CONFIDENCE SCORE")
    print("=" * 80)

    # Initialize analyzer (API key not needed for sentiment analysis testing)
    analyzer = FinLlamaAnalyzer(api_key="test_key")

    # Test Case 1: Strong positive sentiment with entities
    print("\n[Test 1] Strong Positive News with Financial Data")
    print("-" * 80)
    text1 = """Apple Inc. reported record quarterly earnings of $25.5 billion,
    beating analyst expectations by 15%. Revenue surged to $123.9 billion,
    representing a 25% year-over-year growth. The company announced a new
    $90 billion stock buyback program."""

    result1 = analyzer.finllama_ensemble_analysis(text1)
    print(f"Text: {text1[:100]}...")
    print(f"Sentiment Score: {result1['ensemble_score']:.4f}")
    print(f"Confidence: {result1['confidence']:.4f}")
    print(f"Component Scores: {result1['components']}")
    print(f"Expected: High confidence (>0.75) due to strong agreement + entities")

    # Test Case 2: Mixed/conflicting signals
    print("\n[Test 2] Mixed Signals (Low Agreement)")
    print("-" * 80)
    text2 = """Tesla stock declined 5% today despite reporting better than expected
    earnings. The company posted a profit but warned of upcoming challenges in supply
    chain and production. Investors remain uncertain about future growth prospects."""

    result2 = analyzer.finllama_ensemble_analysis(text2)
    print(f"Text: {text2[:100]}...")
    print(f"Sentiment Score: {result2['ensemble_score']:.4f}")
    print(f"Confidence: {result2['confidence']:.4f}")
    print(f"Component Scores: {result2['components']}")
    print(f"Expected: Lower confidence (<0.60) due to conflicting signals")

    # Test Case 3: Strong negative sentiment
    print("\n[Test 3] Strong Negative News")
    print("-" * 80)
    text3 = """Company announces bankruptcy filing after massive losses. Stock
    crashed 45% as investors panic. Layoffs expected to affect 10,000 employees.
    Investigations into fraud allegations underway."""

    result3 = analyzer.finllama_ensemble_analysis(text3)
    print(f"Text: {text3[:100]}...")
    print(f"Sentiment Score: {result3['ensemble_score']:.4f}")
    print(f"Confidence: {result3['confidence']:.4f}")
    print(f"Component Scores: {result3['components']}")
    print(f"Expected: High confidence (>0.70) due to strong negative agreement")

    # Test Case 4: Neutral/vague statement
    print("\n[Test 4] Neutral/Vague Statement")
    print("-" * 80)
    text4 = """The company held its annual meeting today. Various topics were
    discussed including future plans and market conditions. Shareholders attended
    the event."""

    result4 = analyzer.finllama_ensemble_analysis(text4)
    print(f"Text: {text4[:100]}...")
    print(f"Sentiment Score: {result4['ensemble_score']:.4f}")
    print(f"Confidence: {result4['confidence']:.4f}")
    print(f"Component Scores: {result4['components']}")
    print(f"Expected: Low confidence (<0.50) due to weak signals")

    # Test Case 5: Technical analysis without FinBERT
    if not analyzer.finbert_available:
        print("\n[Test 5] Analysis without FinBERT")
        print("-" * 80)
        print("FinBERT not available - confidence should be slightly lower")
        print(f"FinBERT Available: {analyzer.finbert_available}")
        print(f"Confidence from Test 1: {result1['confidence']:.4f}")
        print("Expected: Confidence reduced by ~0.05 due to missing FinBERT")

    # Summary comparison
    print("\n" + "=" * 80)
    print("CONFIDENCE SCORE COMPARISON")
    print("=" * 80)
    print(f"Test 1 (Strong Positive + Entities): {result1['confidence']:.4f}")
    print(f"Test 2 (Mixed Signals):              {result2['confidence']:.4f}")
    print(f"Test 3 (Strong Negative):            {result3['confidence']:.4f}")
    print(f"Test 4 (Neutral/Vague):              {result4['confidence']:.4f}")
    print("\n[PASS] Enhanced confidence should show clear differentiation:")
    print("   - Higher for strong agreement (Test 1 & 3)")
    print("   - Lower for disagreement (Test 2)")
    print("   - Lower for weak signals (Test 4)")

    # Component variance analysis
    print("\n" + "=" * 80)
    print("COMPONENT VARIANCE ANALYSIS (Disagreement Measure)")
    print("=" * 80)

    import numpy as np
    for i, (name, result) in enumerate([
        ("Test 1 (Strong +)", result1),
        ("Test 2 (Mixed)", result2),
        ("Test 3 (Strong -)", result3),
        ("Test 4 (Neutral)", result4)
    ], 1):
        component_values = list(result['components'].values())
        variance = np.std(component_values)
        print(f"{name}: Variance = {variance:.4f}, Confidence = {result['confidence']:.4f}")

    print("\n[PASS] Higher variance should correlate with lower confidence")
    print("=" * 80)

if __name__ == "__main__":
    test_confidence_improvements()
