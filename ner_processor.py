"""
NER (Named Entity Recognition) Processor for Financial Text Analysis
Extracts and processes financial entities from news articles
"""

import spacy
from typing import Dict, List, Tuple
import re
from collections import Counter

class NERFinancialProcessor:
    """Process financial text with Named Entity Recognition"""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NER processor with spaCy model

        Args:
            model_name: spaCy model to load (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading {model_name}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

        # Financial entity patterns for custom recognition
        self.financial_patterns = self._setup_financial_patterns()
        self._add_financial_entity_rules()

    def _setup_financial_patterns(self) -> Dict[str, List[str]]:
        """Setup financial entity patterns"""
        return {
            'financial_metrics': [
                r'\$\d+\.?\d*[BMK]?',  # Money amounts: $5M, $1.2B
                r'\d+\.?\d*%',          # Percentages: 15%, 3.5%
                r'Q[1-4]\s*\d{4}',      # Quarters: Q1 2024
                r'FY\d{4}',             # Fiscal years: FY2024
                r'\d+\s*bps',           # Basis points: 50 bps
            ],
            'financial_terms': [
                'earnings', 'revenue', 'profit', 'loss', 'EBITDA', 'EPS',
                'market cap', 'valuation', 'IPO', 'merger', 'acquisition',
                'dividend', 'buyback', 'guidance', 'forecast', 'outlook'
            ],
            'market_indicators': [
                'bull market', 'bear market', 'correction', 'rally',
                'volatility', 'momentum', 'resistance', 'support',
                'breakout', 'breakdown', 'trend', 'reversal'
            ]
        }

    def _add_financial_entity_rules(self):
        """Add custom entity recognition rules for financial terms"""
        if 'entity_ruler' not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        # Add patterns for financial terms
        patterns = []
        for term in self.financial_patterns['financial_terms']:
            patterns.append({"label": "FINANCIAL_TERM", "pattern": term})

        for indicator in self.financial_patterns['market_indicators']:
            patterns.append({"label": "MARKET_INDICATOR", "pattern": indicator})

        ruler.add_patterns(patterns)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text

        Args:
            text: Input text to process

        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        doc = self.nlp(text)

        entities = {
            'companies': [],
            'people': [],
            'locations': [],
            'money': [],
            'dates': [],
            'percentages': [],
            'financial_terms': [],
            'market_indicators': [],
            'products': []
        }

        # Extract standard entities
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['companies'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['people'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['money'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'PERCENT':
                entities['percentages'].append(ent.text)
            elif ent.label_ == 'FINANCIAL_TERM':
                entities['financial_terms'].append(ent.text)
            elif ent.label_ == 'MARKET_INDICATOR':
                entities['market_indicators'].append(ent.text)
            elif ent.label_ == 'PRODUCT':
                entities['products'].append(ent.text)

        # Extract financial metrics using regex
        for pattern in self.financial_patterns['financial_metrics']:
            matches = re.findall(pattern, text)
            if '$' in pattern:
                entities['money'].extend(matches)
            elif '%' in pattern:
                entities['percentages'].extend(matches)

        # Remove duplicates while preserving order
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return entities

    def enrich_text_with_entities(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Enrich text analysis with entity information

        Args:
            text: Original text
            entities: Extracted entities

        Returns:
            Dictionary with enriched information
        """
        # Count entity mentions
        entity_counts = {key: len(val) for key, val in entities.items() if val}

        # Calculate entity density (entities per 100 words)
        words = text.split()
        total_entities = sum(entity_counts.values())
        entity_density = (total_entities / len(words) * 100) if words else 0

        # Identify primary company (most mentioned organization)
        primary_company = None
        if entities['companies']:
            company_counter = Counter(entities['companies'])
            primary_company = company_counter.most_common(1)[0][0]

        # Check for financial metrics presence
        has_financial_data = bool(entities['money'] or entities['percentages'])

        return {
            'entity_counts': entity_counts,
            'total_entities': total_entities,
            'entity_density': round(entity_density, 2),
            'primary_company': primary_company,
            'has_financial_data': has_financial_data,
            'key_people': entities['people'][:3],  # Top 3 people mentioned
            'key_locations': entities['locations'][:3]  # Top 3 locations
        }

    def analyze_entity_sentiment_context(self, text: str, target_entity: str) -> Dict[str, any]:
        """
        Analyze sentiment context around a specific entity

        Args:
            text: Input text
            target_entity: Entity to analyze (e.g., company name)

        Returns:
            Dictionary with entity-specific sentiment context
        """
        doc = self.nlp(text)

        # Find sentences containing the target entity
        entity_sentences = []
        for sent in doc.sents:
            if target_entity.lower() in sent.text.lower():
                entity_sentences.append(sent.text)

        # Extract verbs and adjectives near the entity
        entity_descriptors = {
            'verbs': [],
            'adjectives': [],
            'context_words': []
        }

        for token in doc:
            if target_entity.lower() in token.text.lower():
                # Get surrounding words (window of 5 words)
                start = max(0, token.i - 5)
                end = min(len(doc), token.i + 6)

                for i in range(start, end):
                    if doc[i].pos_ == 'VERB':
                        entity_descriptors['verbs'].append(doc[i].text)
                    elif doc[i].pos_ == 'ADJ':
                        entity_descriptors['adjectives'].append(doc[i].text)
                    elif doc[i].pos_ in ['NOUN', 'ADV']:
                        entity_descriptors['context_words'].append(doc[i].text)

        return {
            'entity': target_entity,
            'mentions': len(entity_sentences),
            'sentences': entity_sentences,
            'descriptors': entity_descriptors,
            'prominence': len(entity_sentences) / len(list(doc.sents)) if list(doc.sents) else 0
        }

    def extract_financial_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between financial entities

        Args:
            text: Input text

        Returns:
            List of tuples (subject, relation, object)
        """
        doc = self.nlp(text)
        relationships = []

        # Define financial relation verbs
        financial_verbs = {
            'acquire', 'acquires', 'acquired', 'merge', 'merges', 'merged',
            'invest', 'invests', 'invested', 'partner', 'partners', 'partnered',
            'buy', 'buys', 'bought', 'sell', 'sells', 'sold',
            'report', 'reports', 'reported', 'announce', 'announces', 'announced',
            'increase', 'increases', 'increased', 'decrease', 'decreases', 'decreased'
        }

        for token in doc:
            if token.lemma_ in financial_verbs:
                # Find subject (typically a company)
                subject = None
                for child in token.lefts:
                    if child.dep_ in ['nsubj', 'nsubjpass'] and child.ent_type_ == 'ORG':
                        subject = child.text
                        break

                # Find object (what the action is applied to)
                obj = None
                for child in token.rights:
                    if child.dep_ in ['dobj', 'pobj'] and child.ent_type_ in ['ORG', 'MONEY', 'PERCENT']:
                        obj = child.text
                        break

                if subject and obj:
                    relationships.append((subject, token.lemma_, obj))

        return relationships

    def get_entity_enhanced_keywords(self, text: str) -> List[str]:
        """
        Extract entity-enhanced keywords for better sentiment analysis

        Args:
            text: Input text

        Returns:
            List of important keywords with entity context
        """
        doc = self.nlp(text)
        keywords = []

        # Extract noun chunks with entities
        for chunk in doc.noun_chunks:
            # Keep chunks that contain entities or financial terms
            if any(token.ent_type_ in ['ORG', 'PERSON', 'MONEY', 'PERCENT'] for token in chunk):
                keywords.append(chunk.text)
            # Keep chunks with adjectives (sentiment indicators)
            elif any(token.pos_ == 'ADJ' for token in chunk):
                keywords.append(chunk.text)

        # Add financial terms
        entities = self.extract_entities(text)
        keywords.extend(entities.get('financial_terms', []))
        keywords.extend(entities.get('market_indicators', []))

        return list(dict.fromkeys(keywords))  # Remove duplicates

    def preprocess_for_sentiment(self, text: str, target_company: str = None) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for sentiment analysis

        Args:
            text: Input text
            target_company: Optional company name to focus analysis

        Returns:
            Comprehensive preprocessing results
        """
        # Extract all entities
        entities = self.extract_entities(text)

        # Enrich with entity analysis
        enrichment = self.enrich_text_with_entities(text, entities)

        # Extract keywords
        keywords = self.get_entity_enhanced_keywords(text)

        # Extract relationships
        relationships = self.extract_financial_relationships(text)

        # Company-specific analysis if target provided
        company_context = None
        if target_company:
            company_context = self.analyze_entity_sentiment_context(text, target_company)

        return {
            'original_text': text,
            'entities': entities,
            'enrichment': enrichment,
            'keywords': keywords,
            'relationships': relationships,
            'company_context': company_context,
            'processed_text': text  # Keep original for now, can be modified if needed
        }


# Cache for NER processor instance
_ner_processor_cache = None

def get_ner_processor() -> NERFinancialProcessor:
    """Get cached NER processor instance"""
    global _ner_processor_cache
    if _ner_processor_cache is None:
        _ner_processor_cache = NERFinancialProcessor()
    return _ner_processor_cache
