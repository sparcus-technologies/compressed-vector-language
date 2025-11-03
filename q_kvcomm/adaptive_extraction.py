"""
Adaptive information extraction for Q-KVComm
Implements multiple extraction strategies for efficient agent communication
"""

import torch
import numpy as np
import re
import spacy
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("Warning: yake not installed. Install with: pip install yake")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False
    try:
        import spacy.cli
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except:
        print("Warning: spaCy not available. Some extraction features disabled.")
        nlp = None


@dataclass
class ExtractedFact:
    """Represents an extracted fact with metadata"""
    fact_type: str  # 'entity', 'numeric', 'relation', 'api_pattern'
    content: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'type': self.fact_type,
            'content': self.content,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class InformationExtractor:
    """
    Multi-strategy information extraction
    Handles entities, relations, numeric facts, and domain-specific patterns
    """
    
    def __init__(self, extraction_method: str = "yake"):
        """
        Args:
            extraction_method: One of 'yake', 'spacy_ner', 'hybrid', 'simple'
        """
        self.method = extraction_method
        
        # Initialize extractors based on method
        if extraction_method == "yake" and YAKE_AVAILABLE:
            self.yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # max n-gram size
                dedupLim=0.9,
                top=20,
                features=None
            )
        else:
            self.yake_extractor = None
        
        if extraction_method in ["spacy_ner", "hybrid"] and SPACY_AVAILABLE:
            self.nlp = nlp
        else:
            self.nlp = None
        
        # Compiled regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for fast extraction"""
        # Numeric patterns
        self.number_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)?\b')
        
        # API patterns
        self.header_pattern = re.compile(r'([A-Z][\w-]+)\s+header', re.IGNORECASE)
        self.endpoint_pattern = re.compile(r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-{}]+)')
        self.rate_limit_pattern = re.compile(r'(\d+)\s+requests?\s+(?:per|/)\s+(\w+)')
        self.url_pattern = re.compile(r'https?://[\w\-\.]+(?:/[\w\-\./?%&=]*)?')
        
        # Tech specs patterns
        self.version_pattern = re.compile(r'v?\d+\.\d+(?:\.\d+)?')
        self.price_pattern = re.compile(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)')
        self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%')
        
        # Date patterns
        self.date_pattern = re.compile(
            r'\b(?:Q[1-4]\s+)?\d{4}\b|'
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        )
    
    def extract_facts(self, context: str) -> List[ExtractedFact]:
        """
        Main extraction entry point
        
        Args:
            context: Text to extract facts from
            
        Returns:
            List of ExtractedFact objects
        """
        if self.method == "yake":
            return self._extract_yake(context)
        elif self.method == "spacy_ner":
            return self._extract_spacy_ner(context)
        elif self.method == "hybrid":
            return self._extract_hybrid(context)
        elif self.method == "simple":
            return self._extract_simple(context)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _extract_yake(self, context: str) -> List[ExtractedFact]:
        """YAKE-based extraction (RECOMMENDED)"""
        if not self.yake_extractor:
            return self._extract_simple(context)
        
        facts = []
        
        # Extract keyphrases
        keywords = self.yake_extractor.extract_keywords(context)
        
        for keyword, score in keywords:
            # YAKE scores are lower = better, invert for confidence
            confidence = 1.0 / (1.0 + score)
            
            facts.append(ExtractedFact(
                fact_type='keyphrase',
                content=keyword,
                confidence=confidence,
                metadata={'yake_score': score}
            ))
        
        # Enhance with numeric facts
        facts.extend(self._extract_numeric_facts(context))
        
        # Enhance with API patterns
        facts.extend(self._extract_api_patterns(context))
        
        return facts
    
    def _extract_spacy_ner(self, context: str) -> List[ExtractedFact]:
        """SpaCy NER-based extraction"""
        if not self.nlp:
            return self._extract_simple(context)
        
        facts = []
        doc = self.nlp(context)
        
        # Extract named entities
        for ent in doc.ents:
            facts.append(ExtractedFact(
                fact_type='entity',
                content=ent.text,
                confidence=0.9,  # SpaCy is generally reliable
                metadata={
                    'entity_type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
            ))
        
        # Extract numeric facts with context
        facts.extend(self._extract_numeric_facts_spacy(doc))
        
        # Extract noun chunks (important concepts)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word concepts
                facts.append(ExtractedFact(
                    fact_type='concept',
                    content=chunk.text,
                    confidence=0.7,
                    metadata={'pos': 'noun_chunk'}
                ))
        
        return facts
    
    def _extract_hybrid(self, context: str) -> List[ExtractedFact]:
        """
        Hybrid approach: Combines YAKE + SpaCy + domain patterns
        BEST ACCURACY
        """
        facts = []
        seen_content = set()  # Deduplication
        
        # 1. YAKE keyphrases
        if self.yake_extractor:
            yake_facts = self._extract_yake(context)
            for fact in yake_facts:
                if fact.content.lower() not in seen_content:
                    facts.append(fact)
                    seen_content.add(fact.content.lower())
        
        # 2. SpaCy entities
        if self.nlp:
            spacy_facts = self._extract_spacy_ner(context)
            for fact in spacy_facts:
                if fact.content.lower() not in seen_content:
                    facts.append(fact)
                    seen_content.add(fact.content.lower())
        
        # 3. Domain-specific patterns (always applied)
        domain_facts = self._extract_api_patterns(context)
        for fact in domain_facts:
            if fact.content.lower() not in seen_content:
                facts.append(fact)
                seen_content.add(fact.content.lower())
        
        # 4. Relations (if SpaCy available)
        if self.nlp:
            relation_facts = self._extract_relations(context)
            for fact in relation_facts:
                # Relations are unique, don't check duplication
                facts.append(fact)
        
        return facts
    
    def _extract_simple(self, context: str) -> List[ExtractedFact]:
        """Simple regex-based extraction (BASELINE)"""
        facts = []
        
        # Extract numbers with units
        facts.extend(self._extract_numeric_facts(context))
        
        # Extract API patterns
        facts.extend(self._extract_api_patterns(context))
        
        # Extract capitalized terms (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context)
        for term in set(capitalized):
            if len(term) > 3:  # Filter out short acronyms
                facts.append(ExtractedFact(
                    fact_type='proper_noun',
                    content=term,
                    confidence=0.5,
                    metadata={'method': 'regex'}
                ))
        
        return facts
    
    def _extract_numeric_facts(self, context: str) -> List[ExtractedFact]:
        """Extract numeric facts with context"""
        facts = []
        
        # Numbers with units
        for match in self.number_pattern.finditer(context):
            number, unit = match.groups()
            unit = unit or ''
            
            # Get surrounding context
            start = max(0, match.start() - 30)
            end = min(len(context), match.end() + 30)
            context_window = context[start:end]
            
            facts.append(ExtractedFact(
                fact_type='numeric',
                content=f"{number} {unit}".strip(),
                confidence=0.9,
                metadata={
                    'number': number,
                    'unit': unit,
                    'context': context_window
                }
            ))
        
        # Prices
        for match in self.price_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='price',
                content=match.group(0),
                confidence=0.95,
                metadata={'amount': match.group(1)}
            ))
        
        # Percentages
        for match in self.percentage_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='percentage',
                content=match.group(0),
                confidence=0.9,
                metadata={'value': match.group(1)}
            ))
        
        return facts
    
    def _extract_numeric_facts_spacy(self, doc) -> List[ExtractedFact]:
        """Extract numeric facts using SpaCy"""
        facts = []
        
        for token in doc:
            if token.like_num or token.pos_ == "NUM":
                # Get entity context
                entity_text = None
                for ent in doc.ents:
                    if ent.start <= token.i <= ent.end:
                        entity_text = ent.text
                        break
                
                # Get unit (next token if it's a noun)
                unit = ''
                if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ['NOUN', 'PROPN']:
                    unit = doc[token.i + 1].text
                
                facts.append(ExtractedFact(
                    fact_type='numeric',
                    content=f"{token.text} {unit}".strip(),
                    confidence=0.9,
                    metadata={
                        'entity': entity_text,
                        'unit': unit
                    }
                ))
        
        return facts
    
    def _extract_api_patterns(self, context: str) -> List[ExtractedFact]:
        """Extract API-specific information"""
        facts = []
        
        # Headers
        for match in self.header_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='api_header',
                content=match.group(1),
                confidence=0.95,
                metadata={'pattern': 'header'}
            ))
        
        # Endpoints
        for match in self.endpoint_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='api_endpoint',
                content=match.group(0),
                confidence=0.95,
                metadata={'path': match.group(1)}
            ))
        
        # Rate limits
        for match in self.rate_limit_pattern.finditer(context):
            value, period = match.groups()
            facts.append(ExtractedFact(
                fact_type='rate_limit',
                content=f"{value} requests/{period}",
                confidence=0.95,
                metadata={'value': value, 'period': period}
            ))
        
        # URLs
        for match in self.url_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='url',
                content=match.group(0),
                confidence=0.95,
                metadata={'pattern': 'url'}
            ))
        
        # Versions
        for match in self.version_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='version',
                content=match.group(0),
                confidence=0.85,
                metadata={'pattern': 'version'}
            ))
        
        # Dates
        for match in self.date_pattern.finditer(context):
            facts.append(ExtractedFact(
                fact_type='date',
                content=match.group(0),
                confidence=0.85,
                metadata={'pattern': 'date'}
            ))
        
        return facts
    
    def _extract_relations(self, context: str) -> List[ExtractedFact]:
        """Extract relationships between entities"""
        if not self.nlp:
            return []
        
        facts = []
        doc = self.nlp(context)
        
        # Simple subject-verb-object extraction
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ in ["dobj", "attr"]:
                        obj = child.text
                
                if subject and obj:
                    relation = f"{subject} {token.lemma_} {obj}"
                    facts.append(ExtractedFact(
                        fact_type='relation',
                        content=relation,
                        confidence=0.7,
                        metadata={
                            'subject': subject,
                            'predicate': token.lemma_,
                            'object': obj
                        }
                    ))
        
        return facts
    
    def format_facts_for_transmission(
        self, 
        facts: List[ExtractedFact],
        max_tokens: int = 50,
        min_confidence: float = 0.5
    ) -> str:
        """
        Format extracted facts into compact text for KV transmission
        
        Args:
            facts: List of extracted facts
            max_tokens: Maximum tokens to include
            min_confidence: Minimum confidence threshold
            
        Returns:
            Formatted text string
        """
        # Filter by confidence
        filtered = [f for f in facts if f.confidence >= min_confidence]
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Group by type for better organization
        by_type = defaultdict(list)
        for fact in filtered:
            by_type[fact.fact_type].append(fact.content)
        
        # Format output
        parts = []
        token_count = 0
        
        # Priority order: entities, numeric, keyphrases, relations
        priority_order = [
            'entity', 'numeric', 'price', 'rate_limit', 
            'api_endpoint', 'api_header', 'url',
            'keyphrase', 'concept', 'relation'
        ]
        
        for fact_type in priority_order:
            if fact_type not in by_type:
                continue
            
            for content in by_type[fact_type]:
                # Rough token estimate
                estimated_tokens = len(content.split())
                
                if token_count + estimated_tokens > max_tokens:
                    break
                
                parts.append(content)
                token_count += estimated_tokens
            
            if token_count >= max_tokens:
                break
        
        return "; ".join(parts)


class ContextTypeDetector:
    """Automatically detect context type for adaptive extraction"""
    
    @staticmethod
    def detect(context: str) -> str:
        """
        Detect context type
        
        Returns:
            One of: 'api_docs', 'technical_spec', 'product_info', 'general'
        """
        context_lower = context.lower()
        
        # API documentation indicators
        api_indicators = [
            'api', 'endpoint', 'get', 'post', 'put', 'delete',
            'header', 'authentication', 'rate limit', 'request'
        ]
        api_score = sum(1 for ind in api_indicators if ind in context_lower)
        
        # Technical specification indicators
        tech_indicators = [
            'specification', 'requirement', 'parameter', 'configuration',
            'cpu', 'ram', 'gpu', 'ghz', 'mhz', 'cores'
        ]
        tech_score = sum(1 for ind in tech_indicators if ind in context_lower)
        
        # Product information indicators
        product_indicators = [
            'price', 'usd', '$', 'features', 'launched', 'weight',
            'battery', 'display', 'storage'
        ]
        product_score = sum(1 for ind in product_indicators if ind in context_lower)
        
        # Determine type
        max_score = max(api_score, tech_score, product_score)
        
        if max_score >= 3:
            if api_score == max_score:
                return 'api_docs'
            elif tech_score == max_score:
                return 'technical_spec'
            elif product_score == max_score:
                return 'product_info'
        
        return 'general'