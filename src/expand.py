"""
Query expansion module for generating related queries and intent trees.
"""
import re
from typing import List, Dict, Set, Optional
import google.generativeai as genai

from .utils import (
    get_env_var, retry_on_failure, logger, Tokenizer
)

class QueryExpander:
    """Expand user queries into multiple related queries for comprehensive search."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
        self.tokenizer = Tokenizer()
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def expand_query(self, query: str, expansion_types: List[str] = None) -> List[Dict]:
        """Expand a query into multiple related queries."""
        if expansion_types is None:
            expansion_types = ['synonyms', 'intents', 'questions', 'variations']
        
        expanded_queries = []
        
        for expansion_type in expansion_types:
            try:
                if expansion_type == 'synonyms':
                    synonyms = self._generate_synonyms(query)
                    expanded_queries.extend(synonyms)
                elif expansion_type == 'intents':
                    intents = self._generate_intents(query)
                    expanded_queries.extend(intents)
                elif expansion_type == 'questions':
                    questions = self._generate_questions(query)
                    expanded_queries.extend(questions)
                elif expansion_type == 'variations':
                    variations = self._generate_variations(query)
                    expanded_queries.extend(variations)
            except Exception as e:
                logger.error(f"Error in {expansion_type} expansion: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query_dict in expanded_queries:
            query_text = query_dict['query']
            if query_text not in seen:
                seen.add(query_text)
                unique_queries.append(query_dict)
        
        return unique_queries
    
    def _generate_synonyms(self, query: str) -> List[Dict]:
        """Generate synonym-based query variations."""
        prompt = f"""
        Generate 5-8 synonym-based variations of this query: "{query}"
        
        Focus on:
        - Different ways to express the same concept
        - Alternative terminology
        - Related terms that might be used
        
        Return only the variations, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        synonyms = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                    synonyms.append({
                        'query': line,
                        'type': 'synonym',
                        'original': query,
                        'tokens': self.tokenizer.count_tokens(line)
                    })
        
        return synonyms
    
    def _generate_intents(self, query: str) -> List[Dict]:
        """Generate different user intents for the query."""
        prompt = f"""
        Generate 5-8 different user intents for this query: "{query}"
        
        Consider different user goals:
        - Information seeking
        - Problem solving
        - Comparison
        - How-to/instructions
        - Reviews/opinions
        - Best practices
        - Troubleshooting
        
        Return only the intent variations, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        intents = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                    intents.append({
                        'query': line,
                        'type': 'intent',
                        'original': query,
                        'tokens': self.tokenizer.count_tokens(line)
                    })
        
        return intents
    
    def _generate_questions(self, query: str) -> List[Dict]:
        """Generate question-based variations of the query."""
        prompt = f"""
        Generate 5-8 question-based variations of this query: "{query}"
        
        Include different question types:
        - What is...?
        - How to...?
        - Why...?
        - When...?
        - Where...?
        - Which...?
        - Best...?
        - Top...?
        
        Return only the questions, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        questions = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                    questions.append({
                        'query': line,
                        'type': 'question',
                        'original': query,
                        'tokens': self.tokenizer.count_tokens(line)
                    })
        
        return questions
    
    def _generate_variations(self, query: str) -> List[Dict]:
        """Generate general variations of the query."""
        prompt = f"""
        Generate 5-8 different ways to express this query: "{query}"
        
        Consider:
        - Different word orders
        - Alternative phrasings
        - More/less specific versions
        - Formal/informal variations
        - Industry-specific terminology
        
        Return only the variations, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        variations = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                    variations.append({
                        'query': line,
                        'type': 'variation',
                        'original': query,
                        'tokens': self.tokenizer.count_tokens(line)
                    })
        
        return variations

class IntentTree:
    """Generate intent trees for comprehensive query understanding."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = get_env_var("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
        self.tokenizer = Tokenizer()
    
    def generate_intent_tree(self, query: str, max_depth: int = 3) -> Dict:
        """Generate a hierarchical intent tree for a query."""
        tree = {
            'root_query': query,
            'nodes': [],
            'max_depth': max_depth
        }
        
        # Generate main intents
        main_intents = self._generate_main_intents(query)
        
        for intent in main_intents:
            intent_node = {
                'intent': intent,
                'queries': self._generate_queries_for_intent(intent, query),
                'sub_intents': []
            }
            
            # Generate sub-intents if depth allows
            if max_depth > 1:
                sub_intents = self._generate_sub_intents(intent, query)
                for sub_intent in sub_intents:
                    sub_node = {
                        'intent': sub_intent,
                        'queries': self._generate_queries_for_intent(sub_intent, query),
                        'sub_intents': []
                    }
                    intent_node['sub_intents'].append(sub_node)
            
            tree['nodes'].append(intent_node)
        
        return tree
    
    def _generate_main_intents(self, query: str) -> List[str]:
        """Generate main user intents for the query."""
        prompt = f"""
        Generate 4-6 main user intents for this query: "{query}"
        
        Focus on high-level user goals:
        - Information seeking
        - Problem solving
        - Comparison/evaluation
        - How-to/instructions
        - Reviews/opinions
        - Best practices
        - Troubleshooting
        - Decision making
        
        Return only the intents, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        intents = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                    intents.append(line)
        
        return intents[:6]  # Limit to 6 intents
    
    def _generate_sub_intents(self, main_intent: str, original_query: str) -> List[str]:
        """Generate sub-intents for a main intent."""
        prompt = f"""
        Generate 3-4 specific sub-intents for this main intent: "{main_intent}"
        Original query: "{original_query}"
        
        Focus on specific aspects or approaches within the main intent.
        Return only the sub-intents, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        sub_intents = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.')):
                    sub_intents.append(line)
        
        return sub_intents[:4]  # Limit to 4 sub-intents
    
    def _generate_queries_for_intent(self, intent: str, original_query: str) -> List[str]:
        """Generate specific queries for an intent."""
        prompt = f"""
        Generate 3-5 specific queries for this intent: "{intent}"
        Original query: "{original_query}"
        
        Make the queries specific and actionable for this intent.
        Return only the queries, one per line, without numbering or explanations.
        """
        
        response = self.model_instance.generate_content(prompt)
        queries = []
        
        if response.text:
            lines = response.text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    queries.append(line)
        
        return queries[:5]  # Limit to 5 queries
    
    def flatten_intent_tree(self, tree: Dict) -> List[Dict]:
        """Flatten intent tree into a list of queries with metadata."""
        queries = []
        
        for node in tree['nodes']:
            # Add main intent queries
            for query in node['queries']:
                queries.append({
                    'query': query,
                    'intent': node['intent'],
                    'level': 1,
                    'original_query': tree['root_query']
                })
            
            # Add sub-intent queries
            for sub_node in node['sub_intents']:
                for query in sub_node['queries']:
                    queries.append({
                        'query': query,
                        'intent': f"{node['intent']} > {sub_node['intent']}",
                        'level': 2,
                        'original_query': tree['root_query']
                    })
        
        return queries

def expand_query_simple(query: str, max_expansions: int = 20) -> List[str]:
    """Simple query expansion returning just the expanded queries."""
    expander = QueryExpander()
    expanded = expander.expand_query(query)
    
    # Extract just the query strings
    queries = [item['query'] for item in expanded]
    
    # Limit to max_expansions
    return queries[:max_expansions]

def generate_intent_tree_simple(query: str) -> List[str]:
    """Generate intent tree and return flattened queries."""
    intent_tree = IntentTree()
    tree = intent_tree.generate_intent_tree(query)
    flattened = intent_tree.flatten_intent_tree(tree)
    
    # Extract just the query strings
    queries = [item['query'] for item in flattened]
    
    return queries 