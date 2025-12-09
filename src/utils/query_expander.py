"""Query expansion utilities for better entity matching."""

from typing import List, Set
import re


class QueryExpander:
    """Expands queries with related terms for better retrieval."""
    
    def __init__(self):
        """Initialize query expander with expansion rules."""
        # Define expansion mappings
        self.expansions = {
            # Buddhism related
            'buddhism': ['buddha', 'buddhist', 'buddhists', 'buddhism'],
            'buddha': ['buddha', 'buddhism', 'buddhist'],
            'buddhist': ['buddha', 'buddhism', 'buddhist', 'buddhists'],
            
            # Caste related
            'caste': ['caste', 'castes', 'varna', 'jati'],
            'untouchable': ['untouchable', 'untouchables', 'dalit', 'dalits'],
            'dalit': ['dalit', 'dalits', 'untouchable', 'untouchables'],
            
            # Constitution related
            'constitution': ['constitution', 'constitutional', 'constituent assembly'],
            
            # Education related
            'education': ['education', 'educational', 'learning', 'knowledge'],
            
            # Social justice related
            'equality': ['equality', 'equal', 'egalitarian'],
            'justice': ['justice', 'social justice', 'fairness'],
            'rights': ['rights', 'human rights', 'fundamental rights'],
            
            # Religion related
            'hinduism': ['hinduism', 'hindu', 'hindus'],
            'religion': ['religion', 'religious', 'faith'],
            
            # Ambedkar variations
            'ambedkar': ['ambedkar', 'b.r. ambedkar', 'dr. ambedkar', 'babasaheb'],
            'babasaheb': ['babasaheb', 'ambedkar', 'dr. ambedkar'],
        }
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms.
        
        Args:
            query: Original query string
            
        Returns:
            List of expanded query terms
        """
        query_lower = query.lower()
        expanded_terms = set()
        
        # Add original query
        expanded_terms.add(query)
        
        # Extract words from query
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Expand each word
        for word in words:
            if word in self.expansions:
                expanded_terms.update(self.expansions[word])
            else:
                expanded_terms.add(word)
        
        return list(expanded_terms)
    
    def get_expanded_entities(self, query: str) -> Set[str]:
        """
        Get potential entity names from expanded query.
        
        Args:
            query: Original query string
            
        Returns:
            Set of potential entity names
        """
        expanded = self.expand_query(query)
        entities = set()
        
        for term in expanded:
            # Capitalize for entity matching
            entities.add(term.capitalize())
            entities.add(term.title())
            entities.add(term.upper())
            entities.add(term.lower())
        
        return entities
