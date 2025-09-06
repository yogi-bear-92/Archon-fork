
import weakref
from typing import Dict, List, Set

class TagInternManager:
    '''Manages string interning for tags to reduce memory usage.'''
    
    def __init__(self):
        self._interned_tags: Dict[str, str] = {}
        self._tag_references: Dict[str, int] = {}
        self._tag_normalizations: Dict[str, str] = {}
    
    def intern_tag(self, tag: str) -> str:
        '''Intern a tag string to reduce memory usage.'''
        # Normalize the tag
        normalized_tag = self._normalize_tag(tag)
        
        # Check if already interned
        if normalized_tag in self._interned_tags:
            self._tag_references[normalized_tag] += 1
            return self._interned_tags[normalized_tag]
        
        # Intern the tag
        interned = sys.intern(normalized_tag)
        self._interned_tags[normalized_tag] = interned
        self._tag_references[normalized_tag] = 1
        
        return interned
    
    def intern_tag_list(self, tags: List[str]) -> List[str]:
        '''Intern a list of tags.'''
        return [self.intern_tag(tag) for tag in tags]
    
    def _normalize_tag(self, tag: str) -> str:
        '''Normalize tag string for consistent interning.'''
        if tag in self._tag_normalizations:
            return self._tag_normalizations[tag]
        
        # Normalization rules
        normalized = tag.lower().strip()
        
        # Cache normalization
        self._tag_normalizations[tag] = normalized
        return normalized
    
    def release_tag(self, tag: str) -> None:
        '''Release reference to an interned tag.'''
        normalized = self._normalize_tag(tag)
        if normalized in self._tag_references:
            self._tag_references[normalized] -= 1
            
            # Clean up if no more references
            if self._tag_references[normalized] <= 0:
                del self._interned_tags[normalized]
                del self._tag_references[normalized]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        '''Get memory usage statistics.'''
        return {
            "interned_tags_count": len(self._interned_tags),
            "total_references": sum(self._tag_references.values()),
            "normalization_cache_size": len(self._tag_normalizations),
            "estimated_memory_saved_kb": len(self._interned_tags) * 0.1  # Rough estimate
        }

class MemoryEfficientTagProcessor:
    '''Process tags with memory optimization.'''
    
    def __init__(self):
        self.intern_manager = TagInternManager()
        self._frequent_tags_cache: Set[str] = set()
    
    def process_knowledge_item_tags(self, knowledge_item: Dict[str, Any]) -> Dict[str, Any]:
        '''Process tags in a knowledge item for memory efficiency.'''
        if 'metadata' not in knowledge_item or 'tags' not in knowledge_item['metadata']:
            return knowledge_item
        
        tags = knowledge_item['metadata']['tags']
        if isinstance(tags, list):
            # Intern all tags
            interned_tags = self.intern_manager.intern_tag_list(tags)
            knowledge_item['metadata']['tags'] = interned_tags
            
            # Update frequent tags cache
            self._update_frequent_tags_cache(interned_tags)
        
        return knowledge_item
    
    def _update_frequent_tags_cache(self, tags: List[str]) -> None:
        '''Update cache of frequently used tags.'''
        for tag in tags:
            if tag not in self._frequent_tags_cache and len(self._frequent_tags_cache) < 100:
                self._frequent_tags_cache.add(tag)

# Usage example:
processor = MemoryEfficientTagProcessor()
knowledge_item = {"metadata": {"tags": ["AI-Orchestration", "Enterprise-AI", "multi-agent-systems"]}}
optimized_item = processor.process_knowledge_item_tags(knowledge_item)
