
import redis
import json
import hashlib
from typing import List, Optional, Dict, Any

class TagAwareCaching:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_prefix = "kb_tags"
        self.default_ttl = 3600  # 1 hour
    
    def generate_cache_key(self, tags: List[str], query_params: Dict[str, Any]) -> str:
        '''Generate consistent cache key for tag-based queries.'''
        # Sort tags for consistent caching
        sorted_tags = sorted(tags)
        key_data = {
            "tags": sorted_tags,
            "params": sorted(query_params.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"{self.cache_prefix}:query:{key_hash}"
    
    def get_cached_query(self, tags: List[str], query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        '''Retrieve cached query results.'''
        cache_key = self.generate_cache_key(tags, query_params)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
    
    def cache_query_result(self, tags: List[str], query_params: Dict[str, Any], 
                          result: Dict[str, Any], ttl: Optional[int] = None) -> None:
        '''Cache query results with tag-based invalidation support.'''
        cache_key = self.generate_cache_key(tags, query_params)
        ttl = ttl or self.default_ttl
        
        # Store the result
        self.redis_client.setex(cache_key, ttl, json.dumps(result))
        
        # Add cache key to tag-specific sets for invalidation
        for tag in tags:
            tag_key = f"{self.cache_prefix}:tag:{tag}"
            self.redis_client.sadd(tag_key, cache_key)
            self.redis_client.expire(tag_key, ttl + 300)  # Slightly longer TTL
    
    def invalidate_tag_caches(self, tags: List[str]) -> int:
        '''Invalidate all caches related to specific tags.'''
        invalidated_count = 0
        
        for tag in tags:
            tag_key = f"{self.cache_prefix}:tag:{tag}"
            cache_keys = self.redis_client.smembers(tag_key)
            
            if cache_keys:
                # Delete cached results
                self.redis_client.delete(*cache_keys)
                # Delete tag tracking set
                self.redis_client.delete(tag_key)
                invalidated_count += len(cache_keys)
        
        return invalidated_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        '''Get cache performance statistics.'''
        all_keys = self.redis_client.keys(f"{self.cache_prefix}:query:*")
        tag_keys = self.redis_client.keys(f"{self.cache_prefix}:tag:*")
        
        return {
            "cached_queries": len(all_keys),
            "tracked_tags": len(tag_keys),
            "memory_usage_mb": self.redis_client.memory_usage() / (1024 * 1024) if hasattr(self.redis_client, 'memory_usage') else 0
        }

# Usage example:
cache = TagAwareCaching()

# Cache a query result
result = {"items": [...], "total": 42}
cache.cache_query_result(["ai-orchestration", "enterprise"], {"per_page": 50}, result)

# Retrieve cached result
cached = cache.get_cached_query(["ai-orchestration", "enterprise"], {"per_page": 50})

# Invalidate when tags are updated
cache.invalidate_tag_caches(["ai-orchestration"])
