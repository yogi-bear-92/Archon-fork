#!/usr/bin/env python3
"""
Knowledge Base Tagging System Performance Optimization Guide

This script provides concrete implementation guidance for optimizing
the knowledge base tagging system based on the performance analysis results.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any


class PerformanceOptimizationGuide:
    """Implementation guide for performance optimizations."""
    
    def __init__(self):
        self.optimizations = self._define_optimization_catalog()
    
    def _define_optimization_catalog(self) -> Dict[str, Any]:
        """Define comprehensive optimization catalog with implementation details."""
        return {
            "database_optimization": {
                "priority": "high",
                "impact": "50-70% faster queries",
                "implementations": [
                    {
                        "name": "GIN Index for Tag Arrays",
                        "type": "database_index",
                        "sql": """
-- Create GIN index for tag arrays (PostgreSQL)
CREATE INDEX CONCURRENTLY idx_knowledge_items_tags_gin 
ON knowledge_items USING gin(tags);

-- Alternative GiST index for range queries
CREATE INDEX CONCURRENTLY idx_knowledge_items_tags_gist 
ON knowledge_items USING gist(tags);

-- Analyze the table to update statistics
ANALYZE knowledge_items;
""",
                        "benefits": [
                            "Optimizes tag containment queries (@> operator)",
                            "Supports efficient multi-tag intersections",
                            "Scales well with large tag sets"
                        ],
                        "estimated_improvement": "60%",
                        "implementation_time": "30 minutes"
                    },
                    {
                        "name": "Tag Frequency Materialized View",
                        "type": "materialized_view",
                        "sql": """
-- Create materialized view for tag frequency analysis
CREATE MATERIALIZED VIEW tag_frequency AS
SELECT 
    unnest(tags) as tag,
    COUNT(*) as frequency,
    COUNT(*) * 1.0 / (SELECT COUNT(*) FROM knowledge_items) as selectivity
FROM knowledge_items 
WHERE tags IS NOT NULL 
GROUP BY unnest(tags)
ORDER BY frequency DESC;

-- Create index on the materialized view
CREATE INDEX idx_tag_frequency_tag ON tag_frequency(tag);
CREATE INDEX idx_tag_frequency_freq ON tag_frequency(frequency DESC);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_tag_frequency()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY tag_frequency;
END;
$$ LANGUAGE plpgsql;
""",
                        "benefits": [
                            "Enables query optimization based on tag selectivity",
                            "Supports intelligent query planning",
                            "Provides analytics for tag usage patterns"
                        ],
                        "estimated_improvement": "25%",
                        "implementation_time": "45 minutes"
                    }
                ]
            },
            
            "caching_optimization": {
                "priority": "high", 
                "impact": "60-80% reduction in database queries",
                "implementations": [
                    {
                        "name": "Redis Tag-Based Caching",
                        "type": "caching_layer",
                        "code": """
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
""",
                        "benefits": [
                            "Dramatically reduces database load",
                            "Smart invalidation based on tag relationships", 
                            "Scales horizontally with Redis clustering"
                        ],
                        "estimated_improvement": "70%",
                        "implementation_time": "2 hours"
                    }
                ]
            },
            
            "query_optimization": {
                "priority": "medium",
                "impact": "25-40% faster multi-tag queries", 
                "implementations": [
                    {
                        "name": "Tag Selectivity Query Reordering",
                        "type": "query_optimization",
                        "code": """
class TagQueryOptimizer:
    def __init__(self, db_connection):
        self.db = db_connection
        self.tag_selectivity_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cache_update = 0
    
    async def get_tag_selectivity(self, tags: List[str]) -> Dict[str, float]:
        '''Get selectivity scores for tags (lower = more selective).'''
        current_time = time.time()
        
        # Refresh cache if needed
        if current_time - self.last_cache_update > self.cache_ttl:
            await self._refresh_selectivity_cache()
        
        selectivity = {}
        for tag in tags:
            selectivity[tag] = self.tag_selectivity_cache.get(tag, 0.5)  # Default middle selectivity
        
        return selectivity
    
    async def _refresh_selectivity_cache(self):
        '''Refresh tag selectivity cache from database.'''
        query = '''
        SELECT tag, selectivity 
        FROM tag_frequency 
        WHERE selectivity > 0
        '''
        
        result = await self.db.fetch(query)
        self.tag_selectivity_cache = {row['tag']: row['selectivity'] for row in result}
        self.last_cache_update = time.time()
    
    async def optimize_multi_tag_query(self, tags: List[str], base_query: str) -> str:
        '''Reorder tag conditions based on selectivity for optimal performance.'''
        if len(tags) <= 1:
            return base_query
        
        # Get selectivity scores
        selectivity = await self.get_tag_selectivity(tags)
        
        # Sort tags by selectivity (most selective first)
        optimized_tags = sorted(tags, key=lambda t: selectivity[t])
        
        # Build optimized query
        tag_conditions = []
        for i, tag in enumerate(optimized_tags):
            if i == 0:
                # Most selective condition first
                tag_conditions.append(f"tags @> ARRAY['{tag}']")
            else:
                # Additional conditions
                tag_conditions.append(f"AND tags @> ARRAY['{tag}']")
        
        # Combine with base query
        optimized_query = f"{base_query} WHERE {' '.join(tag_conditions)}"
        
        return optimized_query
    
    async def build_efficient_multi_tag_query(self, tags: List[str], per_page: int = 50, offset: int = 0) -> str:
        '''Build an efficient multi-tag query with proper indexing hints.'''
        if not tags:
            return "SELECT * FROM knowledge_items ORDER BY created_at DESC LIMIT $1 OFFSET $2"
        
        selectivity = await self.get_tag_selectivity(tags)
        optimized_tags = sorted(tags, key=lambda t: selectivity[t])
        
        # Use array overlap for efficient multi-tag matching
        query = f'''
        SELECT ki.* 
        FROM knowledge_items ki
        WHERE ki.tags && ARRAY[{','.join([f"'{tag}'" for tag in optimized_tags])}]
        AND ki.tags @> ARRAY[{','.join([f"'{tag}'" for tag in optimized_tags])}]
        ORDER BY ki.created_at DESC
        LIMIT $1 OFFSET $2
        '''
        
        return query

# Usage example:
optimizer = TagQueryOptimizer(db_connection)
optimized_query = await optimizer.optimize_multi_tag_query(
    ["ai-orchestration", "enterprise", "python-framework"],
    "SELECT * FROM knowledge_items"
)
""",
                        "benefits": [
                            "Reduces query execution time for complex tag combinations",
                            "Leverages database query planner more effectively",
                            "Adapts to changing tag distribution patterns"
                        ],
                        "estimated_improvement": "35%",
                        "implementation_time": "1.5 hours"
                    }
                ]
            },
            
            "memory_optimization": {
                "priority": "medium",
                "impact": "15-20% memory reduction",
                "implementations": [
                    {
                        "name": "Tag String Interning and Normalization",
                        "type": "memory_optimization",
                        "code": """
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
""",
                        "benefits": [
                            "Reduces memory usage for repeated tag strings",
                            "Normalizes tag variations automatically",
                            "Provides memory usage analytics"
                        ],
                        "estimated_improvement": "18%",
                        "implementation_time": "1 hour"
                    }
                ]
            },
            
            "monitoring_optimization": {
                "priority": "medium",
                "impact": "Proactive performance issue detection",
                "implementations": [
                    {
                        "name": "Tag Performance Monitoring Dashboard",
                        "type": "monitoring_system",
                        "code": """
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

@dataclass
class TagPerformanceMetric:
    timestamp: float
    tag: str
    operation: str  # 'query', 'update', 'cache_hit', 'cache_miss'
    duration_ms: float
    result_count: Optional[int] = None
    cache_hit: bool = False

class TagPerformanceMonitor:
    '''Monitor and analyze tag-based operation performance.'''
    
    def __init__(self, history_size: int = 10000):
        self.metrics: deque = deque(maxlen=history_size)
        self.tag_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'query_count': 0,
            'total_duration_ms': 0.0,
            'avg_duration_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        self.start_time = time.time()
    
    def record_metric(self, metric: TagPerformanceMetric) -> None:
        '''Record a performance metric.'''
        self.metrics.append(metric)
        self._update_tag_stats(metric)
    
    def record_tag_query(self, tags: List[str], duration_ms: float, 
                        result_count: int, cache_hit: bool = False) -> None:
        '''Record a tag query operation.'''
        for tag in tags:
            metric = TagPerformanceMetric(
                timestamp=time.time(),
                tag=tag,
                operation='query',
                duration_ms=duration_ms / len(tags),  # Distribute duration across tags
                result_count=result_count,
                cache_hit=cache_hit
            )
            self.record_metric(metric)
    
    def _update_tag_stats(self, metric: TagPerformanceMetric) -> None:
        '''Update aggregated statistics for a tag.'''
        stats = self.tag_stats[metric.tag]
        
        if metric.operation == 'query':
            stats['query_count'] += 1
            stats['total_duration_ms'] += metric.duration_ms
            stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['query_count']
            
            if metric.cache_hit:
                stats['cache_hits'] += 1
            else:
                stats['cache_misses'] += 1
    
    def get_tag_performance_report(self, tag: str) -> Dict[str, Any]:
        '''Get performance report for a specific tag.'''
        stats = self.tag_stats.get(tag, {})
        
        if not stats:
            return {"error": f"No performance data for tag: {tag}"}
        
        cache_hit_rate = 0.0
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        
        return {
            "tag": tag,
            "query_count": stats['query_count'],
            "avg_duration_ms": round(stats['avg_duration_ms'], 2),
            "cache_hit_rate": round(cache_hit_rate * 100, 1),
            "performance_grade": self._calculate_performance_grade(stats['avg_duration_ms'], cache_hit_rate)
        }
    
    def get_system_performance_overview(self) -> Dict[str, Any]:
        '''Get overall system performance overview.'''
        if not self.tag_stats:
            return {"error": "No performance data available"}
        
        # Calculate aggregate metrics
        total_queries = sum(stats['query_count'] for stats in self.tag_stats.values())
        avg_response_time = sum(stats['avg_duration_ms'] * stats['query_count'] 
                               for stats in self.tag_stats.values()) / total_queries if total_queries > 0 else 0
        
        total_cache_hits = sum(stats['cache_hits'] for stats in self.tag_stats.values())
        total_cache_operations = sum(stats['cache_hits'] + stats['cache_misses'] 
                                   for stats in self.tag_stats.values())
        overall_cache_hit_rate = total_cache_hits / total_cache_operations if total_cache_operations > 0 else 0
        
        # Identify slow tags
        slow_tags = [(tag, stats['avg_duration_ms']) for tag, stats in self.tag_stats.items() 
                    if stats['avg_duration_ms'] > 100]
        slow_tags.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_queries": total_queries,
            "avg_response_time_ms": round(avg_response_time, 2),
            "overall_cache_hit_rate": round(overall_cache_hit_rate * 100, 1),
            "monitored_tags": len(self.tag_stats),
            "slow_tags": slow_tags[:10],  # Top 10 slowest tags
            "uptime_hours": round((time.time() - self.start_time) / 3600, 1)
        }
    
    def _calculate_performance_grade(self, avg_duration_ms: float, cache_hit_rate: float) -> str:
        '''Calculate performance grade for a tag.'''
        if avg_duration_ms <= 20 and cache_hit_rate >= 0.8:
            return "A"
        elif avg_duration_ms <= 50 and cache_hit_rate >= 0.6:
            return "B"
        elif avg_duration_ms <= 100 and cache_hit_rate >= 0.4:
            return "C"
        else:
            return "D"
    
    def export_metrics_json(self) -> str:
        '''Export metrics as JSON for external analysis.'''
        export_data = {
            "timestamp": time.time(),
            "tag_stats": dict(self.tag_stats),
            "system_overview": self.get_system_performance_overview(),
            "recent_metrics": [asdict(metric) for metric in list(self.metrics)[-100:]]  # Last 100 metrics
        }
        
        return json.dumps(export_data, indent=2)

# Usage example:
monitor = TagPerformanceMonitor()

# Record query performance
monitor.record_tag_query(
    tags=["ai-orchestration", "enterprise"], 
    duration_ms=45.2, 
    result_count=12, 
    cache_hit=False
)

# Get performance reports
tag_report = monitor.get_tag_performance_report("ai-orchestration")
system_overview = monitor.get_system_performance_overview()
""",
                        "benefits": [
                            "Real-time performance monitoring",
                            "Identifies slow tags and queries",
                            "Tracks cache effectiveness",
                            "Provides actionable performance insights"
                        ],
                        "estimated_improvement": "Proactive optimization",
                        "implementation_time": "3 hours"
                    }
                ]
            }
        }
    
    def generate_implementation_roadmap(self) -> Dict[str, Any]:
        """Generate prioritized implementation roadmap."""
        roadmap = {
            "phase_1_critical": {
                "timeframe": "Week 1-2",
                "optimizations": [
                    "Database GIN indexing",
                    "Redis caching layer"
                ],
                "expected_impact": "60-80% performance improvement",
                "effort_estimate": "3-4 days"
            },
            "phase_2_enhancement": {
                "timeframe": "Week 3-4", 
                "optimizations": [
                    "Query optimization with selectivity",
                    "Memory optimization with interning"
                ],
                "expected_impact": "Additional 20-30% improvement",
                "effort_estimate": "2-3 days"
            },
            "phase_3_monitoring": {
                "timeframe": "Week 5",
                "optimizations": [
                    "Performance monitoring dashboard",
                    "Automated optimization alerts"
                ],
                "expected_impact": "Proactive performance management",
                "effort_estimate": "1-2 days"
            }
        }
        
        return roadmap
    
    def print_optimization_guide(self):
        """Print comprehensive optimization guide."""
        print("=" * 80)
        print("🚀 KNOWLEDGE BASE TAGGING SYSTEM OPTIMIZATION GUIDE")
        print("=" * 80)
        
        print("\n📋 IMPLEMENTATION ROADMAP")
        print("-" * 40)
        roadmap = self.generate_implementation_roadmap()
        
        for phase, details in roadmap.items():
            print(f"\n🔸 {phase.upper().replace('_', ' ')}")
            print(f"   Timeframe: {details['timeframe']}")
            print(f"   Optimizations: {', '.join(details['optimizations'])}")
            print(f"   Expected impact: {details['expected_impact']}")
            print(f"   Effort estimate: {details['effort_estimate']}")
        
        print("\n\n🛠️  DETAILED IMPLEMENTATION GUIDES")
        print("-" * 40)
        
        for category, config in self.optimizations.items():
            print(f"\n📊 {category.upper().replace('_', ' ')}")
            print(f"   Priority: {config['priority'].upper()}")
            print(f"   Expected Impact: {config['impact']}")
            
            for impl in config['implementations']:
                print(f"\n   🔧 {impl['name']}")
                print(f"      Type: {impl['type']}")
                print(f"      Estimated improvement: {impl['estimated_improvement']}")
                print(f"      Implementation time: {impl['implementation_time']}")
                print(f"      Benefits:")
                for benefit in impl['benefits']:
                    print(f"         • {benefit}")
        
        print("\n\n🎯 RECOMMENDED NEXT STEPS")
        print("-" * 40)
        print("1. 🔥 Start with database indexing (highest impact, moderate effort)")
        print("2. 🔥 Implement Redis caching layer (high impact, moderate effort)")
        print("3. ⚡ Add query optimization (medium impact, low effort)")
        print("4. ⚡ Implement memory optimization (medium impact, low effort)")
        print("5. ⚡ Set up performance monitoring (long-term value)")
        
        print("\n📈 EXPECTED CUMULATIVE IMPROVEMENTS")
        print("-" * 40)
        print("• After Phase 1: 60-80% faster queries, 70% fewer database hits")
        print("• After Phase 2: 80-100% total improvement from baseline")
        print("• After Phase 3: Proactive performance management and optimization")
        
        print("\n" + "=" * 80)


def main():
    """Main execution function."""
    guide = PerformanceOptimizationGuide()
    
    # Print comprehensive guide
    guide.print_optimization_guide()
    
    # Save implementation details to files
    output_dir = Path("optimization_implementations")
    output_dir.mkdir(exist_ok=True)
    
    for category, config in guide.optimizations.items():
        for impl in config['implementations']:
            if 'sql' in impl:
                sql_file = output_dir / f"{category}_{impl['name'].lower().replace(' ', '_')}.sql"
                with open(sql_file, 'w') as f:
                    f.write(impl['sql'])
                print(f"💾 SQL implementation saved to: {sql_file}")
            
            if 'code' in impl:
                py_file = output_dir / f"{category}_{impl['name'].lower().replace(' ', '_')}.py"
                with open(py_file, 'w') as f:
                    f.write(impl['code'])
                print(f"💾 Python implementation saved to: {py_file}")
    
    # Save roadmap
    roadmap_file = Path("performance_optimization_roadmap.json")
    with open(roadmap_file, 'w') as f:
        json.dump(guide.generate_implementation_roadmap(), f, indent=2)
    
    print(f"\n💾 Implementation roadmap saved to: {roadmap_file.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())