
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
