
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
