
-- Create GIN index for tag arrays (PostgreSQL)
CREATE INDEX CONCURRENTLY idx_knowledge_items_tags_gin 
ON knowledge_items USING gin(tags);

-- Alternative GiST index for range queries
CREATE INDEX CONCURRENTLY idx_knowledge_items_tags_gist 
ON knowledge_items USING gist(tags);

-- Analyze the table to update statistics
ANALYZE knowledge_items;
