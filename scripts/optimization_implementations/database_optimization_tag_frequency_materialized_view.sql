
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
