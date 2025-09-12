-- Migration: URL Detection and Auto-Addition Tables
-- Description: Add tables for tracking URL suggestions and auto-additions

-- Table for storing URL suggestions that require user review
CREATE TABLE IF NOT EXISTS url_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    relevance_score DECIMAL(3,2) NOT NULL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    content_quality_score DECIMAL(3,2) NOT NULL CHECK (content_quality_score >= 0 AND content_quality_score <= 1),
    domain_reputation_score DECIMAL(3,2) NOT NULL CHECK (domain_reputation_score >= 0 AND domain_reputation_score <= 1),
    overall_score DECIMAL(3,2) NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),
    reasoning TEXT NOT NULL,
    recommended_action VARCHAR(20) NOT NULL CHECK (recommended_action IN ('auto_add', 'suggest', 'ignore')),
    source_context TEXT DEFAULT '',
    tags TEXT[] DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    approved_at TIMESTAMP WITH TIME ZONE NULL,
    approved_by UUID NULL,
    rejected_at TIMESTAMP WITH TIME ZONE NULL,
    rejected_by UUID NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for logging automatic additions to knowledge base
CREATE TABLE IF NOT EXISTS url_auto_additions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    overall_score DECIMAL(3,2) NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),
    reasoning TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    source_context TEXT DEFAULT '',
    source_id UUID NULL, -- Links to archon_sources after crawling
    crawl_success BOOLEAN NULL,
    crawl_error TEXT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    crawled_at TIMESTAMP WITH TIME ZONE NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for URL detection configuration and user preferences
CREATE TABLE IF NOT EXISTS url_detection_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NULL, -- NULL for global settings
    project_id UUID NULL, -- NULL for user-wide settings
    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, project_id, setting_key)
);

-- Table for tracking URL detection analytics
CREATE TABLE IF NOT EXISTS url_detection_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL, -- 'detected', 'analyzed', 'auto_added', 'suggested', 'approved', 'rejected'
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    source_context TEXT DEFAULT '',
    overall_score DECIMAL(3,2) NULL,
    user_action VARCHAR(50) NULL, -- 'approved', 'rejected', 'ignored'
    processing_time_ms INTEGER NULL,
    error_message TEXT NULL,
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id VARCHAR(100) NULL,
    user_agent TEXT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_url_suggestions_status ON url_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_url_suggestions_overall_score ON url_suggestions(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_url_suggestions_detected_at ON url_suggestions(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_url_suggestions_domain ON url_suggestions(domain);

CREATE INDEX IF NOT EXISTS idx_url_auto_additions_url ON url_auto_additions(url);
CREATE INDEX IF NOT EXISTS idx_url_auto_additions_added_at ON url_auto_additions(added_at DESC);
CREATE INDEX IF NOT EXISTS idx_url_auto_additions_source_id ON url_auto_additions(source_id);

CREATE INDEX IF NOT EXISTS idx_url_detection_settings_user_project ON url_detection_settings(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_url_detection_settings_key ON url_detection_settings(setting_key);

CREATE INDEX IF NOT EXISTS idx_url_detection_analytics_event_type ON url_detection_analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_url_detection_analytics_domain ON url_detection_analytics(domain);
CREATE INDEX IF NOT EXISTS idx_url_detection_analytics_timestamp ON url_detection_analytics(event_timestamp DESC);

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_url_suggestions_updated_at BEFORE UPDATE ON url_suggestions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_url_detection_settings_updated_at BEFORE UPDATE ON url_detection_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Default global configuration settings
INSERT INTO url_detection_settings (user_id, project_id, setting_key, setting_value)
VALUES 
    (NULL, NULL, 'global_config', '{
        "enabled": true,
        "auto_add_threshold": 0.85,
        "suggest_threshold": 0.6,
        "ignore_threshold": 0.3,
        "max_concurrent_analyses": 10,
        "cache_ttl_hours": 24,
        "excluded_domains": ["localhost", "127.0.0.1", "0.0.0.0", "example.com", "test.com", "internal.local"],
        "preferred_domains": ["github.com", "docs.python.org", "stackoverflow.com", "developer.mozilla.org", "w3.org", "ietf.org"]
    }')
ON CONFLICT (user_id, project_id, setting_key) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE url_suggestions IS 'Stores URLs detected by the system that require user review before adding to knowledge base';
COMMENT ON TABLE url_auto_additions IS 'Logs URLs that were automatically added to the knowledge base without user intervention';
COMMENT ON TABLE url_detection_settings IS 'Stores configuration settings for URL detection system, supports global, user, and project-level settings';
COMMENT ON TABLE url_detection_analytics IS 'Analytics data for URL detection system performance and user behavior';

COMMENT ON COLUMN url_suggestions.overall_score IS 'Weighted score combining relevance, quality, and domain reputation (0.0-1.0)';
COMMENT ON COLUMN url_suggestions.recommended_action IS 'AI recommendation: auto_add (high confidence), suggest (review needed), ignore (low value)';
COMMENT ON COLUMN url_suggestions.source_context IS 'Context where the URL was discovered (e.g., agent_response, task_description, mcp_call)';

COMMENT ON COLUMN url_auto_additions.source_id IS 'References archon_sources.id after successful crawling';
COMMENT ON COLUMN url_auto_additions.crawl_success IS 'Whether the automatic crawling was successful';

-- Example queries for common operations:

-- Get pending URL suggestions for user review, ordered by score
-- SELECT * FROM url_suggestions WHERE status = 'pending' ORDER BY overall_score DESC LIMIT 10;

-- Get URLs auto-added in the last 24 hours
-- SELECT * FROM url_auto_additions WHERE added_at > NOW() - INTERVAL '24 hours' ORDER BY added_at DESC;

-- Get URL detection analytics for the last week
-- SELECT event_type, COUNT(*) as count FROM url_detection_analytics 
-- WHERE event_timestamp > NOW() - INTERVAL '7 days' 
-- GROUP BY event_type ORDER BY count DESC;

-- Update global configuration
-- UPDATE url_detection_settings SET setting_value = '{"enabled": false}' 
-- WHERE user_id IS NULL AND project_id IS NULL AND setting_key = 'global_config';