-- GitHub Monitoring Database Schema
-- Migration: 4_github_monitoring.sql
-- Description: Add tables for GitHub repository monitoring, webhook events, and automation results

-- GitHub repositories being monitored
CREATE TABLE IF NOT EXISTS github_repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_url TEXT NOT NULL UNIQUE,
    repository_name TEXT NOT NULL,
    owner TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    webhook_secret TEXT,
    monitored_paths TEXT[] DEFAULT ARRAY['src/', 'docs/', 'README.md'],
    
    -- Configuration flags
    auto_docs_enabled BOOLEAN DEFAULT true,
    auto_changelog_enabled BOOLEAN DEFAULT true, 
    auto_linting_enabled BOOLEAN DEFAULT false,
    documentation_style TEXT DEFAULT 'comprehensive',
    linting_config JSONB DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Statistics
    total_events_processed INTEGER DEFAULT 0,
    docs_generated INTEGER DEFAULT 0,
    changelogs_generated INTEGER DEFAULT 0,
    lint_runs_completed INTEGER DEFAULT 0
);

-- Index for efficient repository lookups
CREATE INDEX IF NOT EXISTS idx_github_repos_name ON github_repositories(repository_name);
CREATE INDEX IF NOT EXISTS idx_github_repos_owner ON github_repositories(owner);
CREATE INDEX IF NOT EXISTS idx_github_repos_active ON github_repositories(is_active);

-- GitHub webhook events received
CREATE TABLE IF NOT EXISTS github_webhook_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL REFERENCES github_repositories(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL, -- push, pull_request, release, etc.
    event_action TEXT, -- opened, closed, synchronize, etc.
    
    -- Event payload and metadata
    payload JSONB NOT NULL,
    github_delivery_id TEXT,
    github_event_id TEXT,
    
    -- Processing status
    processing_status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_error TEXT,
    
    -- Event metadata
    sender_login TEXT,
    ref_name TEXT, -- branch or tag name
    commit_sha TEXT,
    commit_message TEXT,
    
    -- Timestamps
    received_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    github_created_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for webhook events
CREATE INDEX IF NOT EXISTS idx_webhook_events_repo ON github_webhook_events(repository_id);
CREATE INDEX IF NOT EXISTS idx_webhook_events_type ON github_webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_events_status ON github_webhook_events(processing_status);
CREATE INDEX IF NOT EXISTS idx_webhook_events_received ON github_webhook_events(received_at);
CREATE INDEX IF NOT EXISTS idx_webhook_events_github_id ON github_webhook_events(github_event_id);

-- Documentation generation results
CREATE TABLE IF NOT EXISTS github_docs_generation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL REFERENCES github_repositories(id) ON DELETE CASCADE,
    webhook_event_id UUID REFERENCES github_webhook_events(id) ON DELETE SET NULL,
    
    -- Generation trigger
    trigger_type TEXT NOT NULL, -- webhook, manual, scheduled
    triggered_by_commit TEXT,
    affected_files TEXT[],
    
    -- Generation results
    generation_status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    generated_files TEXT[],
    updated_files TEXT[],
    
    -- AI analysis
    changes_detected JSONB,
    documentation_updates JSONB,
    ai_analysis_summary TEXT,
    
    -- Performance metrics
    processing_time_ms INTEGER,
    files_analyzed INTEGER DEFAULT 0,
    docs_generated INTEGER DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for docs generation
CREATE INDEX IF NOT EXISTS idx_docs_gen_repo ON github_docs_generation(repository_id);
CREATE INDEX IF NOT EXISTS idx_docs_gen_status ON github_docs_generation(generation_status);
CREATE INDEX IF NOT EXISTS idx_docs_gen_trigger ON github_docs_generation(trigger_type);
CREATE INDEX IF NOT EXISTS idx_docs_gen_created ON github_docs_generation(created_at);

-- Changelog generation results  
CREATE TABLE IF NOT EXISTS github_changelog_generation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL REFERENCES github_repositories(id) ON DELETE CASCADE,
    webhook_event_id UUID REFERENCES github_webhook_events(id) ON DELETE SET NULL,
    
    -- Generation trigger
    trigger_type TEXT NOT NULL, -- webhook, manual, scheduled
    commit_range_from TEXT,
    commit_range_to TEXT,
    
    -- Generation results
    generation_status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    changelog_file_path TEXT,
    entries_added INTEGER DEFAULT 0,
    
    -- Changelog analysis
    commits_analyzed JSONB,
    categorized_changes JSONB, -- features, fixes, breaking, etc.
    ai_summary TEXT,
    
    -- Performance metrics
    processing_time_ms INTEGER,
    commits_processed INTEGER DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for changelog generation
CREATE INDEX IF NOT EXISTS idx_changelog_gen_repo ON github_changelog_generation(repository_id);
CREATE INDEX IF NOT EXISTS idx_changelog_gen_status ON github_changelog_generation(generation_status);
CREATE INDEX IF NOT EXISTS idx_changelog_gen_trigger ON github_changelog_generation(trigger_type);
CREATE INDEX IF NOT EXISTS idx_changelog_gen_created ON github_changelog_generation(created_at);

-- Linting results
CREATE TABLE IF NOT EXISTS github_linting_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID NOT NULL REFERENCES github_repositories(id) ON DELETE CASCADE,
    webhook_event_id UUID REFERENCES github_webhook_events(id) ON DELETE SET NULL,
    
    -- Linting trigger
    trigger_type TEXT NOT NULL, -- webhook, manual, scheduled
    triggered_by_commit TEXT,
    files_linted TEXT[],
    
    -- Linting configuration
    linting_tools JSONB, -- which linters were used
    linting_rules JSONB, -- specific rules applied
    
    -- Linting results
    linting_status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    overall_result TEXT, -- passed, failed, warnings
    
    -- Issues found
    total_issues INTEGER DEFAULT 0,
    error_issues INTEGER DEFAULT 0,
    warning_issues INTEGER DEFAULT 0,
    info_issues INTEGER DEFAULT 0,
    
    -- Detailed results by language/tool
    results_by_language JSONB,
    results_by_file JSONB,
    
    -- Performance metrics
    processing_time_ms INTEGER,
    files_processed INTEGER DEFAULT 0,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for linting results
CREATE INDEX IF NOT EXISTS idx_linting_results_repo ON github_linting_results(repository_id);
CREATE INDEX IF NOT EXISTS idx_linting_results_status ON github_linting_results(linting_status);
CREATE INDEX IF NOT EXISTS idx_linting_results_trigger ON github_linting_results(trigger_type);
CREATE INDEX IF NOT EXISTS idx_linting_results_created ON github_linting_results(created_at);
CREATE INDEX IF NOT EXISTS idx_linting_results_overall ON github_linting_results(overall_result);

-- GitHub API rate limit tracking
CREATE TABLE IF NOT EXISTS github_api_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    
    -- Rate limit info
    requests_remaining INTEGER,
    requests_limit INTEGER,
    reset_timestamp TIMESTAMP WITH TIME ZONE,
    
    -- Request details
    response_status INTEGER,
    response_time_ms INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for API usage tracking
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON github_api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_created ON github_api_usage(created_at);

-- Automated update trigger function
CREATE OR REPLACE FUNCTION update_github_repo_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic updated_at
CREATE TRIGGER github_repositories_updated_at_trigger
    BEFORE UPDATE ON github_repositories
    FOR EACH ROW
    EXECUTE FUNCTION update_github_repo_updated_at();

-- View for repository monitoring dashboard
CREATE OR REPLACE VIEW github_monitoring_dashboard AS
SELECT 
    gr.id,
    gr.repository_name,
    gr.repository_url,
    gr.owner,
    gr.is_active,
    gr.auto_docs_enabled,
    gr.auto_changelog_enabled,
    gr.auto_linting_enabled,
    gr.last_processed_at,
    gr.total_events_processed,
    gr.docs_generated,
    gr.changelogs_generated,
    gr.lint_runs_completed,
    
    -- Recent activity counts (last 7 days)
    COALESCE(recent_events.event_count, 0) as recent_events_count,
    COALESCE(recent_docs.docs_count, 0) as recent_docs_count,
    COALESCE(recent_changelog.changelog_count, 0) as recent_changelog_count,
    COALESCE(recent_linting.lint_count, 0) as recent_lint_count,
    
    -- Latest event information
    latest_event.event_type as latest_event_type,
    latest_event.received_at as latest_event_time
    
FROM github_repositories gr

LEFT JOIN (
    SELECT repository_id, COUNT(*) as event_count
    FROM github_webhook_events 
    WHERE received_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    GROUP BY repository_id
) recent_events ON gr.id = recent_events.repository_id

LEFT JOIN (
    SELECT repository_id, COUNT(*) as docs_count
    FROM github_docs_generation 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    AND generation_status = 'completed'
    GROUP BY repository_id
) recent_docs ON gr.id = recent_docs.repository_id

LEFT JOIN (
    SELECT repository_id, COUNT(*) as changelog_count
    FROM github_changelog_generation 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    AND generation_status = 'completed'
    GROUP BY repository_id
) recent_changelog ON gr.id = recent_changelog.repository_id

LEFT JOIN (
    SELECT repository_id, COUNT(*) as lint_count
    FROM github_linting_results 
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
    AND linting_status = 'completed'
    GROUP BY repository_id
) recent_linting ON gr.id = recent_linting.repository_id

LEFT JOIN (
    SELECT DISTINCT ON (repository_id) 
        repository_id, event_type, received_at
    FROM github_webhook_events
    ORDER BY repository_id, received_at DESC
) latest_event ON gr.id = latest_event.repository_id

ORDER BY gr.repository_name;

-- Insert initial migration record
INSERT INTO archon_migrations (migration_name, executed_at) 
VALUES ('4_github_monitoring', CURRENT_TIMESTAMP)
ON CONFLICT (migration_name) DO NOTHING;