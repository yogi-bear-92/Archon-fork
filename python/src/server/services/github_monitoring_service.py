"""
GitHub Monitoring Service

This service monitors GitHub repositories for changes and automatically handles:
- Documentation generation from code changes
- Changelog generation from commits
- Automatic linting and code formatting
- Repository maintenance tasks

Excludes CI/CD and test runners as per requirements.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from src.server.config.logfire_config import get_logger, safe_span, safe_set_attribute
from src.server.services.llm_provider_service import get_llm_client
from .client_manager import get_supabase_client

logger = get_logger(__name__)


class GitHubEventType(str, Enum):
    """Types of GitHub events we monitor."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_REVIEW = "pull_request_review"
    ISSUES = "issues"
    RELEASE = "release"
    CREATE = "create"  # Branch/tag creation
    DELETE = "delete"  # Branch/tag deletion


class MonitoringAction(str, Enum):
    """Types of automatic actions we can take."""
    GENERATE_DOCS = "generate_docs"
    UPDATE_CHANGELOG = "update_changelog"
    RUN_LINTER = "run_linter"
    FORMAT_CODE = "format_code"
    UPDATE_README = "update_readme"
    GENERATE_API_DOCS = "generate_api_docs"
    UPDATE_BADGES = "update_badges"


@dataclass
class GitHubRepository:
    """GitHub repository configuration."""
    owner: str
    name: str
    full_name: str
    clone_url: str
    default_branch: str = "main"
    monitored_paths: List[str] = None
    excluded_paths: List[str] = None
    enabled_actions: List[MonitoringAction] = None


class GitHubEvent(BaseModel):
    """GitHub webhook event data."""
    event_type: GitHubEventType
    repository: str = Field(..., description="Repository full name (owner/repo)")
    ref: Optional[str] = Field(None, description="Git reference (branch/tag)")
    commits: List[Dict[str, Any]] = Field(default_factory=list)
    pull_request: Optional[Dict[str, Any]] = None
    sender: Dict[str, Any] = Field(default_factory=dict)
    action: Optional[str] = None  # opened, closed, synchronize, etc.
    created_at: datetime = Field(default_factory=datetime.now)


class ChangeAnalysis(BaseModel):
    """Analysis of changes in a commit or PR."""
    files_changed: List[str]
    files_added: List[str]
    files_deleted: List[str]
    lines_added: int
    lines_removed: int
    has_code_changes: bool
    has_doc_changes: bool
    has_config_changes: bool
    affected_languages: Set[str]
    needs_docs_update: bool
    needs_changelog_update: bool
    needs_linting: bool


class GitHubMonitoringService:
    """
    Service for monitoring GitHub repositories and automating maintenance tasks.
    """
    
    def __init__(self):
        self.monitored_repos: Dict[str, GitHubRepository] = {}
        self.event_queue: List[GitHubEvent] = []
        self.processing_lock = asyncio.Lock()
        
        # Documentation generation prompt
        self.doc_generation_prompt = """
You are an expert technical writer. Generate comprehensive documentation for the following code changes.

REPOSITORY: {repository}
BRANCH: {branch}
COMMIT: {commit_sha}

CHANGED FILES:
{changed_files}

FILE CONTENTS:
{file_contents}

INSTRUCTIONS:
1. Analyze the code changes and determine what documentation needs to be updated
2. Generate or update relevant documentation sections
3. Focus on API documentation, function/class descriptions, and usage examples
4. Include proper markdown formatting
5. Consider the target audience (developers using this code)

Generate documentation in the following format:
{{
    "documentation_updates": [
        {{
            "file": "docs/api/authentication.md",
            "title": "Authentication API",
            "content": "# Authentication API\\n\\nThis module provides...",
            "type": "api_docs",
            "reason": "Added new JWT authentication methods"
        }}
    ],
    "readme_updates": {{
        "section": "API Reference",
        "content": "## Authentication\\n\\nNew JWT support...",
        "action": "append"
    }},
    "needs_attention": [
        "Manual review needed for security documentation",
        "Consider adding architecture diagram"
    ]
}}
"""

        # Changelog generation prompt
        self.changelog_prompt = """
Generate a changelog entry for the following commits.

REPOSITORY: {repository}
DATE_RANGE: {date_range}
COMMITS: {commits}

COMMIT DETAILS:
{commit_details}

INSTRUCTIONS:
1. Analyze commits and group changes by type (Features, Bug Fixes, Changes, etc.)
2. Write clear, user-focused descriptions
3. Follow conventional changelog format
4. Include breaking changes if any
5. Focus on user-visible changes, not internal refactoring

Generate changelog in this format:
{{
    "version": "v1.2.0",
    "date": "2024-01-15",
    "sections": {{
        "Added": [
            "JWT authentication support for secure API access",
            "New user profile management endpoints"
        ],
        "Changed": [
            "Improved error handling in authentication flow"
        ],
        "Fixed": [
            "Fixed memory leak in session management"
        ],
        "Security": [
            "Enhanced password hashing algorithm"
        ]
    }},
    "breaking_changes": [
        "Authentication tokens now expire after 24 hours instead of 7 days"
    ]
}}
"""

        # Linting configuration
        self.linting_config = {
            "python": {
                "tools": ["ruff", "black", "mypy"],
                "commands": {
                    "ruff": "ruff check --fix .",
                    "black": "black .",
                    "mypy": "mypy --check-untyped-defs ."
                }
            },
            "javascript": {
                "tools": ["eslint", "prettier"],
                "commands": {
                    "eslint": "eslint --fix .",
                    "prettier": "prettier --write ."
                }
            },
            "typescript": {
                "tools": ["eslint", "prettier", "tsc"],
                "commands": {
                    "eslint": "eslint --fix .",
                    "prettier": "prettier --write .",
                    "tsc": "tsc --noEmit"
                }
            }
        }

    async def add_repository(
        self,
        owner: str,
        name: str,
        github_token: Optional[str] = None,
        enabled_actions: Optional[List[MonitoringAction]] = None
    ) -> Dict[str, Any]:
        """Add a repository to monitoring."""
        with safe_span("add_repository_monitoring") as span:
            safe_set_attribute(span, "repository", f"{owner}/{name}")
            
            try:
                # Get repository information from GitHub API
                repo_info = await self._fetch_repository_info(owner, name, github_token)
                
                if not repo_info:
                    return {
                        "success": False,
                        "error": f"Could not fetch repository information for {owner}/{name}"
                    }
                
                # Create repository configuration
                repo = GitHubRepository(
                    owner=owner,
                    name=name,
                    full_name=f"{owner}/{name}",
                    clone_url=repo_info.get("clone_url", ""),
                    default_branch=repo_info.get("default_branch", "main"),
                    enabled_actions=enabled_actions or [
                        MonitoringAction.GENERATE_DOCS,
                        MonitoringAction.UPDATE_CHANGELOG,
                        MonitoringAction.RUN_LINTER
                    ]
                )
                
                self.monitored_repos[repo.full_name] = repo
                
                # Store in database
                await self._store_repository_config(repo)
                
                safe_set_attribute(span, "actions_enabled", len(repo.enabled_actions))
                logger.info(f"✅ Added repository to monitoring: {repo.full_name}")
                
                return {
                    "success": True,
                    "repository": repo.full_name,
                    "enabled_actions": [action.value for action in repo.enabled_actions],
                    "default_branch": repo.default_branch
                }
                
            except Exception as e:
                logger.error(f"❌ Error adding repository to monitoring: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    async def process_webhook_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming GitHub webhook event."""
        with safe_span("process_github_webhook") as span:
            try:
                # Parse webhook event
                event = await self._parse_webhook_event(event_data)
                if not event:
                    return {"success": False, "error": "Could not parse webhook event"}
                
                safe_set_attribute(span, "event_type", event.event_type.value)
                safe_set_attribute(span, "repository", event.repository)
                
                # Check if repository is monitored
                if event.repository not in self.monitored_repos:
                    logger.warning(f"⚠️ Received event for unmonitored repository: {event.repository}")
                    return {
                        "success": True,
                        "message": "Repository not monitored",
                        "repository": event.repository
                    }
                
                # Add to processing queue
                self.event_queue.append(event)
                
                # Process event asynchronously
                asyncio.create_task(self._process_event_async(event))
                
                logger.info(f"🔄 Queued GitHub event: {event.event_type.value} for {event.repository}")
                
                return {
                    "success": True,
                    "event_type": event.event_type.value,
                    "repository": event.repository,
                    "queued_at": event.created_at.isoformat()
                }
                
            except Exception as e:
                logger.error(f"❌ Error processing webhook event: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    async def _process_event_async(self, event: GitHubEvent):
        """Process GitHub event asynchronously."""
        async with self.processing_lock:
            with safe_span("process_github_event_async") as span:
                safe_set_attribute(span, "event_type", event.event_type.value)
                safe_set_attribute(span, "repository", event.repository)
                
                try:
                    repo_config = self.monitored_repos[event.repository]
                    
                    # Analyze changes based on event type
                    if event.event_type == GitHubEventType.PUSH:
                        await self._handle_push_event(event, repo_config)
                    elif event.event_type == GitHubEventType.PULL_REQUEST:
                        await self._handle_pull_request_event(event, repo_config)
                    elif event.event_type == GitHubEventType.RELEASE:
                        await self._handle_release_event(event, repo_config)
                    
                    safe_set_attribute(span, "processing_success", True)
                    logger.info(f"✅ Processed GitHub event: {event.event_type.value} for {event.repository}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing GitHub event: {e}")
                    safe_set_attribute(span, "processing_success", False)
                    safe_set_attribute(span, "error", str(e))

    async def _handle_push_event(self, event: GitHubEvent, repo_config: GitHubRepository):
        """Handle push events (commits to branches)."""
        if not event.commits:
            return
        
        # Analyze changes in the push
        analysis = await self._analyze_changes(event.commits, repo_config)
        
        actions_taken = []
        
        # Generate documentation if code changes detected
        if (MonitoringAction.GENERATE_DOCS in repo_config.enabled_actions and 
            analysis.needs_docs_update):
            doc_result = await self._generate_documentation(event, analysis, repo_config)
            if doc_result.get("success"):
                actions_taken.append("documentation_generated")
        
        # Update changelog for main branch pushes
        if (event.ref == f"refs/heads/{repo_config.default_branch}" and
            MonitoringAction.UPDATE_CHANGELOG in repo_config.enabled_actions):
            changelog_result = await self._update_changelog(event, analysis, repo_config)
            if changelog_result.get("success"):
                actions_taken.append("changelog_updated")
        
        # Run linting on code changes
        if (MonitoringAction.RUN_LINTER in repo_config.enabled_actions and 
            analysis.needs_linting):
            lint_result = await self._run_linting(event, analysis, repo_config)
            if lint_result.get("success"):
                actions_taken.append("linting_completed")
        
        logger.info(f"🔧 Push event actions taken for {repo_config.full_name}: {', '.join(actions_taken) or 'none'}")

    async def _handle_pull_request_event(self, event: GitHubEvent, repo_config: GitHubRepository):
        """Handle pull request events."""
        if not event.pull_request:
            return
        
        action = event.action
        pr_number = event.pull_request.get("number")
        
        actions_taken = []
        
        # On PR opened or synchronized, run linting and generate docs preview
        if action in ["opened", "synchronize"]:
            # Get PR changes
            pr_changes = await self._get_pull_request_changes(event, repo_config)
            
            if pr_changes:
                analysis = await self._analyze_changes(pr_changes, repo_config)
                
                # Run linting on PR changes
                if (MonitoringAction.RUN_LINTER in repo_config.enabled_actions and 
                    analysis.needs_linting):
                    lint_result = await self._run_pr_linting(event, analysis, repo_config)
                    if lint_result.get("success"):
                        actions_taken.append("pr_linting")
                
                # Generate documentation preview
                if (MonitoringAction.GENERATE_DOCS in repo_config.enabled_actions and 
                    analysis.needs_docs_update):
                    docs_result = await self._generate_docs_preview(event, analysis, repo_config)
                    if docs_result.get("success"):
                        actions_taken.append("docs_preview")
        
        logger.info(f"🔧 PR #{pr_number} actions taken for {repo_config.full_name}: {', '.join(actions_taken) or 'none'}")

    async def _handle_release_event(self, event: GitHubEvent, repo_config: GitHubRepository):
        """Handle release events."""
        if event.action == "published":
            # Update documentation for new release
            if MonitoringAction.GENERATE_DOCS in repo_config.enabled_actions:
                await self._update_release_documentation(event, repo_config)

    async def _analyze_changes(self, commits: List[Dict], repo_config: GitHubRepository) -> ChangeAnalysis:
        """Analyze changes in commits to determine what actions are needed."""
        files_changed = set()
        files_added = set()
        files_deleted = set()
        total_additions = 0
        total_deletions = 0
        
        for commit in commits:
            # GitHub API provides added/removed/modified files
            if "added" in commit:
                files_added.update(commit["added"])
            if "removed" in commit:
                files_deleted.update(commit["removed"])
            if "modified" in commit:
                files_changed.update(commit["modified"])
            
            # Count line changes (if available)
            total_additions += commit.get("stats", {}).get("additions", 0)
            total_deletions += commit.get("stats", {}).get("deletions", 0)
        
        all_changed = files_changed | files_added | files_deleted
        
        # Determine file types and languages
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.rb'}
        doc_extensions = {'.md', '.rst', '.txt', '.adoc'}
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
        
        has_code_changes = any(Path(f).suffix.lower() in code_extensions for f in all_changed)
        has_doc_changes = any(Path(f).suffix.lower() in doc_extensions for f in all_changed)
        has_config_changes = any(Path(f).suffix.lower() in config_extensions for f in all_changed)
        
        # Detect programming languages
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        affected_languages = set()
        for file in all_changed:
            ext = Path(file).suffix.lower()
            if ext in language_map:
                affected_languages.add(language_map[ext])
        
        # Determine what actions are needed
        needs_docs_update = (
            has_code_changes or 
            any('src/' in f or 'lib/' in f for f in all_changed) or
            any('api' in f.lower() for f in all_changed)
        )
        
        needs_changelog_update = has_code_changes or len(all_changed) > 2
        needs_linting = has_code_changes
        
        return ChangeAnalysis(
            files_changed=list(files_changed),
            files_added=list(files_added),
            files_deleted=list(files_deleted),
            lines_added=total_additions,
            lines_removed=total_deletions,
            has_code_changes=has_code_changes,
            has_doc_changes=has_doc_changes,
            has_config_changes=has_config_changes,
            affected_languages=affected_languages,
            needs_docs_update=needs_docs_update,
            needs_changelog_update=needs_changelog_update,
            needs_linting=needs_linting
        )

    async def _generate_documentation(
        self, 
        event: GitHubEvent, 
        analysis: ChangeAnalysis, 
        repo_config: GitHubRepository
    ) -> Dict[str, Any]:
        """Generate documentation for code changes."""
        try:
            # Get file contents for changed code files
            code_files = [f for f in analysis.files_changed + analysis.files_added 
                         if any(f.endswith(ext) for ext in ['.py', '.js', '.ts'])]
            
            if not code_files:
                return {"success": True, "message": "No code files to document"}
            
            # Limit to first 10 files to avoid token limits
            code_files = code_files[:10]
            
            file_contents = {}
            for file_path in code_files:
                content = await self._get_file_content(repo_config, file_path, event.ref)
                if content:
                    file_contents[file_path] = content[:2000]  # Truncate for API limits
            
            if not file_contents:
                return {"success": False, "error": "Could not retrieve file contents"}
            
            # Generate documentation using AI
            prompt = self.doc_generation_prompt.format(
                repository=repo_config.full_name,
                branch=event.ref.replace('refs/heads/', '') if event.ref else repo_config.default_branch,
                commit_sha=event.commits[0].get("id", "") if event.commits else "",
                changed_files="\n".join(code_files),
                file_contents=json.dumps(file_contents, indent=2)
            )
            
            async with get_llm_client() as client:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert technical writer. Generate comprehensive documentation in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                doc_result = json.loads(response.choices[0].message.content)
                
                # Store generated documentation
                await self._store_generated_docs(repo_config, doc_result, event)
                
                logger.info(f"📚 Generated documentation for {repo_config.full_name}: {len(doc_result.get('documentation_updates', []))} files")
                
                return {
                    "success": True,
                    "documentation_files": len(doc_result.get('documentation_updates', [])),
                    "needs_attention": doc_result.get('needs_attention', [])
                }
                
        except Exception as e:
            logger.error(f"❌ Error generating documentation: {e}")
            return {"success": False, "error": str(e)}

    async def _update_changelog(
        self, 
        event: GitHubEvent, 
        analysis: ChangeAnalysis, 
        repo_config: GitHubRepository
    ) -> Dict[str, Any]:
        """Update changelog based on commits."""
        try:
            if not event.commits:
                return {"success": True, "message": "No commits to process"}
            
            # Prepare commit details for analysis
            commit_details = []
            for commit in event.commits[-10:]:  # Last 10 commits
                commit_details.append({
                    "sha": commit.get("id", ""),
                    "message": commit.get("message", ""),
                    "author": commit.get("author", {}).get("name", ""),
                    "timestamp": commit.get("timestamp", ""),
                    "added": commit.get("added", []),
                    "modified": commit.get("modified", []),
                    "removed": commit.get("removed", [])
                })
            
            # Generate changelog entry using AI
            prompt = self.changelog_prompt.format(
                repository=repo_config.full_name,
                date_range=datetime.now().strftime("%Y-%m-%d"),
                commits=len(commit_details),
                commit_details=json.dumps(commit_details, indent=2)
            )
            
            async with get_llm_client() as client:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert at writing changelogs. Generate structured changelog entries in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                changelog_result = json.loads(response.choices[0].message.content)
                
                # Store changelog entry
                await self._store_changelog_entry(repo_config, changelog_result, event)
                
                sections = changelog_result.get("sections", {})
                total_entries = sum(len(entries) for entries in sections.values())
                
                logger.info(f"📝 Generated changelog for {repo_config.full_name}: {total_entries} entries")
                
                return {
                    "success": True,
                    "changelog_entries": total_entries,
                    "sections": list(sections.keys()),
                    "breaking_changes": len(changelog_result.get("breaking_changes", []))
                }
                
        except Exception as e:
            logger.error(f"❌ Error updating changelog: {e}")
            return {"success": False, "error": str(e)}

    async def _run_linting(
        self, 
        event: GitHubEvent, 
        analysis: ChangeAnalysis, 
        repo_config: GitHubRepository
    ) -> Dict[str, Any]:
        """Run linting on changed files."""
        try:
            lint_results = {}
            
            for language in analysis.affected_languages:
                if language in self.linting_config:
                    config = self.linting_config[language]
                    
                    # Simulate linting (in real implementation, would run actual tools)
                    # For now, just log what would be run
                    tools_run = []
                    for tool in config["tools"]:
                        if tool in config["commands"]:
                            # In real implementation: subprocess.run(config["commands"][tool])
                            logger.info(f"🔧 Would run {tool}: {config['commands'][tool]}")
                            tools_run.append(tool)
                    
                    lint_results[language] = {
                        "tools_run": tools_run,
                        "status": "completed",
                        "issues_found": 0,  # Would be actual count
                        "issues_fixed": 0   # Would be actual count
                    }
            
            await self._store_lint_results(repo_config, lint_results, event)
            
            total_tools = sum(len(result["tools_run"]) for result in lint_results.values())
            logger.info(f"🔍 Ran linting for {repo_config.full_name}: {total_tools} tools across {len(lint_results)} languages")
            
            return {
                "success": True,
                "languages_processed": list(lint_results.keys()),
                "total_tools_run": total_tools,
                "results": lint_results
            }
            
        except Exception as e:
            logger.error(f"❌ Error running linting: {e}")
            return {"success": False, "error": str(e)}

    # Helper methods for GitHub API interactions and data storage

    async def _fetch_repository_info(self, owner: str, name: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch repository information from GitHub API."""
        try:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.github.com/repos/{owner}/{name}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"⚠️ GitHub API returned {response.status_code} for {owner}/{name}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error fetching repository info: {e}")
            return None

    async def _parse_webhook_event(self, event_data: Dict[str, Any]) -> Optional[GitHubEvent]:
        """Parse GitHub webhook event data."""
        try:
            # Extract event type from headers (would be passed in real webhook)
            event_type = event_data.get("event_type", "push")
            
            return GitHubEvent(
                event_type=GitHubEventType(event_type),
                repository=event_data.get("repository", {}).get("full_name", ""),
                ref=event_data.get("ref"),
                commits=event_data.get("commits", []),
                pull_request=event_data.get("pull_request"),
                sender=event_data.get("sender", {}),
                action=event_data.get("action")
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing webhook event: {e}")
            return None

    async def _get_file_content(self, repo_config: GitHubRepository, file_path: str, ref: Optional[str]) -> Optional[str]:
        """Get file content from GitHub repository."""
        # In real implementation, would use GitHub API to fetch file content
        # For now, return placeholder
        return f"# Content of {file_path}\n# This would be actual file content from GitHub API"

    async def _store_repository_config(self, repo: GitHubRepository):
        """Store repository configuration in database."""
        try:
            client = get_supabase_client()
            
            await client.table("github_repositories").upsert({
                "full_name": repo.full_name,
                "owner": repo.owner,
                "name": repo.name,
                "clone_url": repo.clone_url,
                "default_branch": repo.default_branch,
                "enabled_actions": [action.value for action in repo.enabled_actions],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }).execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Could not store repository config: {e}")

    async def _store_generated_docs(self, repo_config: GitHubRepository, doc_result: Dict, event: GitHubEvent):
        """Store generated documentation results."""
        try:
            client = get_supabase_client()
            
            await client.table("github_documentation").insert({
                "repository": repo_config.full_name,
                "event_type": event.event_type.value,
                "commit_sha": event.commits[0].get("id") if event.commits else None,
                "documentation_updates": doc_result.get("documentation_updates", []),
                "readme_updates": doc_result.get("readme_updates"),
                "needs_attention": doc_result.get("needs_attention", []),
                "created_at": datetime.now().isoformat()
            }).execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Could not store documentation results: {e}")

    async def _store_changelog_entry(self, repo_config: GitHubRepository, changelog_result: Dict, event: GitHubEvent):
        """Store changelog entry."""
        try:
            client = get_supabase_client()
            
            await client.table("github_changelogs").insert({
                "repository": repo_config.full_name,
                "version": changelog_result.get("version"),
                "date": changelog_result.get("date"),
                "sections": changelog_result.get("sections", {}),
                "breaking_changes": changelog_result.get("breaking_changes", []),
                "commit_range": [c.get("id") for c in event.commits] if event.commits else [],
                "created_at": datetime.now().isoformat()
            }).execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Could not store changelog entry: {e}")

    async def _store_lint_results(self, repo_config: GitHubRepository, lint_results: Dict, event: GitHubEvent):
        """Store linting results."""
        try:
            client = get_supabase_client()
            
            await client.table("github_lint_results").insert({
                "repository": repo_config.full_name,
                "event_type": event.event_type.value,
                "commit_sha": event.commits[0].get("id") if event.commits else None,
                "lint_results": lint_results,
                "created_at": datetime.now().isoformat()
            }).execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Could not store lint results: {e}")

    async def get_monitored_repositories(self) -> List[Dict[str, Any]]:
        """Get all monitored repositories."""
        return [
            {
                "full_name": repo.full_name,
                "owner": repo.owner,
                "name": repo.name,
                "default_branch": repo.default_branch,
                "enabled_actions": [action.value for action in repo.enabled_actions]
            }
            for repo in self.monitored_repos.values()
        ]

    async def remove_repository(self, full_name: str) -> bool:
        """Remove repository from monitoring."""
        if full_name in self.monitored_repos:
            del self.monitored_repos[full_name]
            logger.info(f"🗑️ Removed repository from monitoring: {full_name}")
            return True
        return False

    # Placeholder methods for unimplemented features
    async def _run_pr_linting(self, event: GitHubEvent, analysis: ChangeAnalysis, repo_config: GitHubRepository):
        """Run linting specifically for pull requests."""
        return {"success": True, "message": "PR linting would be implemented here"}

    async def _generate_docs_preview(self, event: GitHubEvent, analysis: ChangeAnalysis, repo_config: GitHubRepository):
        """Generate documentation preview for pull requests."""
        return {"success": True, "message": "Docs preview would be implemented here"}

    async def _update_release_documentation(self, event: GitHubEvent, repo_config: GitHubRepository):
        """Update documentation for new releases."""
        return {"success": True, "message": "Release docs update would be implemented here"}

    async def _get_pull_request_changes(self, event: GitHubEvent, repo_config: GitHubRepository):
        """Get changes in a pull request."""
        return []  # Would return actual PR changes


# Global instance
_github_monitoring_service: Optional[GitHubMonitoringService] = None


def get_github_monitoring_service() -> GitHubMonitoringService:
    """Get global GitHub monitoring service instance."""
    global _github_monitoring_service
    
    if _github_monitoring_service is None:
        _github_monitoring_service = GitHubMonitoringService()
    
    return _github_monitoring_service