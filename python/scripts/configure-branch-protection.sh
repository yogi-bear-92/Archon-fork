#!/bin/bash

# 🛡️ GitHub Swarm Branch Protection Configuration Script
# Configures intelligent branch protection rules for automated PR merging

set -e

echo "🛡️ Configuring GitHub Swarm Branch Protection..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get repository info
REPO_OWNER=$(gh repo view --json owner --jq '.owner.login')
REPO_NAME=$(gh repo view --json name --jq '.name')

echo -e "${BLUE}Repository: ${REPO_OWNER}/${REPO_NAME}${NC}"

# Function to configure branch protection
configure_branch_protection() {
    local branch=$1
    local is_main_branch=$2
    
    echo -e "\n${YELLOW}Configuring protection for branch: ${branch}${NC}"
    
    # Create branch protection configuration
    local protection_config='{
        "required_status_checks": {
            "strict": true,
            "contexts": [
                "auto-label",
                "commit-analysis", 
                "development-insights",
                "parallel-optimization-analysis (system_commands, sequential)",
                "parallel-optimization-analysis (system_commands, parallel_basic)",
                "parallel-optimization-analysis (system_commands, parallel_optimized)",
                "parallel-optimization-analysis (api_calls, sequential)",
                "parallel-optimization-analysis (api_calls, parallel_basic)",
                "parallel-optimization-analysis (api_calls, parallel_optimized)",
                "parallel-optimization-analysis (file_processing, sequential)",
                "parallel-optimization-analysis (file_processing, parallel_basic)",
                "parallel-optimization-analysis (file_processing, parallel_optimized)",
                "parallel-optimization-analysis (ansf_analysis, sequential)",
                "parallel-optimization-analysis (ansf_analysis, parallel_basic)",
                "parallel-optimization-analysis (ansf_analysis, parallel_optimized)",
                "performance-summary"
            ]
        },
        "enforce_admins": false,
        "required_pull_request_reviews": null,
        "restrictions": null,
        "allow_force_pushes": false,
        "allow_deletions": false,
        "block_creations": false,
        "required_conversation_resolution": false
    }'
    
    # For main branch, require stricter rules
    if [ "$is_main_branch" = true ]; then
        protection_config='{
            "required_status_checks": {
                "strict": true,
                "contexts": [
                    "auto-label",
                    "commit-analysis",
                    "development-insights",
                    "🛡️ GitHub Swarm Safety Validation / security-validation",
                    "🛡️ GitHub Swarm Safety Validation / code-quality-validation", 
                    "🛡️ GitHub Swarm Safety Validation / container-validation",
                    "🛡️ GitHub Swarm Safety Validation / safety-score-calculation"
                ]
            },
            "enforce_admins": false,
            "required_pull_request_reviews": {
                "dismiss_stale_reviews": true,
                "require_code_owner_reviews": false,
                "required_approving_review_count": 1,
                "require_last_push_approval": false
            },
            "restrictions": null,
            "allow_force_pushes": false,
            "allow_deletions": false,
            "block_creations": false,
            "required_conversation_resolution": true
        }'
    fi
    
    # Apply branch protection
    echo "$protection_config" | gh api \
        "repos/${REPO_OWNER}/${REPO_NAME}/branches/${branch}/protection" \
        --method PUT \
        --input - \
        --silent || {
            echo -e "${RED}❌ Failed to configure protection for branch: ${branch}${NC}"
            return 1
        }
    
    echo -e "${GREEN}✅ Branch protection configured for: ${branch}${NC}"
    
    # Display current protection status
    echo -e "${BLUE}Current protection settings:${NC}"
    gh api "repos/${REPO_OWNER}/${REPO_NAME}/branches/${branch}/protection" \
        --jq '{
            required_status_checks: .required_status_checks.contexts,
            enforce_admins: .enforce_admins.enabled,
            required_reviews: (.required_pull_request_reviews.required_approving_review_count // 0),
            allow_force_pushes: .allow_force_pushes.enabled,
            allow_deletions: .allow_deletions.enabled
        }' 2>/dev/null || echo "Unable to display protection details"
}

# Function to configure GitHub Swarm auto-merge rules  
configure_auto_merge_rules() {
    echo -e "\n${YELLOW}Configuring GitHub Swarm Auto-Merge Rules...${NC}"
    
    # Create auto-merge configuration file
    cat > .github/auto-merge-rules.yml << 'EOF'
# 🤖 GitHub Swarm Auto-Merge Rules Configuration

auto_merge:
  enabled: true
  
  # Target branches for auto-merge
  target_branches:
    - development
    - main
  
  # Required conditions for auto-merge
  conditions:
    # Status checks must pass
    status_checks: required
    
    # Branch must be up-to-date
    up_to_date: required
    
    # No merge conflicts
    mergeable: required
    
    # PR cannot be draft
    draft: false
    
    # Minimum safety score (0-100)
    min_safety_score: 80
    
    # Container health check
    container_health: required
    
    # Security validation
    security_validation: required

  # Branch-specific rules
  branch_rules:
    development:
      # Less strict for development
      min_safety_score: 75
      required_reviews: 0
      allow_force_merge: false
      
    main:
      # Stricter for main/production
      min_safety_score: 85
      required_reviews: 1
      allow_force_merge: false
      require_conversation_resolution: true

  # Safety overrides (use with caution)
  overrides:
    # Emergency merge (requires manual approval)
    emergency_merge: false
    
    # Skip container validation (not recommended)
    skip_container_validation: false
    
    # Skip security scan (not recommended)
    skip_security_scan: false

# GitHub Swarm coordination settings
swarm_settings:
  coordination_timeout: 300  # 5 minutes
  max_retry_attempts: 3
  agent_consensus_required: true
  
# Notification settings
notifications:
  on_merge_success: true
  on_merge_failure: true
  on_safety_validation: true
  
# Monitoring and metrics
monitoring:
  track_merge_performance: true
  export_safety_metrics: true
  retention_days: 30
EOF

    echo -e "${GREEN}✅ Auto-merge rules configuration created${NC}"
}

# Function to verify workflow files exist
verify_workflow_files() {
    echo -e "\n${YELLOW}Verifying GitHub Actions workflow files...${NC}"
    
    local workflows=(
        ".github/workflows/intelligent-pr-auto-merge.yml"
        ".github/workflows/pr-status-monitor.yml"
        ".github/workflows/safety-validation.yml"
        ".github/workflows/notification-system.yml"
    )
    
    local missing_workflows=()
    
    for workflow in "${workflows[@]}"; do
        if [ -f "$workflow" ]; then
            echo -e "${GREEN}✅ Found: ${workflow}${NC}"
        else
            echo -e "${RED}❌ Missing: ${workflow}${NC}"
            missing_workflows+=("$workflow")
        fi
    done
    
    if [ ${#missing_workflows[@]} -gt 0 ]; then
        echo -e "\n${RED}❌ Missing workflow files detected!${NC}"
        echo -e "${YELLOW}Please ensure all GitHub Swarm workflow files are committed.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ All workflow files are present${NC}"
}

# Function to test branch protection
test_branch_protection() {
    local branch=$1
    
    echo -e "\n${YELLOW}Testing branch protection for: ${branch}${NC}"
    
    # Get branch protection status
    local protection_status=$(gh api "repos/${REPO_OWNER}/${REPO_NAME}/branches/${branch}/protection" 2>/dev/null || echo "null")
    
    if [ "$protection_status" = "null" ]; then
        echo -e "${RED}❌ No branch protection found for: ${branch}${NC}"
        return 1
    fi
    
    # Check required status checks
    local required_checks=$(echo "$protection_status" | jq -r '.required_status_checks.contexts[]?' 2>/dev/null | wc -l)
    if [ "$required_checks" -gt 0 ]; then
        echo -e "${GREEN}✅ Required status checks configured: ${required_checks} checks${NC}"
    else
        echo -e "${YELLOW}⚠️ No required status checks configured${NC}"
    fi
    
    # Check other protection settings
    local enforce_admins=$(echo "$protection_status" | jq -r '.enforce_admins.enabled' 2>/dev/null)
    local allow_force_pushes=$(echo "$protection_status" | jq -r '.allow_force_pushes.enabled' 2>/dev/null)
    local allow_deletions=$(echo "$protection_status" | jq -r '.allow_deletions.enabled' 2>/dev/null)
    
    echo -e "${BLUE}Protection settings:${NC}"
    echo -e "  Enforce for admins: ${enforce_admins}"
    echo -e "  Allow force pushes: ${allow_force_pushes}"
    echo -e "  Allow deletions: ${allow_deletions}"
    
    echo -e "${GREEN}✅ Branch protection test completed for: ${branch}${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}🤖 GitHub Swarm Branch Protection Setup${NC}"
    echo -e "${BLUE}=======================================${NC}"
    
    # Check if gh CLI is authenticated
    if ! gh auth status >/dev/null 2>&1; then
        echo -e "${RED}❌ GitHub CLI is not authenticated${NC}"
        echo -e "${YELLOW}Please run: gh auth login${NC}"
        exit 1
    fi
    
    # Verify workflow files
    if ! verify_workflow_files; then
        echo -e "${RED}❌ Workflow verification failed${NC}"
        exit 1
    fi
    
    # Configure auto-merge rules
    configure_auto_merge_rules
    
    # Configure branch protection for development
    if configure_branch_protection "development" false; then
        test_branch_protection "development"
    fi
    
    # Configure branch protection for main (stricter rules)
    if configure_branch_protection "main" true; then
        test_branch_protection "main"
    fi
    
    echo -e "\n${GREEN}🎉 GitHub Swarm branch protection configuration completed!${NC}"
    echo -e "\n${BLUE}📋 Summary:${NC}"
    echo -e "✅ Branch protection rules configured"
    echo -e "✅ Auto-merge rules created"
    echo -e "✅ GitHub Swarm integration enabled"
    echo -e "✅ Safety validation requirements set"
    
    echo -e "\n${YELLOW}🚀 Next Steps:${NC}"
    echo -e "1. Commit and push the auto-merge rules configuration"
    echo -e "2. Test the workflow with a sample PR"
    echo -e "3. Monitor GitHub Actions for successful execution"
    echo -e "4. Review and adjust safety thresholds as needed"
    
    echo -e "\n${BLUE}📊 Monitoring:${NC}"
    echo -e "• GitHub Actions: https://github.com/${REPO_OWNER}/${REPO_NAME}/actions"
    echo -e "• Branch settings: https://github.com/${REPO_OWNER}/${REPO_NAME}/settings/branches"
    echo -e "• Pull requests: https://github.com/${REPO_OWNER}/${REPO_NAME}/pulls"
}

# Execute main function
main "$@"