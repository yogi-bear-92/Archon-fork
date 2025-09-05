#!/usr/bin/env python3
"""
Test script for auto-detection and merge functionality
"""

import requests
import json

def test_auto_detection():
    """Test the auto-detection functionality"""
    
    print("🔍 Testing GitHub auto-detection...")
    
    # Test auto-detection
    try:
        result = requests.get("http://localhost:8181/api/projects")
        if result.status_code == 200:
            projects = result.json().get("projects", [])
            print(f"📊 Found {len(projects)} existing projects")
            
            # Test auto-detection for a new project
            print("\n🔍 Testing auto-detection for new project...")
            
            # Simulate what the MCP tool would do
            import subprocess
            import os
            from pathlib import Path
            
            base_path = Path(os.getcwd()).resolve()
            github_repo = None
            confidence = "low"
            detection_method = "fallback"
            
            # Method 1: Check git remote
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=base_path,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    github_repo = result.stdout.strip()
                    # Convert SSH to HTTPS if needed
                    if github_repo.startswith("git@github.com:"):
                        github_repo = github_repo.replace("git@github.com:", "https://github.com/").replace(".git", "")
                    confidence = "high"
                    detection_method = "git_remote"
                    print(f"✅ Git remote detected: {github_repo}")
            except Exception as e:
                print(f"⚠️  Git remote detection failed: {e}")
            
            # Method 2: Check for .git directory and infer from directory name
            if not github_repo and (base_path / ".git").exists():
                try:
                    # Try to get user from git config
                    user_result = subprocess.run(
                        ["git", "config", "user.name"],
                        cwd=base_path,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if user_result.returncode == 0 and user_result.stdout.strip():
                        username = user_result.stdout.strip().lower().replace(" ", "-")
                        repo_name = base_path.name.lower().replace(" ", "-")
                        github_repo = f"https://github.com/{username}/{repo_name}"
                        confidence = "medium"
                        detection_method = "directory_structure"
                        print(f"✅ Directory structure detected: {github_repo}")
                except Exception as e:
                    print(f"⚠️  Directory structure detection failed: {e}")
            
            print(f"\n🎯 Auto-detection result:")
            print(f"  GitHub Repo: {github_repo}")
            print(f"  Confidence: {confidence}")
            print(f"  Method: {detection_method}")
            
    except Exception as e:
        print(f"❌ Error testing auto-detection: {e}")

def test_merge_preparation():
    """Test merge preparation by analyzing duplicate projects"""
    
    print("\n🔍 Analyzing duplicate projects for merge...")
    
    try:
        result = requests.get("http://localhost:8181/api/projects")
        if result.status_code == 200:
            projects = result.json().get("projects", [])
            
            # Group by GitHub repo
            github_groups = {}
            for p in projects:
                github = p.get('github_repo')
                if github:
                    if github not in github_groups:
                        github_groups[github] = []
                    github_groups[github].append(p)
            
            print(f"📊 Found {len(github_groups)} unique GitHub repositories")
            
            # Find duplicates
            duplicates_found = False
            for github, group in github_groups.items():
                if len(group) > 1:
                    duplicates_found = True
                    print(f"\n🔗 Duplicate group: {github}")
                    print(f"  Projects ({len(group)}):")
                    for i, project in enumerate(group):
                        print(f"    {i+1}. {project['title']} (ID: {project['id']})")
                    
                    # Get task counts
                    for project in group:
                        tasks_result = requests.get(f"http://localhost:8181/api/tasks?project_id={project['id']}")
                        if tasks_result.status_code == 200:
                            tasks = tasks_result.json().get('tasks', [])
                            print(f"      - Tasks: {len(tasks)}")
                    
                    # Suggest merge strategy
                    print(f"  💡 Suggested merge: Keep '{group[0]['title']}' as primary, merge others")
            
            if not duplicates_found:
                print("✅ No duplicate projects found")
                
    except Exception as e:
        print(f"❌ Error analyzing duplicates: {e}")

if __name__ == "__main__":
    test_auto_detection()
    test_merge_preparation()
