#!/usr/bin/env python3
"""
Test script for duplicate project checking functionality
"""

import requests
import json

def test_duplicate_check():
    """Test the duplicate check functionality"""
    
    # Test 1: Check for existing project
    print("🔍 Testing duplicate check for existing project...")
    
    response = requests.get("http://localhost:8181/api/projects")
    if response.status_code == 200:
        projects = response.json().get("projects", [])
        print(f"📊 Found {len(projects)} existing projects")
        
        # Check for similar titles
        test_title = "Flow Nexus Project Structure Optimization"
        similar_projects = []
        
        for project in projects:
            if test_title.lower() in project.get("title", "").lower():
                similar_projects.append({
                    "id": project.get("id"),
                    "title": project.get("title"),
                    "github_repo": project.get("github_repo"),
                    "match_type": "title_similarity"
                })
        
        print(f"🎯 Found {len(similar_projects)} similar projects:")
        for project in similar_projects:
            print(f"  - {project['title']} (GitHub: {project['github_repo']})")
        
        # Test 2: Check GitHub repo duplicates
        test_github_repo = "https://github.com/yogi-bear-92/Archon-fork"
        github_duplicates = []
        
        for project in projects:
            if project.get("github_repo") == test_github_repo:
                github_duplicates.append({
                    "id": project.get("id"),
                    "title": project.get("title"),
                    "github_repo": project.get("github_repo"),
                    "match_type": "github_repo_exact"
                })
        
        print(f"🔗 Found {len(github_duplicates)} projects with same GitHub repo:")
        for project in github_duplicates:
            print(f"  - {project['title']} (ID: {project['id']})")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if len(similar_projects) > 0:
            print(f"  ⚠️  Found {len(similar_projects)} similar projects by title")
            print("  📝 Consider using a more specific title or check if you meant to update an existing project")
        
        if len(github_duplicates) > 0:
            print(f"  ⚠️  Found {len(github_duplicates)} projects with same GitHub repository")
            print("  🔍 This suggests the project might already exist")
        
        if len(similar_projects) == 0 and len(github_duplicates) == 0:
            print("  ✅ No similar projects found - safe to create new project")
            
    else:
        print(f"❌ Failed to fetch projects: {response.status_code}")

if __name__ == "__main__":
    test_duplicate_check()
