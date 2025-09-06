#!/usr/bin/env python3
"""
Script to delete projects without any tasks
"""

import requests
import json

def cleanup_empty_projects():
    """Delete all projects that have no tasks"""
    
    print("🧹 CLEANING UP EMPTY PROJECTS")
    print("=" * 50)
    
    try:
        # Get all projects
        projects_response = requests.get("http://localhost:8181/api/projects")
        if projects_response.status_code != 200:
            print(f"❌ Failed to fetch projects: {projects_response.status_code}")
            return
        
        projects = projects_response.json().get("projects", [])
        print(f"📊 Found {len(projects)} total projects")
        
        empty_projects = []
        
        # Check each project for tasks
        for project in projects:
            project_id = project['id']
            project_title = project['title']
            
            # Get tasks for this project
            tasks_response = requests.get(f"http://localhost:8181/api/tasks?project_id={project_id}")
            if tasks_response.status_code == 200:
                tasks = tasks_response.json().get('tasks', [])
                if len(tasks) == 0:
                    empty_projects.append(project)
                    print(f"❌ EMPTY: {project_title}")
                else:
                    print(f"✅ KEEP: {project_title} ({len(tasks)} tasks)")
            else:
                print(f"⚠️  ERROR: Could not check tasks for {project_title}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total projects: {len(projects)}")
        print(f"  Projects with tasks: {len(projects) - len(empty_projects)}")
        print(f"  Empty projects to delete: {len(empty_projects)}")
        
        if not empty_projects:
            print("\n✅ No empty projects found!")
            return
        
        # Delete empty projects
        print(f"\n🗑️  DELETING {len(empty_projects)} EMPTY PROJECTS...")
        deleted_count = 0
        failed_count = 0
        
        for project in empty_projects:
            project_id = project['id']
            project_title = project['title']
            
            print(f"  Deleting: {project_title}...", end=" ")
            
            delete_response = requests.delete(f"http://localhost:8181/api/projects/{project_id}")
            if delete_response.status_code == 200:
                print("✅ SUCCESS")
                deleted_count += 1
            else:
                print(f"❌ FAILED ({delete_response.status_code})")
                failed_count += 1
        
        print(f"\n🎉 CLEANUP COMPLETE!")
        print(f"  Successfully deleted: {deleted_count}")
        print(f"  Failed to delete: {failed_count}")
        
        # Verify final state
        print(f"\n🔍 VERIFYING FINAL STATE...")
        final_response = requests.get("http://localhost:8181/api/projects")
        if final_response.status_code == 200:
            final_projects = final_response.json().get("projects", [])
            print(f"  Remaining projects: {len(final_projects)}")
            
            # Check if any are still empty
            still_empty = 0
            for project in final_projects:
                project_id = project['id']
                tasks_response = requests.get(f"http://localhost:8181/api/tasks?project_id={project_id}")
                if tasks_response.status_code == 200:
                    tasks = tasks_response.json().get('tasks', [])
                    if len(tasks) == 0:
                        still_empty += 1
            
            if still_empty == 0:
                print("  ✅ All remaining projects have tasks!")
            else:
                print(f"  ⚠️  {still_empty} projects still empty")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    cleanup_empty_projects()
