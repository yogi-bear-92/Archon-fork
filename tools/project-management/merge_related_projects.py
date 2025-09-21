#!/usr/bin/env python3
"""
Script to merge the two related Flow Nexus projects
"""

import requests
import json

def merge_related_projects():
    """Merge the two related Flow Nexus projects"""
    
    primary_id = "4348d6e8-7baa-4206-9b54-fa1bbf9ccdca"  # Flow Nexus Project Structure Optimization
    duplicate_id = "76770723-2db1-480b-9213-e3ab64b3da55"  # Flow Nexus Swarm Integration
    
    print(f"🔄 Merging related Flow Nexus projects...")
    print(f"  Primary: Flow Nexus Project Structure Optimization")
    print(f"  Duplicate: Flow Nexus Swarm Integration")
    print(f"  Reason: Both work on same repository with overlapping concerns")
    
    try:
        # Get primary project
        primary_response = requests.get(f"http://localhost:8181/api/projects/{primary_id}")
        if primary_response.status_code != 200:
            print(f"❌ Failed to get primary project: {primary_response.status_code}")
            return
        
        primary_project = primary_response.json()
        print(f"✅ Primary project: {primary_project['title']}")
        
        # Get duplicate project
        duplicate_response = requests.get(f"http://localhost:8181/api/projects/{duplicate_id}")
        if duplicate_response.status_code != 200:
            print(f"❌ Failed to get duplicate project: {duplicate_response.status_code}")
            return
        
        duplicate_project = duplicate_response.json()
        print(f"✅ Duplicate project: {duplicate_project['title']}")
        
        # Get tasks from duplicate project
        tasks_response = requests.get(f"http://localhost:8181/api/tasks?project_id={duplicate_id}")
        if tasks_response.status_code == 200:
            tasks = tasks_response.json().get('tasks', [])
            print(f"📋 Found {len(tasks)} tasks in duplicate project")
            
            # Move tasks to primary project
            for task in tasks:
                print(f"  Moving task: {task['title']}")
                task_update = requests.put(
                    f"http://localhost:8181/api/tasks/{task['id']}",
                    json={"project_id": primary_id}
                )
                if task_update.status_code == 200:
                    print(f"    ✅ Moved successfully")
                else:
                    print(f"    ❌ Failed to move: {task_update.status_code}")
        
        # Get documents from duplicate project
        docs_response = requests.get(f"http://localhost:8181/api/documents?project_id={duplicate_id}")
        if docs_response.status_code == 200:
            docs = docs_response.json().get('documents', [])
            print(f"📄 Found {len(docs)} documents in duplicate project")
            
            # Move documents to primary project
            for doc in docs:
                print(f"  Moving document: {doc['title']}")
                doc_update = requests.put(
                    f"http://localhost:8181/api/documents/{doc['id']}",
                    json={"project_id": primary_id}
                )
                if doc_update.status_code == 200:
                    print(f"    ✅ Moved successfully")
                else:
                    print(f"    ❌ Failed to move: {doc_update.status_code}")
        
        # Update primary project title to reflect merged scope
        print(f"🔄 Updating primary project title to reflect merged scope...")
        update_response = requests.put(
            f"http://localhost:8181/api/projects/{primary_id}",
            json={
                "title": "Flow Nexus Integration & Structure Optimization",
                "description": "Comprehensive Flow Nexus integration project combining swarm coordination, architectural improvements, and performance optimization. Includes ANSF workflow implementation, multi-agent coordination, and project structure reorganization."
            }
        )
        
        if update_response.status_code == 200:
            print(f"✅ Successfully updated primary project title and description")
        else:
            print(f"❌ Failed to update primary project: {update_response.status_code}")
        
        # Delete duplicate project
        print(f"🗑️  Deleting duplicate project...")
        delete_response = requests.delete(f"http://localhost:8181/api/projects/{duplicate_id}")
        if delete_response.status_code == 200:
            print(f"✅ Successfully deleted duplicate project")
        else:
            print(f"❌ Failed to delete duplicate project: {delete_response.status_code}")
        
        print(f"\n🎉 Merge completed!")
        
        # Verify results
        print(f"\n📊 Verifying merge results...")
        final_response = requests.get(f"http://localhost:8181/api/projects/{primary_id}")
        if final_response.status_code == 200:
            final_project = final_response.json()
            print(f"✅ Final project: {final_project['title']}")
            
            # Check tasks
            final_tasks_response = requests.get(f"http://localhost:8181/api/tasks?project_id={primary_id}")
            if final_tasks_response.status_code == 200:
                final_tasks = final_tasks_response.json().get('tasks', [])
                print(f"  Total tasks: {len(final_tasks)}")
                
                # Group by feature
                features = {}
                for task in final_tasks:
                    feature = task.get('feature', 'unassigned')
                    if feature not in features:
                        features[feature] = []
                    features[feature].append(task['title'])
                
                print(f"  Tasks by feature:")
                for feature, task_list in features.items():
                    print(f"    {feature}: {len(task_list)} tasks")
                    for task_title in task_list:
                        print(f"      - {task_title}")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")

if __name__ == "__main__":
    merge_related_projects()
