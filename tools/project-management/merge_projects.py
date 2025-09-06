#!/usr/bin/env python3
"""
Script to merge duplicate projects
"""

import requests
import json

def merge_projects():
    """Merge the duplicate projects"""
    
    primary_id = "4348d6e8-7baa-4206-9b54-fa1bbf9ccdca"  # Flow Nexus Project Structure Optimization
    duplicate_id = "b6ecf205-71ad-43aa-bf07-85de02743d33"  # Claude Flow Integration
    
    print(f"🔄 Merging projects...")
    print(f"  Primary: {primary_id}")
    print(f"  Duplicate: {duplicate_id}")
    
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
        
        # Merge project data
        print(f"🔄 Merging project data...")
        
        # Merge features
        primary_features = primary_project.get("features", {})
        duplicate_features = duplicate_project.get("features", {})
        merged_features = {**primary_features, **duplicate_features}
        
        # Merge data
        primary_data = primary_project.get("data", {})
        duplicate_data = duplicate_project.get("data", {})
        merged_data = {**primary_data, **duplicate_data}
        
        # Merge technical sources (they are objects, not simple strings)
        primary_tech_sources = primary_project.get("technical_sources", [])
        duplicate_tech_sources = duplicate_project.get("technical_sources", [])
        # Create a set of source IDs to avoid duplicates
        primary_tech_ids = {source.get("source_id") for source in primary_tech_sources if source.get("source_id")}
        duplicate_tech_ids = {source.get("source_id") for source in duplicate_tech_sources if source.get("source_id")}
        all_tech_ids = primary_tech_ids.union(duplicate_tech_ids)
        
        # Rebuild the technical sources list with unique sources
        merged_tech_sources = []
        seen_ids = set()
        for source in primary_tech_sources + duplicate_tech_sources:
            source_id = source.get("source_id")
            if source_id and source_id not in seen_ids:
                merged_tech_sources.append(source)
                seen_ids.add(source_id)
        
        # Merge business sources (same approach)
        primary_business_sources = primary_project.get("business_sources", [])
        duplicate_business_sources = duplicate_project.get("business_sources", [])
        merged_business_sources = []
        seen_business_ids = set()
        for source in primary_business_sources + duplicate_business_sources:
            source_id = source.get("source_id")
            if source_id and source_id not in seen_business_ids:
                merged_business_sources.append(source)
                seen_business_ids.add(source_id)
        
        # Update primary project
        update_response = requests.put(
            f"http://localhost:8181/api/projects/{primary_id}",
            json={
                "features": merged_features,
                "data": merged_data,
                "technical_sources": merged_tech_sources,
                "business_sources": merged_business_sources,
            }
        )
        
        if update_response.status_code == 200:
            print(f"✅ Successfully updated primary project with merged data")
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
            print(f"  Technical sources: {len(final_project.get('technical_sources', []))}")
            print(f"  Business sources: {len(final_project.get('business_sources', []))}")
            
            # Check tasks
            final_tasks_response = requests.get(f"http://localhost:8181/api/tasks?project_id={primary_id}")
            if final_tasks_response.status_code == 200:
                final_tasks = final_tasks_response.json().get('tasks', [])
                print(f"  Total tasks: {len(final_tasks)}")
                for task in final_tasks:
                    print(f"    - {task['title']} (Status: {task['status']})")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")

if __name__ == "__main__":
    merge_projects()
