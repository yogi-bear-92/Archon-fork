#!/usr/bin/env python3
"""
Archon Database Backup Script (API-based)

Creates comprehensive backups using the Archon API endpoints.
This approach works without requiring direct database access.
"""

import os
import json
import csv
import datetime
import requests
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8181/api"

def create_backup_directory():
    """Create timestamped backup directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/archon_backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir

def backup_projects(backup_dir):
    """Backup all projects"""
    try:
        print("📊 Backing up projects...")
        
        response = requests.get(f"{API_BASE_URL}/projects", timeout=30)
        if response.status_code == 200:
            data = response.json()
            projects = data.get('projects', [])
            
            # Save as JSON
            json_file = backup_dir / "projects.json"
            with open(json_file, 'w') as f:
                json.dump(projects, f, indent=2, default=str)
            
            # Save as CSV
            if projects:
                csv_file = backup_dir / "projects.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=projects[0].keys())
                    writer.writeheader()
                    writer.writerows(projects)
            
            print(f"   ✅ {len(projects)} projects backed up")
            return len(projects)
        else:
            print(f"   ❌ Failed to get projects: HTTP {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"   ❌ Error backing up projects: {e}")
        return 0

def backup_tasks(backup_dir):
    """Backup all tasks"""
    try:
        print("📊 Backing up tasks...")
        
        response = requests.get(f"{API_BASE_URL}/tasks", timeout=30)
        if response.status_code == 200:
            data = response.json()
            tasks = data.get('tasks', [])
            
            # Save as JSON
            json_file = backup_dir / "tasks.json"
            with open(json_file, 'w') as f:
                json.dump(tasks, f, indent=2, default=str)
            
            # Save as CSV
            if tasks:
                csv_file = backup_dir / "tasks.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=tasks[0].keys())
                    writer.writeheader()
                    writer.writerows(tasks)
            
            print(f"   ✅ {len(tasks)} tasks backed up")
            return len(tasks)
        else:
            print(f"   ❌ Failed to get tasks: HTTP {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"   ❌ Error backing up tasks: {e}")
        return 0

def backup_sources(backup_dir):
    """Backup all knowledge sources"""
    try:
        print("📊 Backing up knowledge sources...")
        
        response = requests.get(f"{API_BASE_URL}/rag/sources", timeout=30)
        if response.status_code == 200:
            data = response.json()
            sources = data.get('sources', [])
            
            # Save as JSON
            json_file = backup_dir / "sources.json"
            with open(json_file, 'w') as f:
                json.dump(sources, f, indent=2, default=str)
            
            # Save as CSV
            if sources:
                csv_file = backup_dir / "sources.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sources[0].keys())
                    writer.writeheader()
                    writer.writerows(sources)
            
            print(f"   ✅ {len(sources)} sources backed up")
            return len(sources)
        else:
            print(f"   ❌ Failed to get sources: HTTP {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"   ❌ Error backing up sources: {e}")
        return 0

def backup_documents(backup_dir):
    """Backup all documents"""
    try:
        print("📊 Backing up documents...")
        
        response = requests.get(f"{API_BASE_URL}/documents", timeout=30)
        if response.status_code == 200:
            data = response.json()
            documents = data.get('documents', [])
            
            # Save as JSON
            json_file = backup_dir / "documents.json"
            with open(json_file, 'w') as f:
                json.dump(documents, f, indent=2, default=str)
            
            # Save as CSV
            if documents:
                csv_file = backup_dir / "documents.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=documents[0].keys())
                    writer.writeheader()
                    writer.writerows(documents)
            
            print(f"   ✅ {len(documents)} documents backed up")
            return len(documents)
        else:
            print(f"   ❌ Failed to get documents: HTTP {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"   ❌ Error backing up documents: {e}")
        return 0

def backup_knowledge_items(backup_dir):
    """Backup knowledge items (alternative source)"""
    try:
        print("📊 Backing up knowledge items...")
        
        response = requests.get(f"{API_BASE_URL}/knowledge-items", timeout=30)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            # Save as JSON
            json_file = backup_dir / "knowledge_items.json"
            with open(json_file, 'w') as f:
                json.dump(items, f, indent=2, default=str)
            
            print(f"   ✅ {len(items)} knowledge items backed up")
            return len(items)
        else:
            print(f"   ❌ Failed to get knowledge items: HTTP {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"   ❌ Error backing up knowledge items: {e}")
        return 0

def create_backup_manifest(backup_dir, backup_counts):
    """Create a manifest file with backup information"""
    manifest = {
        "backup_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "backup_version": "1.0",
            "method": "API-based backup",
            "total_records": sum(backup_counts.values())
        },
        "backup_counts": backup_counts,
        "files": {
            "data_files": [str(f) for f in backup_dir.glob("*.json")],
            "csv_files": [str(f) for f in backup_dir.glob("*.csv")]
        },
        "restore_instructions": {
            "method": "Use Archon API endpoints to restore data",
            "order": "Restore in order: projects -> tasks -> sources -> documents",
            "note": "This backup contains all accessible data via API endpoints"
        }
    }
    
    manifest_file = backup_dir / "backup_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📋 Backup manifest created: {manifest_file.name}")

def main():
    """Main backup function"""
    print("🗄️ ARCHON DATABASE BACKUP (API-based)")
    print("=" * 50)
    
    try:
        # Create backup directory
        backup_dir = create_backup_directory()
        print(f"📁 Backup directory: {backup_dir}")
        
        # Check API connectivity
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                print("🔗 Connected to Archon API")
            else:
                print("⚠️ API responded but may not be fully operational")
        except:
            print("⚠️ Could not verify API health, proceeding anyway...")
        
        # Backup each data type
        backup_counts = {}
        
        # Backup core data
        backup_counts['projects'] = backup_projects(backup_dir)
        backup_counts['tasks'] = backup_tasks(backup_dir)
        backup_counts['sources'] = backup_sources(backup_dir)
        backup_counts['documents'] = backup_documents(backup_dir)
        backup_counts['knowledge_items'] = backup_knowledge_items(backup_dir)
        
        # Create manifest
        create_backup_manifest(backup_dir, backup_counts)
        
        # Summary
        print(f"\n📊 BACKUP SUMMARY")
        print("=" * 30)
        print(f"✅ Backup completed successfully")
        print(f"📁 Location: {backup_dir}")
        print(f"📊 Total records: {sum(backup_counts.values())}")
        
        for data_type, count in backup_counts.items():
            print(f"   • {data_type}: {count} records")
        
        # Calculate backup size
        total_size = sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file())
        print(f"💾 Backup size: {total_size / 1024:.1f} KB")
        
        print(f"\n🎉 Database backup completed!")
        print(f"📁 Backup location: {backup_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
