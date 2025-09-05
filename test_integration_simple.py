#!/usr/bin/env python3
"""
Simple integration test for AI tagging functionality.

This script checks if the AI tagging components are properly integrated
without requiring the full Python environment.
"""

import os
import sys


def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False


def check_import_in_file(file_path, import_statement, description):
    """Check if an import statement exists in a file."""
    if not os.path.exists(file_path):
        print(f"❌ {description}: File not found - {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            if import_statement in content:
                print(f"✅ {description}: {import_statement}")
                return True
            else:
                print(f"❌ {description}: {import_statement} (NOT FOUND)")
                return False
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False


def main():
    """Run integration checks."""
    print("🔍 AI Tagging Integration Check\n")
    
    # Check if files exist
    print("📁 Checking file existence:")
    print("-" * 40)
    
    files_to_check = [
        ("python/src/server/services/ai_tag_generation_service.py", "AI Tag Generation Service"),
        ("python/src/server/services/ai_tagging_background_service.py", "AI Tagging Background Service"),
        ("python/src/server/api_routes/ai_tagging_api.py", "AI Tagging API Routes"),
    ]
    
    file_results = []
    for file_path, description in files_to_check:
        result = check_file_exists(file_path, description)
        file_results.append(result)
    
    print()
    
    # Check imports
    print("🔗 Checking import integration:")
    print("-" * 40)
    
    import_checks = [
        ("python/src/server/main.py", "from .api_routes.ai_tagging_api import ai_tagging_router", "AI Tagging Router Import"),
        ("python/src/server/main.py", "app.include_router(ai_tagging_router)", "AI Tagging Router Registration"),
        ("python/src/server/services/crawling/document_storage_operations.py", "from ..ai_tag_generation_service import get_ai_tag_service", "AI Tag Service Import"),
        ("python/src/server/services/serena_coordination_hooks.py", "from .ai_tagging_background_service import get_ai_tagging_background_service", "AI Tagging Background Service Import"),
    ]
    
    import_results = []
    for file_path, import_statement, description in import_checks:
        result = check_import_in_file(file_path, import_statement, description)
        import_results.append(result)
    
    print()
    
    # Check API endpoint registration
    print("🌐 Checking API endpoint integration:")
    print("-" * 40)
    
    api_checks = [
        ("python/src/server/api_routes/ai_tagging_api.py", "ai_tagging_router = APIRouter(prefix=\"/api/ai-tagging\"", "API Router Definition"),
        ("python/src/server/api_routes/ai_tagging_api.py", "@ai_tagging_router.post(\"/generate-tags\")", "Generate Tags Endpoint"),
        ("python/src/server/api_routes/ai_tagging_api.py", "@ai_tagging_router.get(\"/status\")", "Status Endpoint"),
    ]
    
    api_results = []
    for file_path, check_string, description in api_checks:
        result = check_import_in_file(file_path, check_string, description)
        api_results.append(result)
    
    print()
    
    # Summary
    print("📊 Integration Summary:")
    print("=" * 50)
    
    total_files = len(file_results)
    total_imports = len(import_results)
    total_apis = len(api_results)
    
    files_passed = sum(file_results)
    imports_passed = sum(import_results)
    apis_passed = sum(api_results)
    
    print(f"Files Created: {files_passed}/{total_files}")
    print(f"Imports Integrated: {imports_passed}/{total_imports}")
    print(f"API Endpoints: {apis_passed}/{total_apis}")
    
    total_checks = total_files + total_imports + total_apis
    total_passed = files_passed + imports_passed + apis_passed
    
    print(f"\n🎯 Overall: {total_passed}/{total_checks} checks passed")
    
    if total_passed == total_checks:
        print("🎉 All integration checks passed! AI tagging is properly integrated.")
        print("\n📋 Next steps:")
        print("   1. Restart the Python backend to load new services")
        print("   2. Test the API endpoints at http://localhost:8181/api/ai-tagging/status")
        print("   3. Try crawling new content to see AI tags in action")
    else:
        print("⚠️  Some integration checks failed. Review the output above.")
    
    return total_passed == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
