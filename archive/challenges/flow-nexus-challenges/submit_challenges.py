#!/usr/bin/env python3
"""
Flow Nexus Challenge Submission Script
Attempts to submit all completed challenges
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🚀 Starting Flow Nexus Challenge Submission Process...")
    print("━" * 70)
    
    # Check authentication status
    print("🔐 Checking authentication status...")
    success, stdout, stderr = run_command("npx flow-nexus challenge status", capture_output=False)
    print()
    
    # Define challenges with their solution files
    challenges = {
        "neural-trading-bot-challenge": "solution.py",
        "agent-spawning-master": "solution.js",
        "flow-nexus-trading-workflow": "solution.js", 
        "neural-trading-trials": "solution.js",
        "neural-mesh-coordinator": "solution.js",
        "lightning-deploy-master": "solution.js",
        "ruv-economy-dominator": "solution.js",
        "bug-hunters-gauntlet": "solution.js",
        "algorithm-duel-arena": "solution.js",
        "neural-conductor": "solution.js",
        "swarm-warfare-commander": "solution.js",
        "phantom-constructor": "solution.js",
        "system-sage-trials": "solution.js",
        "labyrinth-architect": "solution.js"
    }
    
    # Known UUIDs (if any)
    known_uuids = {
        "neural-trading-bot-challenge": "c94777b9-6af5-4b15-8411-8391aa640864"
    }
    
    print(f"📊 Found {len(challenges)} challenges ready for submission")
    print("━" * 70)
    
    success_count = 0
    failed_count = 0
    
    for challenge_dir, solution_file in challenges.items():
        solution_path = f"./challenges/{challenge_dir}/{solution_file}"
        
        print(f"🎯 Attempting to submit: {challenge_dir}")
        print(f"   Solution file: {solution_path}")
        
        # Check if solution file exists
        if not os.path.exists(solution_path):
            print(f"   ❌ Solution file not found: {solution_path}")
            failed_count += 1
            continue
        
        # Try with known UUID if available
        if challenge_dir in known_uuids:
            uuid = known_uuids[challenge_dir]
            print(f"   🔑 Using known UUID: {uuid}")
            cmd = f'npx flow-nexus challenge submit -i "{uuid}" --solution "{solution_path}"'
        else:
            # Try with challenge directory name as fallback
            print(f"   🔍 Attempting with directory name: {challenge_dir}")
            cmd = f'npx flow-nexus challenge submit -i "{challenge_dir}" --solution "{solution_path}"'
        
        success, stdout, stderr = run_command(cmd, capture_output=False)
        
        if success:
            print(f"   ✅ Successfully submitted: {challenge_dir}")
            success_count += 1
        else:
            print(f"   ❌ Failed to submit: {challenge_dir}")
            failed_count += 1
        
        print("   " + "━" * 60)
        print()
    
    print("🏆 SUBMISSION SUMMARY")
    print("━" * 70)
    print(f"✅ Successful submissions: {success_count}")
    print(f"❌ Failed submissions: {failed_count}")
    print(f"📊 Total challenges: {len(challenges)}")
    
    # Show final status
    print("\n🔍 Final challenge status:")
    run_command("npx flow-nexus challenge status", capture_output=False)
    
    print("\n💎 Current rUv balance:")
    run_command("npx flow-nexus credits balance", capture_output=False)
    
    print("\n🎯 Submission process complete!")
    print("━" * 70)

if __name__ == "__main__":
    main()
