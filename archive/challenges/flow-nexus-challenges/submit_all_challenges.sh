#!/bin/bash

# Flow Nexus Challenge Submission Script
# This script attempts to submit all completed challenges

echo "🚀 Starting Flow Nexus Challenge Submission Process..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check authentication status
echo "🔐 Checking authentication status..."
npx flow-nexus challenge status

# Define challenge mappings (directory name -> solution file)
declare -A challenges=(
    ["neural-trading-bot-challenge"]="solution.py"
    ["agent-spawning-master"]="solution.js"
    ["flow-nexus-trading-workflow"]="solution.js"
    ["neural-trading-trials"]="solution.js"
    ["neural-mesh-coordinator"]="solution.js"
    ["lightning-deploy-master"]="solution.js"
    ["ruv-economy-dominator"]="solution.js"
    ["bug-hunters-gauntlet"]="solution.js"
    ["algorithm-duel-arena"]="solution.js"
    ["neural-conductor"]="solution.js"
    ["swarm-warfare-commander"]="solution.js"
    ["phantom-constructor"]="solution.js"
    ["system-sage-trials"]="solution.js"
    ["labyrinth-architect"]="solution.js"
)

# Known UUIDs (if any)
declare -A challenge_uuids=(
    ["neural-trading-bot-challenge"]="c94777b9-6af5-4b15-8411-8391aa640864"
)

echo "📊 Found ${#challenges[@]} challenges ready for submission"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Attempt to submit each challenge
success_count=0
failed_count=0

for challenge_dir in "${!challenges[@]}"; do
    solution_file="${challenges[$challenge_dir]}"
    solution_path="./challenges/$challenge_dir/$solution_file"
    
    echo "🎯 Attempting to submit: $challenge_dir"
    echo "   Solution file: $solution_path"
    
    # Check if solution file exists
    if [[ ! -f "$solution_path" ]]; then
        echo "   ❌ Solution file not found: $solution_path"
        ((failed_count++))
        continue
    fi
    
    # Try with known UUID if available
    if [[ -n "${challenge_uuids[$challenge_dir]}" ]]; then
        uuid="${challenge_uuids[$challenge_dir]}"
        echo "   🔑 Using known UUID: $uuid"
        
        if npx flow-nexus challenge submit -i "$uuid" --solution "$solution_path"; then
            echo "   ✅ Successfully submitted: $challenge_dir"
            ((success_count++))
        else
            echo "   ❌ Failed to submit: $challenge_dir"
            ((failed_count++))
        fi
    else
        # Try with challenge directory name as fallback
        echo "   🔍 Attempting with directory name: $challenge_dir"
        
        if npx flow-nexus challenge submit -i "$challenge_dir" --solution "$solution_path"; then
            echo "   ✅ Successfully submitted: $challenge_dir"
            ((success_count++))
        else
            echo "   ❌ Failed to submit: $challenge_dir (UUID required)"
            ((failed_count++))
        fi
    fi
    
    echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Small delay between submissions
    sleep 1
done

echo ""
echo "🏆 SUBMISSION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Successful submissions: $success_count"
echo "❌ Failed submissions: $failed_count"
echo "📊 Total challenges: ${#challenges[@]}"

# Show final status
echo ""
echo "🔍 Final challenge status:"
npx flow-nexus challenge status

echo ""
echo "💎 Current rUv balance:"
npx flow-nexus credits balance

echo ""
echo "🎯 Submission process complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
