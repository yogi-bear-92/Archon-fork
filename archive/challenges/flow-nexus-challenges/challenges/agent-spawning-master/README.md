# Agent Spawning Master Challenge

**Difficulty:** Beginner  
**Reward:** 150 rUv + 10 rUv participation  
**Challenge ID:** `71fb989e-43d8-40b5-9c67-85815081d974`  
**Archon Project ID:** `ea8c3fe2-f7e4-4f41-825a-58279ec6f80d`

## 🎯 Challenge Description

Initialize a swarm and spawn your first autonomous agent using Claude-Flow MCP tools. Learn swarm coordination basics and agent deployment fundamentals.

## 📋 Requirements

Use the claude-flow MCP tools to:
1. Initialize swarm with mesh topology
2. Spawn a coordinator agent
3. Return swarm status

## ✅ Test Cases

- Initialize mesh swarm with coordinator
- Expected: "Swarm initialized, agent spawned successfully"
- Should create swarm and spawn coordinator agent

## 🛠️ Tools Required

- `mcp__claude-flow__swarm_init`
- `mcp__claude-flow__agent_spawn`
- `mcp__claude-flow__swarm_status`

## 📁 Project Structure

```
agent-spawning-master/
├── README.md              # This file
├── challenge-info.json    # Challenge metadata
├── starter-code.js        # Original starter template
├── solution.js           # Your solution
└── test.js              # Test runner
```