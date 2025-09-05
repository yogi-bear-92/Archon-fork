/**
 * Agent Spawning Master Challenge Solution
 * Flow Nexus Challenge ID: 71fb989e-43d8-40b5-9c67-85815081d974
 * 
 * This solution demonstrates proper swarm initialization and agent deployment
 * using Claude-Flow MCP tools with mesh topology.
 */

// Solution implementation for Agent Spawning Master Challenge
async function spawnAgentSwarm() {
  try {
    console.log("🚀 Initializing Agent Spawning Master Challenge...");
    
    // 1. Initialize swarm with mesh topology
    console.log("📡 Step 1: Initializing swarm with mesh topology...");
    const swarmResult = await mcp_flow_nexus_swarm_init({
      topology: "mesh",
      maxAgents: 5,
      strategy: "balanced"
    });
    
    if (!swarmResult.success) {
      throw new Error(`Swarm initialization failed: ${swarmResult.error}`);
    }
    
    console.log(`✅ Swarm initialized successfully!`);
    console.log(`   Swarm ID: ${swarmResult.swarm_id}`);
    console.log(`   Topology: ${swarmResult.topology}`);
    console.log(`   Max Agents: ${swarmResult.max_agents}`);
    console.log(`   Agents Deployed: ${swarmResult.agents_deployed}`);
    
    // 2. Spawn a coordinator agent
    console.log("🤖 Step 2: Spawning coordinator agent...");
    const agentResult = await mcp_flow_nexus_agent_spawn({
      type: "coordinator",
      name: "primary-coordinator",
      capabilities: [
        "task-coordination",
        "swarm-management", 
        "performance-monitoring"
      ]
    });
    
    if (!agentResult.success) {
      throw new Error(`Agent spawning failed: ${agentResult.error}`);
    }
    
    console.log(`✅ Coordinator agent spawned successfully!`);
    console.log(`   Agent ID: ${agentResult.agent_id}`);
    console.log(`   Agent Name: ${agentResult.name}`);
    console.log(`   Agent Type: ${agentResult.type}`);
    console.log(`   Capabilities: ${agentResult.capabilities.join(", ")}`);
    console.log(`   Sandbox ID: ${agentResult.sandbox_id}`);
    
    // 3. Return swarm status
    console.log("📊 Step 3: Retrieving swarm status...");
    const statusResult = await mcp_flow_nexus_swarm_status();
    
    if (!statusResult.success) {
      throw new Error(`Status retrieval failed: ${statusResult.error}`);
    }
    
    console.log(`✅ Swarm status retrieved successfully!`);
    console.log(`   Active Agents: ${statusResult.swarm.agents.length}`);
    console.log(`   Swarm Status: ${statusResult.swarm.status}`);
    console.log(`   Runtime: ${statusResult.swarm.runtime_minutes} minutes`);
    
    // Prepare result object
    const result = {
      success: true,
      swarm: {
        id: swarmResult.swarm_id,
        topology: swarmResult.topology,
        strategy: swarmResult.strategy,
        status: statusResult.swarm.status,
        maxAgents: swarmResult.max_agents,
        activeAgents: statusResult.swarm.agents.length
      },
      coordinator: {
        id: agentResult.agent_id,
        name: agentResult.name,
        type: agentResult.type,
        capabilities: agentResult.capabilities,
        status: agentResult.status,
        sandboxId: agentResult.sandbox_id
      },
      message: "Swarm initialized with mesh topology and coordinator agent spawned successfully",
      challengeCompleted: true,
      timestamp: new Date().toISOString()
    };
    
    console.log("🎉 Challenge completed successfully!");
    console.log("📈 Expected rUv reward: 150 (base) + participation bonus");
    
    return result;
    
  } catch (error) {
    console.error("❌ Challenge execution failed:", error.message);
    return {
      success: false,
      error: error.message,
      challengeCompleted: false,
      timestamp: new Date().toISOString()
    };
  }
}

// Execute the challenge solution
spawnAgentSwarm().then(result => {
  console.log("🔥 Final Result:", JSON.stringify(result, null, 2));
}).catch(error => {
  console.error("💥 Execution Error:", error);
});

// Export for testing and validation
module.exports = { spawnAgentSwarm };