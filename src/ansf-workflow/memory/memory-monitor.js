/**
 * ANSF Phase 1 Memory Monitor
 * Critical memory management for 92MB available constraint
 */

class ANSFMemoryMonitor {
    constructor(options = {}) {
        this.maxMemoryPercent = options.maxMemoryPercent || 99;
        this.emergencyThreshold = options.emergencyThreshold || 98;
        this.alertInterval = options.alertInterval || 5000; // 5 seconds
        this.isMonitoring = false;
        this.agents = new Map();
        this.serenaCacheBudget = options.serenaCacheBudget || 25; // 25MB
    }

    /**
     * Start memory monitoring with emergency protocols
     */
    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        console.log(`🧠 ANSF Memory Monitor: Starting with ${this.serenaCacheBudget}MB Serena cache budget`);
        
        this.monitorInterval = setInterval(() => {
            this.checkMemoryUsage();
        }, this.alertInterval);
    }

    /**
     * Check system memory and trigger protocols
     */
    checkMemoryUsage() {
        const memInfo = this.getMemoryInfo();
        
        if (memInfo.usagePercent >= this.emergencyThreshold) {
            this.triggerEmergencyProtocols(memInfo);
        } else if (memInfo.usagePercent >= this.maxMemoryPercent - 5) {
            this.triggerLimitedMode(memInfo);
        }
        
        return memInfo;
    }

    /**
     * Get memory information (macOS compatible)
     */
    getMemoryInfo() {
        // Simulated memory info for macOS environment
        return {
            total: 17179869184, // 16GB
            available: 92 * 1024 * 1024, // 92MB available
            usagePercent: 99.5,
            serenaCacheUsage: this.getCurrentSerenaCacheUsage()
        };
    }

    /**
     * Emergency protocols - single agent mode
     */
    triggerEmergencyProtocols(memInfo) {
        console.log('🚨 EMERGENCY: Memory critical - activating emergency protocols');
        
        // Scale down to single agent
        this.scaleToEmergencyMode();
        
        // Clear non-essential caches
        this.clearNonEssentialCaches();
        
        // Notify coordination layer
        this.notifyCoordination('EMERGENCY', memInfo);
    }

    /**
     * Limited mode - 2-3 agents maximum
     */
    triggerLimitedMode(memInfo) {
        console.log('⚠️  LIMITED: Memory high - scaling to limited mode');
        
        this.scaleToLimitedMode();
        this.optimizeSerenaCacheUsage();
        this.notifyCoordination('LIMITED', memInfo);
    }

    /**
     * Scale to emergency mode (1 agent)
     */
    scaleToEmergencyMode() {
        const activeAgents = Array.from(this.agents.values()).filter(a => a.status === 'active');
        
        if (activeAgents.length > 1) {
            // Keep only the orchestrator agent
            activeAgents.slice(1).forEach(agent => {
                this.pauseAgent(agent.id, 'EMERGENCY_SCALE_DOWN');
            });
        }
    }

    /**
     * Scale to limited mode (2-3 agents)
     */
    scaleToLimitedMode() {
        const activeAgents = Array.from(this.agents.values()).filter(a => a.status === 'active');
        
        if (activeAgents.length > 3) {
            // Keep orchestrator + 2 essential agents
            activeAgents.slice(3).forEach(agent => {
                this.pauseAgent(agent.id, 'LIMITED_SCALE_DOWN');
            });
        }
    }

    /**
     * Optimize Serena cache usage
     */
    optimizeSerenaCacheUsage() {
        const currentUsage = this.getCurrentSerenaCacheUsage();
        
        if (currentUsage > this.serenaCacheBudget) {
            console.log(`📦 Serena Cache: Optimizing from ${currentUsage}MB to ${this.serenaCacheBudget}MB`);
            // Implement cache cleanup logic
            this.clearSerenaCacheToTarget(this.serenaCacheBudget);
        }
    }

    /**
     * Get current Serena cache usage
     */
    getCurrentSerenaCacheUsage() {
        // Simulated cache usage
        return Math.random() * 30 + 10; // 10-40MB range
    }

    /**
     * Clear Serena cache to target size
     */
    clearSerenaCacheToTarget(targetSizeMB) {
        console.log(`🧹 Clearing Serena cache to ${targetSizeMB}MB target`);
        // Implementation would interface with Serena MCP server
        return targetSizeMB;
    }

    /**
     * Pause agent with reason
     */
    pauseAgent(agentId, reason) {
        const agent = this.agents.get(agentId);
        if (agent) {
            agent.status = 'paused';
            agent.pauseReason = reason;
            console.log(`⏸️  Agent ${agentId} paused: ${reason}`);
        }
    }

    /**
     * Register agent for monitoring
     */
    registerAgent(agent) {
        this.agents.set(agent.id, {
            ...agent,
            status: 'active',
            registeredAt: Date.now()
        });
        
        console.log(`📋 Registered agent: ${agent.name} (${agent.type})`);
    }

    /**
     * Notify coordination layer
     */
    notifyCoordination(level, memInfo) {
        const notification = {
            level,
            timestamp: Date.now(),
            memoryInfo: memInfo,
            activeAgents: Array.from(this.agents.values()).filter(a => a.status === 'active').length,
            serenaCacheUsage: memInfo.serenaCacheUsage
        };
        
        // Would send to Claude Flow coordination layer
        console.log(`📡 Coordination notification:`, notification);
        
        return notification;
    }

    /**
     * Stop monitoring
     */
    stopMonitoring() {
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
            this.isMonitoring = false;
            console.log('🛑 ANSF Memory Monitor stopped');
        }
    }

    /**
     * Get monitoring status
     */
    getStatus() {
        return {
            isMonitoring: this.isMonitoring,
            agents: Array.from(this.agents.values()),
            memoryInfo: this.checkMemoryUsage(),
            serenaCacheBudget: this.serenaCacheBudget
        };
    }
}

module.exports = { ANSFMemoryMonitor };