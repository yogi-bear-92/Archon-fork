/**
 * ANSF Phase 1 Main Orchestrator
 * Integrates Memory Monitor, Hierarchical Coordinator, and Neural Connector
 */

const { ANSFMemoryMonitor } = require('./memory/memory-monitor');
const { HierarchicalCoordinator } = require('./coordination/hierarchical-coordinator');
const { DistributedNeuralConnector } = require('./neural/distributed-connector');

class ANSFPhase1Orchestrator {
    constructor(options = {}) {
        this.phase = 1;
        this.swarmId = options.swarmId || 'swarm_1757099879115_tw86n5ast';
        this.clusterId = options.clusterId || 'dnc_66761a355235';
        this.archonTaskId = options.archonTaskId || '49426ba1-2d54-4d67-bf9b-e1bc00a2cde4';
        
        // Initialize core components
        this.memoryMonitor = new ANSFMemoryMonitor({
            maxMemoryPercent: 99,
            emergencyThreshold: 98.5,
            serenaCacheBudget: 25
        });
        
        this.coordinator = new HierarchicalCoordinator({
            swarmId: this.swarmId,
            maxAgents: 3
        });
        
        this.neuralConnector = new DistributedNeuralConnector({
            clusterId: this.clusterId,
            targetAccuracy: 86.6
        });
        
        this.status = 'initialized';
        this.deploymentResults = {};
    }

    /**
     * Execute complete ANSF Phase 1 deployment
     */
    async executePhase1() {
        console.log('🚀 ANSF Phase 1 Orchestrator: Beginning deployment');
        console.log(`📊 Target: 92MB memory constraint, 25MB Serena cache, 86.6% neural accuracy`);
        
        this.status = 'executing';
        const executionStart = Date.now();
        
        try {
            // Step 1: Initialize memory monitoring
            console.log('\n🧠 Step 1: Initialize Memory Monitoring');
            await this.initializeMemoryMonitoring();
            
            // Step 2: Setup hierarchical coordination
            console.log('\n🏗️  Step 2: Setup Hierarchical Coordination');
            await this.setupCoordination();
            
            // Step 3: Connect to neural cluster
            console.log('\n🔗 Step 3: Connect to Neural Cluster');
            await this.connectNeuralCluster();
            
            // Step 4: Execute integrated ANSF workflow
            console.log('\n⚡ Step 4: Execute Integrated ANSF Workflow');
            await this.executeIntegratedWorkflow();
            
            // Step 5: Validate and complete
            console.log('\n✅ Step 5: Validate and Complete Phase 1');
            await this.validateAndComplete();
            
            this.status = 'completed';
            const executionTime = Date.now() - executionStart;
            
            console.log(`\n🎉 ANSF Phase 1 deployment completed in ${executionTime}ms`);
            
            return this.getDeploymentSummary();
            
        } catch (error) {
            this.status = 'failed';
            console.error('\n❌ ANSF Phase 1 deployment failed:', error);
            
            // Execute emergency protocols
            await this.executeEmergencyProtocols(error);
            
            throw error;
        }
    }

    /**
     * Step 1: Initialize memory monitoring
     */
    async initializeMemoryMonitoring() {
        this.memoryMonitor.startMonitoring();
        
        const memoryStatus = this.memoryMonitor.checkMemoryUsage();
        console.log(`💾 Memory status: ${memoryStatus.usagePercent}% used, ${memoryStatus.available / (1024*1024)}MB available`);
        
        this.deploymentResults.memoryMonitoring = {
            status: 'active',
            initialMemoryState: memoryStatus,
            serenaCacheBudget: '25MB'
        };
        
        if (memoryStatus.usagePercent > 99) {
            console.log('⚠️  High memory usage detected - enabling conservative mode');
        }
    }

    /**
     * Step 2: Setup hierarchical coordination
     */
    async setupCoordination() {
        const coordinationStatus = await this.coordinator.initialize();
        
        console.log(`👥 Hierarchical coordination active with ${coordinationStatus.agents.length} agents`);
        
        // Register agents with memory monitor
        coordinationStatus.agents.forEach(agent => {
            this.memoryMonitor.registerAgent(agent);
        });
        
        this.deploymentResults.coordination = {
            status: 'active',
            swarmId: coordinationStatus.swarmId,
            agents: coordinationStatus.agents.map(a => ({
                name: a.name,
                type: a.type,
                level: a.level
            })),
            hierarchy: coordinationStatus.coordination
        };
    }

    /**
     * Step 3: Connect to neural cluster
     */
    async connectNeuralCluster() {
        const connectionResult = await this.neuralConnector.connect();
        
        if (connectionResult.success !== false) {
            console.log(`🤖 Neural cluster connected: ${connectionResult.nodes.length} nodes active`);
            
            this.deploymentResults.neuralCluster = {
                status: 'connected',
                clusterId: this.clusterId,
                nodes: connectionResult.nodes.length,
                topology: connectionResult.topology,
                architecture: connectionResult.architecture
            };
        } else {
            throw new Error(`Neural cluster connection failed: ${connectionResult.error}`);
        }
    }

    /**
     * Step 4: Execute integrated ANSF workflow
     */
    async executeIntegratedWorkflow() {
        const taskDescription = `ANSF Phase 1 Integration: Deploy memory-critical 3-agent hierarchy with Serena semantic analysis (25MB cache) and neural cluster training (target: 86.6% accuracy). Implement emergency protocols for 92MB memory constraint.`;
        
        // Execute through hierarchical coordinator
        const orchestrationResult = await this.coordinator.orchestratePhase1(taskDescription);
        
        // Start neural training
        const trainingResult = await this.neuralConnector.startANSFTraining({
            epochs: 3,
            batch_size: 12, // Reduced for memory constraints
            learning_rate: 0.0005
        });
        
        this.deploymentResults.workflow = {
            orchestration: orchestrationResult,
            neuralTraining: trainingResult
        };
        
        console.log(`🔄 Integrated workflow executed - Training accuracy: ${trainingResult.current_accuracy}%`);
    }

    /**
     * Step 5: Validate and complete
     */
    async validateAndComplete() {
        // Validate neural accuracy
        const accuracyValidation = await this.neuralConnector.validateAccuracy();
        
        // Check memory status
        const finalMemoryStatus = this.memoryMonitor.checkMemoryUsage();
        
        // Get coordination status
        const coordinationStatus = this.coordinator.getStatus();
        
        this.deploymentResults.validation = {
            neuralAccuracy: accuracyValidation,
            memoryStatus: finalMemoryStatus,
            coordinationHealth: {
                activeAgents: coordinationStatus.agents.filter(a => a.status === 'active').length,
                totalTasks: coordinationStatus.tasks.length
            }
        };
        
        console.log(`📈 Validation complete:`);
        console.log(`   - Neural accuracy: ${accuracyValidation.current_accuracy}% (target: ${accuracyValidation.target_accuracy}%)`);
        console.log(`   - Memory usage: ${finalMemoryStatus.usagePercent}%`);
        console.log(`   - Active agents: ${this.deploymentResults.validation.coordinationHealth.activeAgents}`);
        
        return this.deploymentResults.validation;
    }

    /**
     * Execute emergency protocols
     */
    async executeEmergencyProtocols(error) {
        console.log('🚨 Executing emergency protocols due to failure');
        
        try {
            // Scale to emergency mode
            if (this.memoryMonitor.isMonitoring) {
                const memoryInfo = this.memoryMonitor.checkMemoryUsage();
                if (memoryInfo.usagePercent > 99) {
                    this.memoryMonitor.triggerEmergencyProtocols(memoryInfo);
                }
            }
            
            // Attempt graceful shutdown of components
            if (this.coordinator.isActive) {
                await this.coordinator.shutdown();
            }
            
            if (this.neuralConnector.isConnected) {
                await this.neuralConnector.disconnect();
            }
            
            console.log('🛡️  Emergency protocols executed successfully');
            
        } catch (emergencyError) {
            console.error('❌ Emergency protocols failed:', emergencyError);
        }
    }

    /**
     * Get deployment summary
     */
    getDeploymentSummary() {
        const summary = {
            phase: this.phase,
            status: this.status,
            swarmId: this.swarmId,
            clusterId: this.clusterId,
            archonTaskId: this.archonTaskId,
            timestamp: Date.now(),
            results: this.deploymentResults,
            performance: {
                memoryConstraint: '92MB available',
                serenaCacheBudget: '25MB',
                targetNeuralAccuracy: '86.6%',
                actualNeuralAccuracy: this.deploymentResults.validation?.neuralAccuracy?.current_accuracy || 'pending',
                deploymentSuccess: this.status === 'completed'
            }
        };
        
        return summary;
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            phase: this.phase,
            status: this.status,
            components: {
                memoryMonitor: this.memoryMonitor.isMonitoring,
                coordinator: this.coordinator.isActive,
                neuralConnector: this.neuralConnector.isConnected
            },
            deploymentProgress: Object.keys(this.deploymentResults),
            timestamp: Date.now()
        };
    }

    /**
     * Shutdown all components
     */
    async shutdown() {
        console.log('🛑 Shutting down ANSF Phase 1 Orchestrator');
        
        if (this.memoryMonitor.isMonitoring) {
            this.memoryMonitor.stopMonitoring();
        }
        
        if (this.coordinator.isActive) {
            await this.coordinator.shutdown();
        }
        
        if (this.neuralConnector.isConnected) {
            await this.neuralConnector.disconnect();
        }
        
        this.status = 'shutdown';
        console.log('✅ ANSF Phase 1 Orchestrator shutdown complete');
    }
}

module.exports = { ANSFPhase1Orchestrator };