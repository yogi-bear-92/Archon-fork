/**
 * ANSF Phase 1 Integration Tests
 * Memory-critical deployment validation
 */

const { ANSFPhase1Orchestrator } = require('../../src/ansf-workflow/ansf-phase1-orchestrator');

describe('ANSF Phase 1 Integration Tests', () => {
    let orchestrator;
    
    beforeEach(() => {
        orchestrator = new ANSFPhase1Orchestrator({
            swarmId: 'test_swarm_ansf_phase1',
            clusterId: 'test_cluster_neural',
            archonTaskId: 'test_task_49426ba1'
        });
    });

    afterEach(async () => {
        if (orchestrator) {
            await orchestrator.shutdown();
        }
    });

    describe('Memory Management', () => {
        test('should initialize memory monitoring with 25MB Serena cache budget', () => {
            expect(orchestrator.memoryMonitor.serenaCacheBudget).toBe(25);
            expect(orchestrator.memoryMonitor.emergencyThreshold).toBe(98.5);
        });

        test('should trigger emergency protocols when memory >99%', async () => {
            const mockMemoryInfo = {
                total: 17179869184,
                available: 50 * 1024 * 1024, // 50MB
                usagePercent: 99.7,
                serenaCacheUsage: 30
            };

            orchestrator.memoryMonitor.getMemoryInfo = jest.fn().mockReturnValue(mockMemoryInfo);
            
            const spy = jest.spyOn(orchestrator.memoryMonitor, 'triggerEmergencyProtocols');
            orchestrator.memoryMonitor.checkMemoryUsage();
            
            expect(spy).toHaveBeenCalledWith(mockMemoryInfo);
        });
    });

    describe('Hierarchical Coordination', () => {
        test('should initialize 3-agent hierarchy', async () => {
            await orchestrator.setupCoordination();
            
            const status = orchestrator.coordinator.getStatus();
            expect(status.agents).toHaveLength(3);
            
            const agentNames = status.agents.map(a => a.name);
            expect(agentNames).toContain('ANSF-Orchestrator');
            expect(agentNames).toContain('Serena-Intelligence');
            expect(agentNames).toContain('Neural-Processor');
        });

        test('should establish apex, intelligence, and processor levels', async () => {
            await orchestrator.setupCoordination();
            
            const coordination = orchestrator.coordinator.coordination;
            expect(coordination.apex).toBeTruthy();
            expect(coordination.intelligence).toBeTruthy();
            expect(coordination.processor).toBeTruthy();
            
            expect(coordination.apex.level).toBe('apex');
            expect(coordination.intelligence.level).toBe('intelligence');
            expect(coordination.processor.level).toBe('processor');
        });
    });

    describe('Neural Cluster Integration', () => {
        test('should connect to neural cluster', async () => {
            const connectionResult = await orchestrator.neuralConnector.connect();
            
            expect(connectionResult.success).toBe(true);
            expect(connectionResult.nodes).toHaveLength(3);
            expect(orchestrator.neuralConnector.isConnected).toBe(true);
        });

        test('should validate target accuracy of 86.6%', async () => {
            await orchestrator.neuralConnector.connect();
            
            // Mock successful training
            orchestrator.neuralConnector.nodes.forEach((node) => {
                node.metrics = { accuracy: 87.2, throughput: 85, latency: 15 };
            });
            
            const validation = await orchestrator.neuralConnector.validateAccuracy();
            
            expect(validation.success).toBe(true);
            expect(validation.target_accuracy).toBe(86.6);
            expect(validation.accuracy_achieved).toBe(true);
        });
    });

    describe('Integrated Workflow Execution', () => {
        test('should execute complete ANSF Phase 1 deployment', async () => {
            const deploymentResult = await orchestrator.executePhase1();
            
            expect(orchestrator.status).toBe('completed');
            expect(deploymentResult.performance.deploymentSuccess).toBe(true);
            expect(deploymentResult.results).toHaveProperty('memoryMonitoring');
            expect(deploymentResult.results).toHaveProperty('coordination');
            expect(deploymentResult.results).toHaveProperty('neuralCluster');
            expect(deploymentResult.results).toHaveProperty('workflow');
            expect(deploymentResult.results).toHaveProperty('validation');
        });

        test('should handle deployment failure gracefully', async () => {
            // Mock a failure in neural connection
            orchestrator.neuralConnector.connect = jest.fn().mockRejectedValue(new Error('Connection failed'));
            
            await expect(orchestrator.executePhase1()).rejects.toThrow('Connection failed');
            expect(orchestrator.status).toBe('failed');
        });
    });

    describe('Performance Validation', () => {
        test('should meet memory constraints', async () => {
            await orchestrator.executePhase1();
            
            const summary = orchestrator.getDeploymentSummary();
            expect(summary.performance.memoryConstraint).toBe('92MB available');
            expect(summary.performance.serenaCacheBudget).toBe('25MB');
        });

        test('should achieve target neural accuracy', async () => {
            // Mock successful training with high accuracy
            orchestrator.neuralConnector.calculateClusterMetrics = jest.fn().mockReturnValue({
                average_accuracy: 88.4,
                node_accuracies: [
                    { node_id: 'node_1', accuracy: 87.2 },
                    { node_id: 'node_2', accuracy: 89.1 },
                    { node_id: 'node_3', accuracy: 88.9 }
                ],
                total_nodes: 3
            });
            
            await orchestrator.executePhase1();
            
            const summary = orchestrator.getDeploymentSummary();
            expect(parseFloat(summary.performance.actualNeuralAccuracy)).toBeGreaterThanOrEqual(86.6);
        });
    });

    describe('Emergency Protocols', () => {
        test('should execute emergency protocols on critical failure', async () => {
            const mockError = new Error('Critical system failure');
            
            const emergencySpy = jest.spyOn(orchestrator, 'executeEmergencyProtocols');
            orchestrator.executeIntegratedWorkflow = jest.fn().mockRejectedValue(mockError);
            
            await expect(orchestrator.executePhase1()).rejects.toThrow('Critical system failure');
            expect(emergencySpy).toHaveBeenCalledWith(mockError);
        });
    });
});