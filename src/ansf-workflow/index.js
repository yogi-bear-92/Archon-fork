/**
 * ANSF Phase 1 Main Entry Point
 * Execute memory-critical deployment
 */

import { ANSFPhase1Orchestrator } from './ansf-phase1-orchestrator.js';

async function executeANSFPhase1() {
    console.log('🎯 ANSF Phase 1 Deployment Starting...');
    console.log('📊 Constraints: 92MB memory, 25MB Serena cache, 86.6% neural accuracy target');
    
    const orchestrator = new ANSFPhase1Orchestrator({
        swarmId: 'swarm_1757099879115_tw86n5ast',
        clusterId: 'dnc_66761a355235', 
        archonTaskId: '49426ba1-2d54-4d67-bf9b-e1bc00a2cde4'
    });

    try {
        const deploymentSummary = await orchestrator.executePhase1();
        
        console.log('\n🎉 ANSF Phase 1 Deployment Summary:');
        console.log('===============================================');
        console.log(`Status: ${deploymentSummary.status}`);
        console.log(`Swarm ID: ${deploymentSummary.swarmId}`);
        console.log(`Neural Cluster: ${deploymentSummary.clusterId}`);
        console.log(`Archon Task: ${deploymentSummary.archonTaskId}`);
        console.log('\nPerformance Metrics:');
        console.log(`  Memory Constraint: ${deploymentSummary.performance.memoryConstraint}`);
        console.log(`  Serena Cache Budget: ${deploymentSummary.performance.serenaCacheBudget}`);
        console.log(`  Target Neural Accuracy: ${deploymentSummary.performance.targetNeuralAccuracy}`);
        console.log(`  Actual Neural Accuracy: ${deploymentSummary.performance.actualNeuralAccuracy}`);
        console.log(`  Deployment Success: ${deploymentSummary.performance.deploymentSuccess}`);
        
        console.log('\nDeployment Components:');
        Object.keys(deploymentSummary.results).forEach(component => {
            console.log(`  ✅ ${component}: deployed`);
        });
        
        return deploymentSummary;
        
    } catch (error) {
        console.error('\n❌ ANSF Phase 1 Deployment Failed:', error.message);
        
        // Attempt graceful shutdown
        try {
            await orchestrator.shutdown();
        } catch (shutdownError) {
            console.error('⚠️  Shutdown error:', shutdownError.message);
        }
        
        throw error;
    }
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
    executeANSFPhase1()
        .then(summary => {
            console.log('\n🚀 ANSF Phase 1 completed successfully!');
            process.exit(0);
        })
        .catch(error => {
            console.error('\n💥 ANSF Phase 1 failed:', error);
            process.exit(1);
        });
}

export { executeANSFPhase1, ANSFPhase1Orchestrator };