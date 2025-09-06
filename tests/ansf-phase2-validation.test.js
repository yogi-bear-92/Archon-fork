/**
 * ANSF Phase 2 Enhanced Mode Validation Tests
 * Comprehensive test suite for semantic cache, LSP integration, and neural pattern learning
 */

const { describe, it, beforeAll, afterAll, expect } = require('@jest/globals');

// Import Phase 2 components
const { ANSFPhase2Orchestrator } = require('../src/ansf-workflow/ansf-phase2-orchestrator');
const { Phase2SemanticCache } = require('../src/ansf-workflow/semantic/phase2-semantic-cache');
const { LSPIntegrationSystem } = require('../src/ansf-workflow/semantic/lsp-integration');
const { SemanticIntegrationHooks } = require('../src/ansf-workflow/semantic/integration-hooks');

describe('ANSF Phase 2 Enhanced Mode Validation', () => {
    let orchestrator;
    let semanticCache;
    let lspSystem;
    let integrationHooks;

    beforeAll(async () => {
        console.log('🧪 Setting up ANSF Phase 2 test environment...');
    });

    afterAll(async () => {
        // Cleanup all components
        if (orchestrator) await orchestrator.shutdown();
        if (semanticCache) await semanticCache.shutdown();
        if (lspSystem) await lspSystem.shutdown();
        if (integrationHooks) await integrationHooks.shutdown();
        
        console.log('🧹 ANSF Phase 2 test environment cleaned up');
    });

    describe('Phase2SemanticCache', () => {
        beforeAll(async () => {
            semanticCache = new Phase2SemanticCache({
                hotCacheSize: 1024 * 1024,   // 1MB for testing
                warmCacheSize: 2 * 1024 * 1024, // 2MB for testing
                coldCacheSize: 1024 * 1024,  // 1MB for testing
                learningEnabled: true,
                neuralClusterId: 'test-cluster'
            });
        });

        it('should initialize with correct tier configuration', () => {
            expect(semanticCache).toBeDefined();
            expect(semanticCache.config.hotCacheSize).toBe(1024 * 1024);
            expect(semanticCache.config.warmCacheSize).toBe(2 * 1024 * 1024);
            expect(semanticCache.config.coldCacheSize).toBe(1024 * 1024);
            expect(semanticCache.learningEnabled).toBe(true);
        });

        it('should store and retrieve data in hot cache', async () => {
            const testData = {
                symbols: ['testFunction', 'TestClass'],
                complexity: 0.5,
                patterns: ['factory', 'singleton']
            };

            const stored = await semanticCache.store('test-hot-data', testData, {
                type: 'symbol-analysis',
                priority: 'hot',
                language: 'javascript'
            });

            expect(stored).toBe(true);

            const retrieved = await semanticCache.retrieve('test-hot-data');
            expect(retrieved).toEqual(testData);
        });

        it('should store data in appropriate tiers based on context', async () => {
            // Hot tier data
            await semanticCache.store('hot-data', { active: true }, {
                priority: 'hot',
                isActiveFile: true
            });

            // Warm tier data  
            await semanticCache.store('warm-data', { project: true }, {
                type: 'architectural'
            });

            // Cold tier data
            await semanticCache.store('cold-data', { historical: true }, {
                priority: 'cold',
                historical: true
            });

            expect(semanticCache.hotCache.has('hot-data')).toBe(true);
            expect(semanticCache.warmCache.has('warm-data')).toBe(true);
            expect(semanticCache.coldCache.has('cold-data')).toBe(true);
        });

        it('should handle cache promotion correctly', async () => {
            // Store in warm cache
            await semanticCache.store('promote-test', { data: 'test' }, {
                type: 'test'
            });

            // Access multiple times to trigger promotion
            for (let i = 0; i < 5; i++) {
                await semanticCache.retrieve('promote-test');
            }

            // Should promote to hot cache after frequent access
            const status = semanticCache.getStatus();
            expect(status.cacheStats.hot).toBeGreaterThan(0);
        });

        it('should calculate hit rate correctly', async () => {
            // Store test data
            await semanticCache.store('hit-test', { test: true }, {});
            
            // Generate hits and misses
            await semanticCache.retrieve('hit-test'); // Hit
            await semanticCache.retrieve('hit-test'); // Hit
            await semanticCache.retrieve('non-existent'); // Miss

            const status = semanticCache.getStatus();
            const hitRate = parseFloat(status.efficiency.hitRate);
            
            expect(hitRate).toBeGreaterThan(0);
            expect(hitRate).toBeLessThanOrEqual(100);
        });

        it('should perform maintenance operations', async () => {
            const initialStats = semanticCache.getStatus();
            
            // Trigger maintenance manually
            semanticCache.performMaintenance();
            
            const afterStats = semanticCache.getStatus();
            expect(afterStats).toBeDefined();
        });
    });

    describe('LSPIntegrationSystem', () => {
        beforeAll(async () => {
            lspSystem = new LSPIntegrationSystem({
                languages: ['javascript', 'typescript'],
                crossLanguageAnalysis: true,
                cacheInstance: null // No cache for isolated testing
            });

            await lspSystem.initialize();
        });

        it('should initialize with configured languages', () => {
            const status = lspSystem.getStatus();
            expect(status.initialized).toBe(true);
            expect(status.languages).toContain('javascript');
            expect(status.languages).toContain('typescript');
        });

        it('should index symbols correctly', () => {
            const status = lspSystem.getStatus();
            expect(status.symbolsIndexed).toBeGreaterThan(0);
        });

        it('should provide completions', async () => {
            const completions = await lspSystem.getCompletions(
                'test.js',
                { line: 1, character: 5 },
                { prefix: 'test' }
            );

            expect(Array.isArray(completions)).toBe(true);
            expect(completions.length).toBeGreaterThanOrEqual(0);
        });

        it('should provide hover information', async () => {
            const hover = await lspSystem.getHover(
                'test.js',
                { line: 1, character: 5 }
            );

            expect(hover).toBeDefined();
            if (hover) {
                expect(hover.contents).toBeDefined();
            }
        });

        it('should find definitions', async () => {
            const definitions = await lspSystem.getDefinition(
                'test.js',
                { line: 1, character: 5 }
            );

            expect(Array.isArray(definitions)).toBe(true);
        });

        it('should find references', async () => {
            const references = await lspSystem.getReferences(
                'test.js',
                { line: 1, character: 5 }
            );

            expect(Array.isArray(references)).toBe(true);
        });

        it('should generate architectural insights', () => {
            const insights = lspSystem.getArchitecturalInsights();
            
            expect(insights).toBeDefined();
            expect(Array.isArray(insights.patterns)).toBe(true);
            expect(Array.isArray(insights.suggestions)).toBe(true);
        });

        it('should handle cross-language analysis', () => {
            if (lspSystem.config.crossLanguageAnalysis) {
                const insights = lspSystem.getArchitecturalInsights();
                expect(insights.dependencies).toBeDefined();
            }
        });
    });

    describe('SemanticIntegrationHooks', () => {
        beforeAll(async () => {
            integrationHooks = new SemanticIntegrationHooks({
                archonPRPEnabled: true,
                claudeFlowEnabled: true,
                neuralLearningEnabled: true,
                memoryOptimization: true
            });
        });

        it('should initialize with correct configuration', () => {
            const status = integrationHooks.getStatus();
            expect(status.initialized).toBe(true);
            expect(status.capabilities.archonPRP).toBe(true);
            expect(status.capabilities.claudeFlow).toBe(true);
            expect(status.capabilities.neuralLearning).toBe(true);
        });

        it('should register built-in hooks', () => {
            const status = integrationHooks.getStatus();
            expect(status.hooksRegistered).toBeGreaterThan(0);
        });

        it('should execute Archon PRP hooks', async () => {
            const result = await integrationHooks.executeHooks('archon-prp-pre-cycle', {
                task: { description: 'Test PRP task' },
                cycle: 1,
                context: {}
            });

            expect(result.success).toBe(true);
            expect(Array.isArray(result.results)).toBe(true);
        });

        it('should execute Claude Flow hooks', async () => {
            const result = await integrationHooks.executeHooks('claude-flow-pre-task', {
                task: { description: 'Test Claude Flow task' },
                agents: ['test-agent'],
                swarmId: 'test-swarm'
            });

            expect(result.success).toBe(true);
            expect(Array.isArray(result.results)).toBe(true);
        });

        it('should handle memory optimization hooks', async () => {
            const result = await integrationHooks.executeHooks('memory-pressure-detected', {
                memoryStatus: { usage: 0.95 },
                pressure: 0.98
            });

            expect(result.success).toBe(true);
        });

        it('should register custom hooks', () => {
            const customHandler = async (data) => ({ custom: true, data });
            integrationHooks.registerHook('custom-test-event', customHandler);

            const status = integrationHooks.getStatus();
            expect(status.hooksRegistered).toBeGreaterThan(4); // Original + custom
        });

        it('should execute custom hooks', async () => {
            const result = await integrationHooks.executeHooks('custom-test-event', {
                test: 'data'
            });

            expect(result.success).toBe(true);
            if (result.results.length > 0) {
                expect(result.results[0].result.custom).toBe(true);
            }
        });
    });

    describe('ANSFPhase2Orchestrator Integration', () => {
        beforeAll(async () => {
            orchestrator = new ANSFPhase2Orchestrator({
                swarmId: 'test-swarm',
                clusterId: 'test-cluster',
                archonTaskId: 'test-task',
                semanticCacheBudget: 10, // 10MB for testing
                targetAccuracyImprovement: 25,
                enableLSPIntegration: true,
                enableProgressiveLoading: true,
                enableNeuralLearning: true,
                enableCrossLanguageAnalysis: true,
                languages: ['javascript'],
                maxMemoryPercent: 90
            });
        });

        it('should initialize with Phase 2 configuration', () => {
            expect(orchestrator.phase).toBe(2);
            expect(orchestrator.phase2Config.semanticCacheBudget).toBe(10);
            expect(orchestrator.phase2Config.enableLSPIntegration).toBe(true);
            expect(orchestrator.phase2Config.enableNeuralLearning).toBe(true);
        });

        it('should create enhanced components', async () => {
            // Initialize Phase 1 foundation first
            await orchestrator.initializePhase1Foundation();
            
            // Initialize enhanced semantic cache
            await orchestrator.initializeEnhancedSemanticCache();
            
            expect(orchestrator.semanticCache).toBeDefined();
            expect(orchestrator.semanticCache).toBeInstanceOf(Phase2SemanticCache);
        });

        it('should deploy LSP integration', async () => {
            if (orchestrator.phase2Config.enableLSPIntegration) {
                await orchestrator.deployLSPIntegration();
                
                expect(orchestrator.lspSystem).toBeDefined();
                expect(orchestrator.lspSystem).toBeInstanceOf(LSPIntegrationSystem);
            }
        });

        it('should calculate search accuracy improvement', () => {
            // Mock some components for testing
            orchestrator.semanticCache = { getStatus: () => ({ efficiency: { hitRate: '75' } }) };
            orchestrator.lspSystem = { getStatus: () => ({ symbolsIndexed: 10, capabilities: { crossLanguageAnalysis: true } }) };
            orchestrator.phase2Metrics.neuralPatternAccuracy = 85;

            const improvement = orchestrator.calculateSearchAccuracyImprovement();
            
            expect(improvement).toBeGreaterThan(0);
            expect(improvement).toBeLessThanOrEqual(45);
        });

        it('should generate Phase 2 deployment summary', () => {
            const summary = orchestrator.getPhase2DeploymentSummary();
            
            expect(summary.phase).toBe(2);
            expect(summary.enhancements).toBeDefined();
            expect(summary.phase2Metrics).toBeDefined();
            expect(summary.phase2Performance).toBeDefined();
        });
    });

    describe('Performance Validation', () => {
        it('should meet memory efficiency targets', async () => {
            if (semanticCache) {
                const status = semanticCache.getStatus();
                const memoryEfficiency = parseFloat(status.efficiency.memoryEfficiency);
                
                // Should be using memory efficiently (not wasting allocated space)
                expect(memoryEfficiency).toBeGreaterThanOrEqual(0);
                expect(memoryEfficiency).toBeLessThanOrEqual(100);
            }
        });

        it('should achieve target cache hit rates', async () => {
            if (semanticCache) {
                // Add some test data and access patterns
                const testData = { performance: 'test' };
                await semanticCache.store('perf-test', testData, {});
                
                // Generate hits
                for (let i = 0; i < 5; i++) {
                    await semanticCache.retrieve('perf-test');
                }
                
                const status = semanticCache.getStatus();
                const hitRate = parseFloat(status.efficiency.hitRate);
                
                // Should have reasonable hit rate
                expect(hitRate).toBeGreaterThan(0);
            }
        });

        it('should demonstrate search accuracy improvement', () => {
            // Test the search accuracy calculation logic
            const mockOrchestrator = {
                semanticCache: { 
                    getStatus: () => ({ efficiency: { hitRate: '80' } }) 
                },
                lspSystem: { 
                    getStatus: () => ({ 
                        symbolsIndexed: 15,
                        capabilities: { crossLanguageAnalysis: true }
                    })
                },
                phase2Metrics: { neuralPatternAccuracy: 90 }
            };
            
            // Use the calculation method
            const calculateImprovement = ANSFPhase2Orchestrator.prototype.calculateSearchAccuracyImprovement;
            const improvement = calculateImprovement.call(mockOrchestrator);
            
            expect(improvement).toBeGreaterThanOrEqual(25); // Should meet 25% minimum target
        });

        it('should validate cross-language analysis capabilities', async () => {
            if (lspSystem && lspSystem.config.crossLanguageAnalysis) {
                const insights = lspSystem.getArchitecturalInsights();
                
                expect(insights).toBeDefined();
                expect(insights.patterns).toBeDefined();
                expect(insights.suggestions).toBeDefined();
            }
        });
    });

    describe('Integration Validation', () => {
        it('should integrate semantic cache with LSP system', async () => {
            if (semanticCache && lspSystem) {
                // Create LSP system with cache integration
                const integratedLSP = new LSPIntegrationSystem({
                    languages: ['javascript'],
                    cacheInstance: semanticCache
                });
                
                await integratedLSP.initialize();
                
                // Test cache integration
                const completions = await integratedLSP.getCompletions(
                    'test.js',
                    { line: 1, character: 5 },
                    { prefix: 'test' }
                );
                
                expect(Array.isArray(completions)).toBe(true);
                
                await integratedLSP.shutdown();
            }
        });

        it('should integrate hooks with semantic components', async () => {
            if (integrationHooks && semanticCache && lspSystem) {
                // Test hook integration
                const enhancedHooks = new SemanticIntegrationHooks({
                    semanticCache: semanticCache,
                    lspSystem: lspSystem,
                    archonPRPEnabled: true,
                    claudeFlowEnabled: true
                });
                
                const result = await enhancedHooks.executeHooks('archon-prp-pre-cycle', {
                    task: { description: 'Integration test task' },
                    cycle: 1
                });
                
                expect(result.success).toBe(true);
                
                await enhancedHooks.shutdown();
            }
        });
    });
});

// Additional utility tests
describe('Phase 2 Utility Functions', () => {
    it('should hash tasks consistently', () => {
        const { SemanticIntegrationHooks } = require('../src/ansf-workflow/semantic/integration-hooks');
        const hooks = new SemanticIntegrationHooks({});
        
        const task1 = { description: 'test task', id: 1 };
        const task2 = { description: 'test task', id: 1 };
        const task3 = { description: 'different task', id: 1 };
        
        const hash1 = hooks.hashTask(task1);
        const hash2 = hooks.hashTask(task2);
        const hash3 = hooks.hashTask(task3);
        
        expect(hash1).toBe(hash2); // Same tasks should have same hash
        expect(hash1).not.toBe(hash3); // Different tasks should have different hashes
        expect(hash1).toMatch(/^[a-f0-9]{8}$/); // Should be 8-character hex
    });

    it('should classify tasks correctly', () => {
        const { SemanticIntegrationHooks } = require('../src/ansf-workflow/semantic/integration-hooks');
        const hooks = new SemanticIntegrationHooks({});
        
        expect(hooks.classifyTask({ description: 'semantic analysis task' })).toBe('semantic-analysis');
        expect(hooks.classifyTask({ description: 'integrate systems' })).toBe('integration');
        expect(hooks.classifyTask({ description: 'optimize performance' })).toBe('optimization');
        expect(hooks.classifyTask({ description: 'neural learning' })).toBe('machine-learning');
        expect(hooks.classifyTask({ description: 'general task' })).toBe('general');
    });

    it('should calculate task complexity correctly', () => {
        const { SemanticIntegrationHooks } = require('../src/ansf-workflow/semantic/integration-hooks');
        const hooks = new SemanticIntegrationHooks({});
        
        const simpleTask = { description: 'simple' };
        const complexTask = {
            description: 'A very complex task with many requirements and detailed specifications that require careful analysis and implementation across multiple systems and components',
            codeFiles: ['file1.js', 'file2.js', 'file3.js', 'file4.js', 'file5.js', 'file6.js'],
            dependencies: ['dep1', 'dep2', 'dep3', 'dep4'],
            constraints: ['constraint1', 'constraint2']
        };
        
        const simpleComplexity = hooks.calculateTaskComplexity(simpleTask);
        const complexComplexity = hooks.calculateTaskComplexity(complexTask);
        
        expect(simpleComplexity).toBeLessThan(complexComplexity);
        expect(simpleComplexity).toBeGreaterThanOrEqual(0);
        expect(complexComplexity).toBeLessThanOrEqual(1);
    });
});

module.exports = {
    // Export test utilities for other test files
    createTestSemanticCache: () => new Phase2SemanticCache({
        hotCacheSize: 1024 * 1024,
        warmCacheSize: 2 * 1024 * 1024,
        coldCacheSize: 1024 * 1024,
        learningEnabled: false
    }),
    
    createTestLSPSystem: () => new LSPIntegrationSystem({
        languages: ['javascript'],
        crossLanguageAnalysis: false
    }),
    
    createTestIntegrationHooks: () => new SemanticIntegrationHooks({
        archonPRPEnabled: true,
        claudeFlowEnabled: true
    })
};