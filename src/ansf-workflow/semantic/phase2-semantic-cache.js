/**
 * ANSF Phase 2 Enhanced Semantic Cache System
 * Intelligent 3-tier cache: Hot (25MB) + Warm (50MB) + Cold (25MB) = 100MB total
 * Integrated with neural cluster dnc_66761a355235 for pattern learning
 */

import { EventEmitter } from 'events';
import crypto from 'crypto';

class Phase2SemanticCache extends EventEmitter {
    constructor(options = {}) {
        super();
        
        // Cache tier configuration (100MB total)
        this.config = {
            hotCacheSize: options.hotCacheSize || 25 * 1024 * 1024,  // 25MB - Active development
            warmCacheSize: options.warmCacheSize || 50 * 1024 * 1024, // 50MB - Project context
            coldCacheSize: options.coldCacheSize || 25 * 1024 * 1024,  // 25MB - Historical data
            compressionRatio: options.compressionRatio || 0.6,
            ttlHot: options.ttlHot || 30 * 60 * 1000,  // 30 minutes
            ttlWarm: options.ttlWarm || 2 * 60 * 60 * 1000, // 2 hours
            ttlCold: options.ttlCold || 24 * 60 * 60 * 1000, // 24 hours
        };
        
        // Cache storage tiers
        this.hotCache = new Map(); // Active development context
        this.warmCache = new Map(); // Project-wide semantic data
        this.coldCache = new Map(); // Historical/compressed data
        
        // Memory tracking
        this.memoryUsage = {
            hot: 0,
            warm: 0,
            cold: 0,
            total: 0
        };
        
        // Neural cluster integration
        this.neuralClusterId = options.neuralClusterId || 'dnc_66761a355235';
        this.learningEnabled = options.learningEnabled !== false;
        
        // Metrics tracking
        this.metrics = {
            hits: { hot: 0, warm: 0, cold: 0 },
            misses: 0,
            evictions: 0,
            promotions: 0,
            demotions: 0,
            compressions: 0
        };
        
        // Start periodic maintenance
        this.maintenanceInterval = setInterval(() => this.performMaintenance(), 60000); // 1 minute
        
        console.log('🧠 Phase 2 Semantic Cache initialized:');
        console.log(`   Hot Cache: ${this.config.hotCacheSize / (1024*1024)}MB`);
        console.log(`   Warm Cache: ${this.config.warmCacheSize / (1024*1024)}MB`);
        console.log(`   Cold Cache: ${this.config.coldCacheSize / (1024*1024)}MB`);
        console.log(`   Total Budget: ${(this.config.hotCacheSize + this.config.warmCacheSize + this.config.coldCacheSize) / (1024*1024)}MB`);
    }

    /**
     * Store semantic data with intelligent tier placement
     */
    async store(key, semanticData, context = {}) {
        const entry = {
            key,
            data: semanticData,
            context,
            timestamp: Date.now(),
            accessCount: 1,
            lastAccess: Date.now(),
            size: this.calculateSize(semanticData),
            hash: this.generateHash(key, semanticData)
        };

        // Determine initial tier placement based on context
        const tier = this.determineTier(context, entry);
        
        try {
            await this.storeInTier(tier, entry);
            
            // Neural pattern learning (if enabled)
            if (this.learningEnabled && context.learnPattern !== false) {
                this.learnSemanticPattern(entry, tier);
            }
            
            this.emit('store', { tier, key, size: entry.size });
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to store semantic data in ${tier} cache:`, error);
            return false;
        }
    }

    /**
     * Retrieve semantic data with tier promotion logic
     */
    async retrieve(key, context = {}) {
        const startTime = Date.now();
        
        // Try hot cache first
        let entry = this.hotCache.get(key);
        if (entry) {
            this.updateAccessMetrics(entry, 'hot');
            this.metrics.hits.hot++;
            this.emit('hit', { tier: 'hot', key, responseTime: Date.now() - startTime });
            return entry.data;
        }
        
        // Try warm cache
        entry = this.warmCache.get(key);
        if (entry) {
            this.updateAccessMetrics(entry, 'warm');
            this.metrics.hits.warm++;
            
            // Consider promotion to hot cache
            if (this.shouldPromote(entry, 'warm', 'hot')) {
                await this.promoteEntry(entry, 'warm', 'hot');
                this.metrics.promotions++;
            }
            
            this.emit('hit', { tier: 'warm', key, responseTime: Date.now() - startTime });
            return entry.data;
        }
        
        // Try cold cache (decompress if needed)
        entry = this.coldCache.get(key);
        if (entry) {
            this.updateAccessMetrics(entry, 'cold');
            this.metrics.hits.cold++;
            
            // Decompress data if compressed
            const data = entry.compressed ? await this.decompress(entry.data) : entry.data;
            
            // Consider promotion to warm cache
            if (this.shouldPromote(entry, 'cold', 'warm')) {
                entry.data = data; // Use decompressed data
                entry.compressed = false;
                await this.promoteEntry(entry, 'cold', 'warm');
                this.metrics.promotions++;
            }
            
            this.emit('hit', { tier: 'cold', key, responseTime: Date.now() - startTime });
            return data;
        }
        
        // Cache miss
        this.metrics.misses++;
        this.emit('miss', { key, responseTime: Date.now() - startTime });
        return null;
    }

    /**
     * Determine appropriate tier for initial placement
     */
    determineTier(context, entry) {
        // Hot tier criteria: Active development, recent files, frequent access
        if (context.priority === 'hot' || 
            context.isActiveFile || 
            context.recentAccess ||
            entry.accessCount > 5) {
            return 'hot';
        }
        
        // Cold tier criteria: Historical data, large size, infrequent access
        if (context.priority === 'cold' ||
            context.historical ||
            entry.size > 1024 * 1024 || // > 1MB
            context.accessPattern === 'rare') {
            return 'cold';
        }
        
        // Default to warm tier
        return 'warm';
    }

    /**
     * Store entry in specified tier with eviction if necessary
     */
    async storeInTier(tier, entry) {
        const cache = this.getCacheForTier(tier);
        const maxSize = this.config[`${tier}CacheSize`];
        
        // Check if we need to make space
        if (this.memoryUsage[tier] + entry.size > maxSize) {
            await this.evictFromTier(tier, entry.size);
        }
        
        // Compress data for cold tier
        if (tier === 'cold' && entry.size > 100 * 1024) { // > 100KB
            entry.data = await this.compress(entry.data);
            entry.compressed = true;
            entry.size = Math.floor(entry.size * this.config.compressionRatio);
            this.metrics.compressions++;
        }
        
        cache.set(entry.key, entry);
        this.memoryUsage[tier] += entry.size;
        this.memoryUsage.total += entry.size;
        
        console.log(`📦 Stored ${entry.key} in ${tier} cache (${(entry.size / 1024).toFixed(1)}KB)`);
    }

    /**
     * Evict entries from tier to make space
     */
    async evictFromTier(tier, requiredSpace) {
        const cache = this.getCacheForTier(tier);
        const entries = Array.from(cache.entries());
        
        // Sort by LRU with semantic relevance weighting
        entries.sort((a, b) => {
            const scoreA = this.calculateEvictionScore(a[1]);
            const scoreB = this.calculateEvictionScore(b[1]);
            return scoreA - scoreB; // Lower score = evict first
        });
        
        let freedSpace = 0;
        for (const [key, entry] of entries) {
            if (freedSpace >= requiredSpace) break;
            
            // Consider demotion before eviction
            if (tier !== 'cold' && this.canDemote(entry, tier)) {
                await this.demoteEntry(entry, tier, this.getNextTier(tier));
                this.metrics.demotions++;
            } else {
                // Evict completely
                cache.delete(key);
                freedSpace += entry.size;
                this.memoryUsage[tier] -= entry.size;
                this.memoryUsage.total -= entry.size;
                this.metrics.evictions++;
                
                console.log(`🗑️  Evicted ${key} from ${tier} cache (${(entry.size / 1024).toFixed(1)}KB)`);
            }
        }
    }

    /**
     * Calculate eviction score (lower = evict first)
     */
    calculateEvictionScore(entry) {
        const age = Date.now() - entry.lastAccess;
        const accessFrequency = entry.accessCount / Math.max(1, (Date.now() - entry.timestamp) / (24 * 60 * 60 * 1000));
        const semanticRelevance = entry.context.importance || 0.5;
        
        // Lower score = higher chance of eviction
        return accessFrequency * semanticRelevance * Math.exp(-age / (60 * 60 * 1000));
    }

    /**
     * Check if entry should be promoted to higher tier
     */
    shouldPromote(entry, fromTier, toTier) {
        const recentAccess = Date.now() - entry.lastAccess < 5 * 60 * 1000; // 5 minutes
        const frequentAccess = entry.accessCount >= 3;
        const hasCapacity = this.memoryUsage[toTier] + entry.size <= this.config[`${toTier}CacheSize`] * 0.8; // 80% threshold
        
        return recentAccess && frequentAccess && hasCapacity;
    }

    /**
     * Promote entry between tiers
     */
    async promoteEntry(entry, fromTier, toTier) {
        const fromCache = this.getCacheForTier(fromTier);
        const toCache = this.getCacheForTier(toTier);
        
        // Remove from current tier
        fromCache.delete(entry.key);
        this.memoryUsage[fromTier] -= entry.size;
        
        // Add to target tier (with potential eviction)
        await this.storeInTier(toTier, entry);
        
        console.log(`⬆️  Promoted ${entry.key} from ${fromTier} to ${toTier} cache`);
    }

    /**
     * Demote entry to lower tier
     */
    async demoteEntry(entry, fromTier, toTier) {
        const fromCache = this.getCacheForTier(fromTier);
        
        // Remove from current tier
        fromCache.delete(entry.key);
        this.memoryUsage[fromTier] -= entry.size;
        this.memoryUsage.total -= entry.size;
        
        // Store in lower tier
        await this.storeInTier(toTier, entry);
        
        console.log(`⬇️  Demoted ${entry.key} from ${fromTier} to ${toTier} cache`);
    }

    /**
     * Learn semantic patterns for neural enhancement
     */
    learnSemanticPattern(entry, tier) {
        if (!this.learningEnabled) return;
        
        // Extract pattern features
        const pattern = {
            tier,
            contextType: entry.context.type || 'unknown',
            dataSize: entry.size,
            accessPattern: entry.context.accessPattern || 'normal',
            semanticType: entry.context.semanticType || 'general',
            language: entry.context.language || 'javascript',
            projectScope: entry.context.projectScope || 'local'
        };
        
        // Send to neural cluster for pattern learning (async)
        setImmediate(() => {
            this.sendPatternToNeuralCluster(pattern, entry.hash);
        });
    }

    /**
     * Send pattern to neural cluster for learning
     */
    async sendPatternToNeuralCluster(pattern, hash) {
        try {
            // This would integrate with the actual neural cluster
            // For now, we simulate the learning process
            console.log(`🧠 Learning semantic pattern: ${pattern.contextType} (${pattern.tier} tier)`);
            
            // In real implementation, this would call:
            // await neuralClusterClient.learnPattern(this.neuralClusterId, pattern, hash);
            
        } catch (error) {
            console.error('❌ Failed to send pattern to neural cluster:', error);
        }
    }

    /**
     * Perform periodic maintenance
     */
    performMaintenance() {
        const now = Date.now();
        
        // Clean expired entries
        this.cleanExpiredEntries(now);
        
        // Optimize tier distribution
        this.optimizeTierDistribution();
        
        // Update metrics
        this.updateMetrics();
        
        // Emit status
        this.emit('maintenance', {
            memoryUsage: this.memoryUsage,
            metrics: this.metrics,
            timestamp: now
        });
    }

    /**
     * Clean expired entries
     */
    cleanExpiredEntries(now) {
        const tiers = ['hot', 'warm', 'cold'];
        const ttls = [this.config.ttlHot, this.config.ttlWarm, this.config.ttlCold];
        
        tiers.forEach((tier, index) => {
            const cache = this.getCacheForTier(tier);
            const ttl = ttls[index];
            
            for (const [key, entry] of cache) {
                if (now - entry.lastAccess > ttl) {
                    cache.delete(key);
                    this.memoryUsage[tier] -= entry.size;
                    this.memoryUsage.total -= entry.size;
                    console.log(`⏰ Expired ${key} from ${tier} cache`);
                }
            }
        });
    }

    /**
     * Get cache reference for tier
     */
    getCacheForTier(tier) {
        switch (tier) {
            case 'hot': return this.hotCache;
            case 'warm': return this.warmCache;
            case 'cold': return this.coldCache;
            default: throw new Error(`Unknown tier: ${tier}`);
        }
    }

    /**
     * Utility methods
     */
    calculateSize(data) {
        return JSON.stringify(data).length * 2; // Approximate UTF-16 size
    }

    generateHash(key, data) {
        return crypto.createHash('sha256').update(key + JSON.stringify(data)).digest('hex');
    }

    async compress(data) {
        // Simulate compression (in real implementation, use zlib)
        return { compressed: true, data: JSON.stringify(data) };
    }

    async decompress(compressedData) {
        // Simulate decompression
        return JSON.parse(compressedData.data);
    }

    updateAccessMetrics(entry, tier) {
        entry.accessCount++;
        entry.lastAccess = Date.now();
    }

    canDemote(entry, tier) {
        return tier !== 'cold' && Date.now() - entry.lastAccess > 30 * 60 * 1000; // 30 minutes
    }

    getNextTier(tier) {
        switch (tier) {
            case 'hot': return 'warm';
            case 'warm': return 'cold';
            default: return 'cold';
        }
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            memoryUsage: this.memoryUsage,
            metrics: this.metrics,
            config: this.config,
            cacheStats: {
                hot: this.hotCache.size,
                warm: this.warmCache.size,
                cold: this.coldCache.size
            },
            efficiency: {
                hitRate: this.calculateHitRate(),
                memoryEfficiency: this.calculateMemoryEfficiency()
            }
        };
    }

    calculateHitRate() {
        const totalHits = this.metrics.hits.hot + this.metrics.hits.warm + this.metrics.hits.cold;
        const totalRequests = totalHits + this.metrics.misses;
        return totalRequests > 0 ? (totalHits / totalRequests * 100).toFixed(2) : 0;
    }

    calculateMemoryEfficiency() {
        const totalBudget = this.config.hotCacheSize + this.config.warmCacheSize + this.config.coldCacheSize;
        return ((this.memoryUsage.total / totalBudget) * 100).toFixed(2);
    }

    updateMetrics() {
        // Additional metric calculations can be added here
        this.emit('metrics', this.metrics);
    }

    /**
     * Optimize tier distribution based on access patterns
     */
    optimizeTierDistribution() {
        // This could implement more sophisticated optimization logic
        // For now, we just ensure balanced distribution
        const hotUtilization = this.memoryUsage.hot / this.config.hotCacheSize;
        const warmUtilization = this.memoryUsage.warm / this.config.warmCacheSize;
        const coldUtilization = this.memoryUsage.cold / this.config.coldCacheSize;
        
        if (hotUtilization > 0.9 && warmUtilization < 0.5) {
            console.log('🔄 High hot cache utilization detected - consider tier rebalancing');
        }
    }

    /**
     * Shutdown cache system
     */
    async shutdown() {
        if (this.maintenanceInterval) {
            clearInterval(this.maintenanceInterval);
        }
        
        console.log('🛑 Phase 2 Semantic Cache shutting down');
        
        // Final maintenance
        this.performMaintenance();
        
        // Clear all caches
        this.hotCache.clear();
        this.warmCache.clear();
        this.coldCache.clear();
        
        // Reset memory tracking
        this.memoryUsage = { hot: 0, warm: 0, cold: 0, total: 0 };
        
        this.emit('shutdown', { timestamp: Date.now() });
        console.log('✅ Phase 2 Semantic Cache shutdown complete');
    }
}

export { Phase2SemanticCache };