#!/usr/bin/env node

/**
 * Lunar-Inspired API Gateway for MCP Integration
 * 
 * Based on Lunar.dev architecture patterns:
 * - Centralized MCP Aggregation (inspired by Lunar MCPX)
 * - Advanced Traffic Shaping (inspired by Lunar Proxy)
 * - AI-Aware Policy Enforcement
 * - Live API Traffic Visibility
 * 
 * Integrates: Claude Flow + Flow-Nexus + Serena + Archon PRP
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import fetch from 'node-fetch';
import { EventEmitter } from 'events';
import { promisify } from 'util';
import { exec } from 'child_process';

const execAsync = promisify(exec);

// Lunar-inspired configuration
const GATEWAY_CONFIG = {
  name: 'Lunar-Inspired MCP Gateway',
  version: '1.0.0',
  port: 8050,
  
  // Traffic Management (Lunar Proxy inspired)
  trafficControl: {
    globalRateLimit: 100, // requests per minute
    retryAttempts: 3,
    retryBackoff: 'exponential',
    circuitBreakerThreshold: 5,
    timeoutMs: 10000
  },
  
  // Service Registry (Lunar MCPX inspired) 
  services: {
    archon: {
      name: 'Archon PRP',
      baseUrl: 'http://127.0.0.1:8181',
      healthEndpoint: '/health',
      priority: 1,
      rateLimit: 50, // per minute
      tools: ['archon_project_create', 'archon_rag_query', 'archon_task_create']
    },
    claudeFlow: {
      name: 'Claude Flow',
      command: 'npx claude-flow@alpha',
      priority: 2,
      rateLimit: 30,
      tools: ['claude_flow_sparc_execute', 'claude_flow_swarm_init']
    },
    serena: {
      name: 'Serena Intelligence',
      port: 8054,
      priority: 3,
      rateLimit: 40,
      tools: ['serena_analyze_code', 'serena_semantic_search']
    }
  }
};

// Lunar-inspired Traffic Control System
class TrafficController extends EventEmitter {
  constructor() {
    super();
    this.requestCounts = new Map();
    this.circuitBreakers = new Map();
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageLatency: 0,
      tokenUsage: 0
    };
  }

  async checkRateLimit(serviceId, userId = 'default') {
    const key = `${serviceId}:${userId}`;
    const now = Date.now();
    const windowStart = now - 60000; // 1 minute window
    
    if (!this.requestCounts.has(key)) {
      this.requestCounts.set(key, []);
    }
    
    const requests = this.requestCounts.get(key);
    const recentRequests = requests.filter(time => time > windowStart);
    
    const service = GATEWAY_CONFIG.services[serviceId];
    const limit = service?.rateLimit || GATEWAY_CONFIG.trafficControl.globalRateLimit;
    
    if (recentRequests.length >= limit) {
      throw new Error(`Rate limit exceeded for ${serviceId}: ${recentRequests.length}/${limit} requests per minute`);
    }
    
    recentRequests.push(now);
    this.requestCounts.set(key, recentRequests);
    return true;
  }

  checkCircuitBreaker(serviceId) {
    const breaker = this.circuitBreakers.get(serviceId);
    if (!breaker) {
      this.circuitBreakers.set(serviceId, { 
        failures: 0, 
        lastFailure: null, 
        state: 'closed' 
      });
      return true;
    }

    const { failures, lastFailure, state } = breaker;
    const threshold = GATEWAY_CONFIG.trafficControl.circuitBreakerThreshold;
    const resetTime = 60000; // 1 minute

    if (state === 'open') {
      if (Date.now() - lastFailure > resetTime) {
        breaker.state = 'half-open';
        this.emit('circuitBreakerHalfOpen', serviceId);
        return true;
      }
      throw new Error(`Circuit breaker open for ${serviceId}`);
    }

    if (failures >= threshold) {
      breaker.state = 'open';
      breaker.lastFailure = Date.now();
      this.emit('circuitBreakerOpen', serviceId);
      throw new Error(`Circuit breaker opened for ${serviceId} after ${failures} failures`);
    }

    return true;
  }

  recordSuccess(serviceId, latency) {
    this.metrics.totalRequests++;
    this.metrics.successfulRequests++;
    this.updateAverageLatency(latency);
    
    // Reset circuit breaker on success
    const breaker = this.circuitBreakers.get(serviceId);
    if (breaker) {
      breaker.failures = 0;
      if (breaker.state === 'half-open') {
        breaker.state = 'closed';
        this.emit('circuitBreakerClosed', serviceId);
      }
    }
    
    this.emit('requestSuccess', { serviceId, latency });
  }

  recordFailure(serviceId, error) {
    this.metrics.totalRequests++;
    this.metrics.failedRequests++;
    
    // Increment circuit breaker failures
    const breaker = this.circuitBreakers.get(serviceId) || { failures: 0, state: 'closed' };
    breaker.failures++;
    breaker.lastFailure = Date.now();
    this.circuitBreakers.set(serviceId, breaker);
    
    this.emit('requestFailure', { serviceId, error });
  }

  updateAverageLatency(latency) {
    const total = this.metrics.successfulRequests;
    const current = this.metrics.averageLatency;
    this.metrics.averageLatency = (current * (total - 1) + latency) / total;
  }

  getMetrics() {
    return {
      ...this.metrics,
      successRate: this.metrics.totalRequests > 0 
        ? (this.metrics.successfulRequests / this.metrics.totalRequests * 100).toFixed(2)
        : 0,
      circuitBreakerStates: Object.fromEntries(this.circuitBreakers)
    };
  }
}

// Lunar-inspired Service Discovery and Health Management
class ServiceRegistry extends EventEmitter {
  constructor(trafficController) {
    super();
    this.trafficController = trafficController;
    this.serviceStates = new Map();
    this.healthCheckInterval = 30000; // 30 seconds
    this.startHealthChecks();
  }

  async checkServiceHealth(serviceId, config) {
    try {
      if (config.baseUrl && config.healthEndpoint) {
        const response = await fetch(`${config.baseUrl}${config.healthEndpoint}`, {
          timeout: 5000
        });
        return { healthy: response.ok, latency: Date.now() - response.startTime };
      } else if (config.command) {
        const start = Date.now();
        await execAsync(`${config.command} --version 2>/dev/null || echo "available"`);
        return { healthy: true, latency: Date.now() - start };
      } else if (config.port) {
        // Simple port check
        try {
          const response = await fetch(`http://localhost:${config.port}`, { timeout: 2000 });
          return { healthy: true, latency: Date.now() - response.startTime };
        } catch (error) {
          return { healthy: false, error: error.message };
        }
      }
      return { healthy: true, latency: 0 };
    } catch (error) {
      return { healthy: false, error: error.message };
    }
  }

  startHealthChecks() {
    setInterval(async () => {
      for (const [serviceId, config] of Object.entries(GATEWAY_CONFIG.services)) {
        const health = await this.checkServiceHealth(serviceId, config);
        const previousHealth = this.serviceStates.get(serviceId)?.healthy;
        
        this.serviceStates.set(serviceId, {
          ...health,
          lastCheck: new Date().toISOString(),
          config
        });

        if (previousHealth !== health.healthy) {
          this.emit('serviceStateChange', { serviceId, health });
        }
      }
    }, this.healthCheckInterval);
  }

  getServiceState(serviceId) {
    return this.serviceStates.get(serviceId) || { healthy: false };
  }

  getAllServiceStates() {
    return Object.fromEntries(this.serviceStates);
  }

  selectService(serviceId) {
    const state = this.getServiceState(serviceId);
    if (!state.healthy) {
      throw new Error(`Service ${serviceId} is unhealthy`);
    }
    return GATEWAY_CONFIG.services[serviceId];
  }
}

// Main Lunar-Inspired MCP Gateway
class LunarInspiredGateway {
  constructor() {
    this.trafficController = new TrafficController();
    this.serviceRegistry = new ServiceRegistry(this.trafficController);
    this.server = new Server(
      {
        name: GATEWAY_CONFIG.name,
        version: GATEWAY_CONFIG.version,
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupEventListeners();
    this.setupMCPHandlers();
    
    console.log('🌙 Lunar-Inspired MCP Gateway initialized');
  }

  setupEventListeners() {
    // Traffic control events
    this.trafficController.on('circuitBreakerOpen', (serviceId) => {
      console.warn(`🔴 Circuit breaker opened for ${serviceId}`);
    });

    this.trafficController.on('circuitBreakerClosed', (serviceId) => {
      console.log(`🟢 Circuit breaker closed for ${serviceId}`);
    });

    // Service registry events  
    this.serviceRegistry.on('serviceStateChange', ({ serviceId, health }) => {
      const status = health.healthy ? '🟢 HEALTHY' : '🔴 UNHEALTHY';
      console.log(`${status} ${serviceId}: ${health.error || 'OK'}`);
    });
  }

  setupMCPHandlers() {
    // Resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'lunar://metrics/traffic',
            name: 'Traffic Metrics',
            description: 'Real-time API traffic metrics and performance data',
            mimeType: 'application/json',
          },
          {
            uri: 'lunar://services/health',
            name: 'Service Health',
            description: 'Current health status of all integrated services',
            mimeType: 'application/json',
          },
          {
            uri: 'lunar://policies/enforcement',
            name: 'Policy Enforcement',
            description: 'Current traffic policies and rate limits',
            mimeType: 'application/json',
          }
        ],
      };
    });

    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      switch (uri) {
        case 'lunar://metrics/traffic':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  timestamp: new Date().toISOString(),
                  metrics: this.trafficController.getMetrics(),
                  serviceStates: this.serviceRegistry.getAllServiceStates()
                }, null, 2),
              },
            ],
          };

        case 'lunar://services/health':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  services: this.serviceRegistry.getAllServiceStates(),
                  timestamp: new Date().toISOString()
                }, null, 2),
              },
            ],
          };

        case 'lunar://policies/enforcement':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  globalConfig: GATEWAY_CONFIG.trafficControl,
                  serviceConfigs: Object.fromEntries(
                    Object.entries(GATEWAY_CONFIG.services).map(([id, config]) => [
                      id, 
                      { 
                        rateLimit: config.rateLimit, 
                        priority: config.priority,
                        tools: config.tools
                      }
                    ])
                  )
                }, null, 2),
              },
            ],
          };

        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    });

    // Tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const tools = [
        {
          name: 'lunar_gateway_status',
          description: 'Get comprehensive gateway status including traffic metrics and service health',
          inputSchema: {
            type: 'object',
            properties: {
              includeMetrics: { type: 'boolean', default: true },
              includeServices: { type: 'boolean', default: true }
            }
          }
        },
        {
          name: 'lunar_policy_update',
          description: 'Update traffic policies and rate limits dynamically',
          inputSchema: {
            type: 'object',
            properties: {
              serviceId: { type: 'string' },
              policy: {
                type: 'object',
                properties: {
                  rateLimit: { type: 'number' },
                  priority: { type: 'number' }
                }
              }
            },
            required: ['serviceId', 'policy']
          }
        }
      ];

      // Add service-specific tools dynamically
      for (const [serviceId, config] of Object.entries(GATEWAY_CONFIG.services)) {
        const serviceState = this.serviceRegistry.getServiceState(serviceId);
        if (serviceState.healthy) {
          for (const toolName of config.tools) {
            tools.push({
              name: toolName,
              description: `${config.name} - ${toolName.replace(serviceId + '_', '').replace('_', ' ')}`,
              inputSchema: {
                type: 'object',
                properties: {
                  // Dynamic schema based on service
                },
                additionalProperties: true
              }
            });
          }
        }
      }

      return { tools };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args = {} } = request.params;
      const startTime = Date.now();

      try {
        if (name.startsWith('lunar_')) {
          return await this.handleLunarTool(name, args);
        }

        // Route to appropriate service
        const serviceId = this.detectServiceFromTool(name);
        if (!serviceId) {
          throw new Error(`Unknown tool: ${name}`);
        }

        // Apply traffic control
        await this.trafficController.checkRateLimit(serviceId);
        this.trafficController.checkCircuitBreaker(serviceId);

        const result = await this.executeServiceTool(serviceId, name, args);
        const latency = Date.now() - startTime;
        
        this.trafficController.recordSuccess(serviceId, latency);
        return result;

      } catch (error) {
        const serviceId = this.detectServiceFromTool(name);
        if (serviceId) {
          this.trafficController.recordFailure(serviceId, error);
        }
        
        return {
          content: [
            {
              type: 'text',
              text: `Gateway Error: ${error.message}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  detectServiceFromTool(toolName) {
    for (const [serviceId, config] of Object.entries(GATEWAY_CONFIG.services)) {
      if (config.tools.some(tool => toolName.startsWith(tool.split('_')[0]))) {
        return serviceId;
      }
    }
    return null;
  }

  async executeServiceTool(serviceId, toolName, args) {
    const service = this.serviceRegistry.selectService(serviceId);
    
    if (service.baseUrl) {
      // HTTP service call
      const endpoint = toolName.replace(`${serviceId}_`, '').replace('_', '/');
      const response = await fetch(`${service.baseUrl}/api/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(args),
        timeout: GATEWAY_CONFIG.trafficControl.timeoutMs
      });
      
      const result = await response.json();
      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
      };
    } else if (service.command) {
      // Command-line service
      const command = toolName.replace(`${serviceId}_`, '').replace('_', '-');
      const cmdArgs = Object.entries(args).map(([k, v]) => `--${k}=${v}`);
      
      const { stdout, stderr } = await execAsync(
        `${service.command} ${command} ${cmdArgs.join(' ')}`
      );
      
      return {
        content: [{ type: 'text', text: stdout || stderr || 'Command executed successfully' }],
      };
    }

    throw new Error(`Service ${serviceId} configuration not supported`);
  }

  async handleLunarTool(name, args) {
    switch (name) {
      case 'lunar_gateway_status':
        const status = {
          timestamp: new Date().toISOString(),
          gateway: GATEWAY_CONFIG.name,
          version: GATEWAY_CONFIG.version
        };

        if (args.includeMetrics !== false) {
          status.metrics = this.trafficController.getMetrics();
        }

        if (args.includeServices !== false) {
          status.services = this.serviceRegistry.getAllServiceStates();
        }

        return {
          content: [{ type: 'text', text: JSON.stringify(status, null, 2) }],
        };

      case 'lunar_policy_update':
        const { serviceId, policy } = args;
        if (GATEWAY_CONFIG.services[serviceId]) {
          Object.assign(GATEWAY_CONFIG.services[serviceId], policy);
          return {
            content: [{ 
              type: 'text', 
              text: `Policy updated for ${serviceId}: ${JSON.stringify(policy)}` 
            }],
          };
        }
        throw new Error(`Service ${serviceId} not found`);

      default:
        throw new Error(`Unknown lunar tool: ${name}`);
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('🌙 Lunar-Inspired MCP Gateway running on stdio transport');
    
    // Log initial service discovery
    setTimeout(() => {
      const services = this.serviceRegistry.getAllServiceStates();
      console.error(`📊 Discovered ${Object.keys(services).length} services:`);
      for (const [serviceId, state] of Object.entries(services)) {
        const status = state.healthy ? '🟢' : '🔴';
        console.error(`   ${status} ${serviceId}: ${state.error || 'OK'}`);
      }
    }, 2000);
  }
}

// Start the gateway
if (import.meta.url === `file://${process.argv[1]}`) {
  const gateway = new LunarInspiredGateway();
  gateway.run().catch(console.error);
}

export default LunarInspiredGateway;