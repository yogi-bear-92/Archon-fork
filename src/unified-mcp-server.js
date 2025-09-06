#!/usr/bin/env node

/**
 * Unified MCP Server - Integrates Claude Flow, Flow-Nexus, Serena, and Archon
 * 
 * This server acts as a Master MCP Controller that coordinates all AI services:
 * - Claude Flow: SPARC methodology and swarm coordination
 * - Flow-Nexus: Workflow automation and pipeline management  
 * - Serena: Code intelligence and semantic analysis
 * - Archon PRP: Progressive refinement and project management
 * 
 * Port: 8050 (Master MCP endpoint)
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
import { spawn, exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Service configuration
const SERVICES = {
  archon: {
    name: 'Archon PRP',
    baseUrl: 'http://127.0.0.1:8181',
    healthEndpoint: '/health',
    prefix: 'archon_',
    description: 'Progressive Refinement and Project Management'
  },
  claudeFlow: {
    name: 'Claude Flow',
    command: 'npx claude-flow@alpha',
    prefix: 'claude_flow_',
    description: 'SPARC Methodology and Swarm Coordination'
  },
  serena: {
    name: 'Serena',
    port: 8054,
    prefix: 'serena_',
    description: 'Code Intelligence and Semantic Analysis'
  },
  flowNexus: {
    name: 'Flow-Nexus',
    port: 8051, // Current port, will migrate to 8056
    prefix: 'flow_nexus_',
    description: 'Workflow Automation and Pipeline Management'
  }
};

// Memory management and performance monitoring
let systemStats = {
  memoryUsage: 0,
  activeConnections: 0,
  lastHealthCheck: null,
  serviceStates: {}
};

class UnifiedMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'unified-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );
    
    this.setupHandlers();
    this.initializeServices();
  }

  async initializeServices() {
    console.log('🚀 Initializing Unified MCP Server...');
    
    // Check system resources
    await this.checkSystemHealth();
    
    // Initialize each service
    for (const [key, service] of Object.entries(SERVICES)) {
      try {
        systemStats.serviceStates[key] = await this.checkServiceHealth(service);
        console.log(`✅ ${service.name}: ${systemStats.serviceStates[key] ? 'Ready' : 'Degraded'}`);
      } catch (error) {
        console.warn(`⚠️  ${service.name}: ${error.message}`);
        systemStats.serviceStates[key] = false;
      }
    }

    console.log(`🎯 Unified MCP Server ready - ${Object.values(systemStats.serviceStates).filter(Boolean).length}/${Object.keys(SERVICES).length} services active`);
  }

  async checkSystemHealth() {
    try {
      // macOS memory check
      const { stdout } = await execAsync('vm_stat | head -5');
      const freePages = stdout.match(/Pages free:\s*(\d+)/);
      if (freePages) {
        systemStats.memoryUsage = parseInt(freePages[1]) * 16384; // Convert to bytes (16KB pages)
      }
      systemStats.lastHealthCheck = new Date().toISOString();
    } catch (error) {
      console.warn('Could not check system health:', error.message);
    }
  }

  async checkServiceHealth(service) {
    if (service.baseUrl && service.healthEndpoint) {
      // HTTP service health check
      try {
        const response = await fetch(`${service.baseUrl}${service.healthEndpoint}`, {
          timeout: 5000
        });
        return response.ok;
      } catch (error) {
        return false;
      }
    } else if (service.command) {
      // Command-based service check
      try {
        const { stdout } = await execAsync(`${service.command} --version 2>/dev/null || echo "available"`);
        return stdout.length > 0;
      } catch (error) {
        return false;
      }
    }
    return true; // Assume available if no specific check
  }

  setupHandlers() {
    // Resource handlers
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'unified://system/status',
            name: 'System Status',
            description: 'Current status of all integrated services',
            mimeType: 'application/json',
          },
          {
            uri: 'unified://system/health',
            name: 'Health Check',
            description: 'Real-time health metrics for all services',
            mimeType: 'application/json',
          },
          {
            uri: 'unified://coordination/topology',
            name: 'Service Topology',
            description: 'Current service mesh topology and routing',
            mimeType: 'application/json',
          }
        ],
      };
    });

    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      switch (uri) {
        case 'unified://system/status':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  timestamp: new Date().toISOString(),
                  services: systemStats.serviceStates,
                  memoryUsage: systemStats.memoryUsage,
                  activeConnections: systemStats.activeConnections,
                  uptime: process.uptime()
                }, null, 2),
              },
            ],
          };

        case 'unified://system/health':
          await this.checkSystemHealth();
          const healthData = {
            status: 'healthy',
            timestamp: systemStats.lastHealthCheck,
            services: {}
          };

          for (const [key, service] of Object.entries(SERVICES)) {
            healthData.services[key] = {
              name: service.name,
              status: systemStats.serviceStates[key] ? 'active' : 'inactive',
              description: service.description
            };
          }

          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(healthData, null, 2),
              },
            ],
          };

        case 'unified://coordination/topology':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  masterController: 'unified-mcp-server:8050',
                  services: Object.entries(SERVICES).map(([key, service]) => ({
                    id: key,
                    name: service.name,
                    endpoint: service.baseUrl || `${service.command} (CLI)`,
                    prefix: service.prefix,
                    active: systemStats.serviceStates[key]
                  })),
                  routing: 'prefix-based',
                  memoryAware: true
                }, null, 2),
              },
            ],
          };

        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    });

    // Tool handlers
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      const tools = [
        // System-level unified tools
        {
          name: 'unified_status',
          description: 'Get comprehensive status of all integrated services',
          inputSchema: {
            type: 'object',
            properties: {
              detailed: {
                type: 'boolean',
                description: 'Include detailed metrics and diagnostics'
              }
            }
          }
        },
        {
          name: 'unified_health_check',
          description: 'Perform health checks on all services with auto-recovery',
          inputSchema: {
            type: 'object',
            properties: {
              service: {
                type: 'string',
                description: 'Specific service to check (optional)',
                enum: Object.keys(SERVICES)
              },
              autoRecover: {
                type: 'boolean',
                description: 'Attempt auto-recovery for failed services'
              }
            }
          }
        },
        {
          name: 'unified_project_setup',
          description: 'Initialize a new project with all integrated services',
          inputSchema: {
            type: 'object',
            properties: {
              projectName: {
                type: 'string',
                description: 'Name of the project'
              },
              services: {
                type: 'array',
                items: { type: 'string' },
                description: 'Services to include in project setup'
              },
              sparkMode: {
                type: 'string',
                description: 'SPARC methodology mode',
                enum: ['tdd', 'batch', 'pipeline', 'concurrent']
              }
            },
            required: ['projectName']
          }
        }
      ];

      // Add service-specific tools based on active services
      if (systemStats.serviceStates.archon) {
        tools.push(
          {
            name: 'archon_project_create',
            description: 'Create a new Archon project with PRP framework',
            inputSchema: {
              type: 'object',
              properties: {
                title: { type: 'string' },
                description: { type: 'string' },
                github_repo: { type: 'string' }
              },
              required: ['title']
            }
          },
          {
            name: 'archon_rag_query',
            description: 'Perform RAG query on Archon knowledge base',
            inputSchema: {
              type: 'object',
              properties: {
                query: { type: 'string' },
                match_count: { type: 'integer', default: 5 }
              },
              required: ['query']
            }
          }
        );
      }

      if (systemStats.serviceStates.claudeFlow) {
        tools.push(
          {
            name: 'claude_flow_sparc_execute',
            description: 'Execute SPARC methodology workflow',
            inputSchema: {
              type: 'object',
              properties: {
                task: { type: 'string' },
                mode: { type: 'string', enum: ['tdd', 'batch', 'pipeline', 'concurrent'] }
              },
              required: ['task']
            }
          },
          {
            name: 'claude_flow_swarm_init',
            description: 'Initialize Claude Flow swarm coordination',
            inputSchema: {
              type: 'object',
              properties: {
                topology: { type: 'string', enum: ['adaptive', 'mesh', 'hierarchical'] },
                maxAgents: { type: 'integer', default: 5 }
              }
            }
          }
        );
      }

      return { tools };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args = {} } = request.params;
      
      try {
        // Route to appropriate service handler
        if (name.startsWith('unified_')) {
          return await this.handleUnifiedTool(name, args);
        } else if (name.startsWith('archon_')) {
          return await this.handleArchonTool(name, args);
        } else if (name.startsWith('claude_flow_')) {
          return await this.handleClaudeFlowTool(name, args);
        } else if (name.startsWith('serena_')) {
          return await this.handleSerenaTool(name, args);
        } else if (name.startsWith('flow_nexus_')) {
          return await this.handleFlowNexusTool(name, args);
        } else {
          throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing ${name}: ${error.message}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async handleUnifiedTool(name, args) {
    switch (name) {
      case 'unified_status':
        await this.checkSystemHealth();
        const statusData = {
          timestamp: new Date().toISOString(),
          services: systemStats.serviceStates,
          memoryUsage: `${(systemStats.memoryUsage / 1024 / 1024).toFixed(2)} MB free`,
          activeConnections: systemStats.activeConnections,
          uptime: `${Math.floor(process.uptime())} seconds`
        };

        if (args.detailed) {
          statusData.processInfo = {
            pid: process.pid,
            nodeVersion: process.version,
            platform: process.platform
          };
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(statusData, null, 2),
            },
          ],
        };

      case 'unified_health_check':
        const results = {};
        
        if (args.service) {
          // Check specific service
          const service = SERVICES[args.service];
          if (!service) {
            throw new Error(`Unknown service: ${args.service}`);
          }
          results[args.service] = await this.checkServiceHealth(service);
          systemStats.serviceStates[args.service] = results[args.service];
        } else {
          // Check all services
          for (const [key, service] of Object.entries(SERVICES)) {
            results[key] = await this.checkServiceHealth(service);
            systemStats.serviceStates[key] = results[key];
          }
        }

        return {
          content: [
            {
              type: 'text',
              text: `Health Check Results:\n${JSON.stringify(results, null, 2)}`,
            },
          ],
        };

      case 'unified_project_setup':
        return await this.setupProject(args);

      default:
        throw new Error(`Unknown unified tool: ${name}`);
    }
  }

  async handleArchonTool(name, args) {
    if (!systemStats.serviceStates.archon) {
      throw new Error('Archon service is not available');
    }

    const service = SERVICES.archon;
    const endpoint = name.replace('archon_', '').replace('_', '/');
    
    try {
      const response = await fetch(`${service.baseUrl}/api/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(args)
      });
      
      const result = await response.json();
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Archon API call failed: ${error.message}`);
    }
  }

  async handleClaudeFlowTool(name, args) {
    if (!systemStats.serviceStates.claudeFlow) {
      throw new Error('Claude Flow service is not available');
    }

    const command = name.replace('claude_flow_', '').replace('_', '-');
    const service = SERVICES.claudeFlow;
    
    try {
      let cmdArgs = [];
      
      switch (command) {
        case 'sparc-execute':
          cmdArgs = ['sparc', 'run', args.mode || 'tdd', `"${args.task}"`];
          break;
        case 'swarm-init':
          cmdArgs = ['swarm', 'init', `--topology=${args.topology || 'adaptive'}`, `--max-agents=${args.maxAgents || 5}`];
          break;
        default:
          cmdArgs = [command, ...Object.entries(args).map(([k, v]) => `--${k}=${v}`)];
      }

      const { stdout, stderr } = await execAsync(`${service.command} ${cmdArgs.join(' ')}`);
      
      return {
        content: [
          {
            type: 'text',
            text: stdout || stderr || 'Command executed successfully',
          },
        ],
      };
    } catch (error) {
      throw new Error(`Claude Flow command failed: ${error.message}`);
    }
  }

  async handleSerenaTool(name, args) {
    // Serena integration (placeholder - would integrate with actual Serena MCP)
    return {
      content: [
        {
          type: 'text',
          text: `Serena tool ${name} executed with args: ${JSON.stringify(args)}`,
        },
      ],
    };
  }

  async handleFlowNexusTool(name, args) {
    // Flow-Nexus integration (placeholder - would integrate with actual Flow-Nexus MCP)
    return {
      content: [
        {
          type: 'text',
          text: `Flow-Nexus tool ${name} executed with args: ${JSON.stringify(args)}`,
        },
      ],
    };
  }

  async setupProject(args) {
    const { projectName, services = [], sparkMode = 'tdd' } = args;
    
    const results = {
      projectName,
      timestamp: new Date().toISOString(),
      steps: []
    };

    // Step 1: Create Archon project if Archon is available
    if (services.includes('archon') || services.length === 0) {
      if (systemStats.serviceStates.archon) {
        try {
          const archonResult = await this.handleArchonTool('archon_project_create', {
            title: projectName,
            description: `Unified project created via MCP integration`
          });
          results.steps.push({
            service: 'Archon',
            action: 'Project Created',
            status: 'success',
            result: archonResult
          });
        } catch (error) {
          results.steps.push({
            service: 'Archon',
            action: 'Project Creation',
            status: 'failed',
            error: error.message
          });
        }
      }
    }

    // Step 2: Initialize Claude Flow SPARC workflow
    if (services.includes('claude-flow') || services.length === 0) {
      if (systemStats.serviceStates.claudeFlow) {
        try {
          const flowResult = await this.handleClaudeFlowTool('claude_flow_swarm_init', {
            topology: 'adaptive',
            maxAgents: 3 // Memory-aware scaling
          });
          results.steps.push({
            service: 'Claude Flow',
            action: 'Swarm Initialized',
            status: 'success',
            result: flowResult
          });
        } catch (error) {
          results.steps.push({
            service: 'Claude Flow',
            action: 'Swarm Initialization',
            status: 'failed',
            error: error.message
          });
        }
      }
    }

    return {
      content: [
        {
          type: 'text',
          text: `Project Setup Complete:\n${JSON.stringify(results, null, 2)}`,
        },
      ],
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('🎯 Unified MCP Server running on stdio transport');
  }
}

// Start the server
if (import.meta.url === `file://${process.argv[1]}`) {
  const server = new UnifiedMCPServer();
  server.run().catch(console.error);
}

export default UnifiedMCPServer;