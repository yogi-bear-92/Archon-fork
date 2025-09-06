/**
 * Claude Flow Integration Service
 * 
 * Provides TypeScript API interface for Claude Flow orchestration
 * integrated with Archon's backend services.
 */

export interface SwarmConfig {
  topology: 'adaptive' | 'mesh' | 'hierarchical';
  maxAgents: number;
  archonIntegration: boolean;
}

export interface AgentSpawnConfig {
  objective: string;
  agents: string[];
  strategy: 'development' | 'research' | 'analysis' | 'testing' | 'optimization';
  archonTaskId?: string;
}

export interface SparcWorkflowConfig {
  task: string;
  mode: 'tdd' | 'batch' | 'pipeline' | 'concurrent' | 'spec-pseudocode' | 'architect';
  archonProjectId?: string;
}

export interface SwarmStatus {
  timestamp: string;
  memoryAvailable: boolean;
  configPresent: boolean;
  rawStatus: string;
}

export interface AgentMetrics {
  [category: string]: any;
}

export interface ClaudeFlowResponse<T = any> {
  status: 'success' | 'error';
  message?: string;
  data?: T;
  error?: string;
}

class ClaudeFlowService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8181';
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<ClaudeFlowResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/claude-flow${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
      }

      return {
        status: 'success',
        data,
      };
    } catch (error) {
      console.error(`Claude Flow API error:`, error);
      return {
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  /**
   * Initialize Claude Flow swarm with Archon integration
   */
  async initializeSwarm(config: SwarmConfig): Promise<ClaudeFlowResponse> {
    return this.request('/swarm/init', {
      method: 'POST',
      body: JSON.stringify({
        topology: config.topology,
        max_agents: config.maxAgents,
        archon_integration: config.archonIntegration,
      }),
    });
  }

  /**
   * Spawn agents for a specific objective
   */
  async spawnAgents(config: AgentSpawnConfig): Promise<ClaudeFlowResponse> {
    return this.request('/agents/spawn', {
      method: 'POST',
      body: JSON.stringify({
        objective: config.objective,
        agents: config.agents,
        strategy: config.strategy,
        archon_task_id: config.archonTaskId,
      }),
    });
  }

  /**
   * Execute SPARC methodology workflow
   */
  async executeSparceWorkflow(config: SparcWorkflowConfig): Promise<ClaudeFlowResponse> {
    return this.request('/sparc/execute', {
      method: 'POST',
      body: JSON.stringify({
        task: config.task,
        mode: config.mode,
        archon_project_id: config.archonProjectId,
      }),
    });
  }

  /**
   * Get current swarm status and metrics
   */
  async getSwarmStatus(): Promise<ClaudeFlowResponse<SwarmStatus>> {
    return this.request('/status');
  }

  /**
   * Get agent performance metrics
   */
  async getAgentMetrics(): Promise<ClaudeFlowResponse<AgentMetrics>> {
    return this.request('/metrics');
  }

  /**
   * Execute Claude Flow hooks
   */
  async executeHook(hookName: string, context: Record<string, any>): Promise<ClaudeFlowResponse> {
    return this.request('/hooks/execute', {
      method: 'POST',
      body: JSON.stringify({
        hook_name: hookName,
        context,
      }),
    });
  }

  /**
   * Perform memory operations
   */
  async memoryOperation(operation: 'store' | 'retrieve' | 'search', key?: string, value?: any): Promise<ClaudeFlowResponse> {
    return this.request('/memory', {
      method: 'POST',
      body: JSON.stringify({
        operation,
        key,
        value,
      }),
    });
  }

  /**
   * Execute neural pattern training
   */
  async neuralTraining(patterns: any[], modelType: string = 'performance'): Promise<ClaudeFlowResponse> {
    return this.request('/neural/train', {
      method: 'POST',
      body: JSON.stringify({
        patterns,
        model_type: modelType,
      }),
    });
  }

  /**
   * Get available agent types
   */
  async getAgentTypes(): Promise<ClaudeFlowResponse<Record<string, string[]>>> {
    return this.request('/agents/types');
  }

  /**
   * Get available SPARC modes
   */
  async getSparcModes(): Promise<ClaudeFlowResponse<Record<string, string>>> {
    return this.request('/sparc/modes');
  }

  /**
   * Health check for Claude Flow service
   */
  async healthCheck(): Promise<ClaudeFlowResponse> {
    return this.request('/health');
  }
}

export const claudeFlowService = new ClaudeFlowService();