/**
 * Claude Flow Agent Spawner Component
 * 
 * Provides UI for spawning and managing Claude Flow agents
 * with Archon task integration.
 */

import React, { useState, useEffect } from 'react';
import { Bot, Play, Target, Layers, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { claudeFlowService, AgentSpawnConfig } from '../../services/claudeFlowService';
import { Button } from '../ui/Button';

interface AgentSpawnerProps {
  currentTaskId?: string;
  onAgentsSpawned?: (result: any) => void;
  className?: string;
}

interface AgentTypes {
  [category: string]: string[];
}

const STRATEGY_DESCRIPTIONS = {
  development: 'Code development, testing, and implementation',
  research: 'Information gathering and analysis', 
  analysis: 'Code review, performance analysis, and optimization',
  testing: 'Test creation, validation, and quality assurance',
  optimization: 'Performance tuning and resource optimization'
};

export const AgentSpawner: React.FC<AgentSpawnerProps> = ({
  currentTaskId,
  onAgentsSpawned,
  className = ''
}) => {
  const [config, setConfig] = useState<AgentSpawnConfig>({
    objective: '',
    agents: [],
    strategy: 'development',
  });
  
  const [agentTypes, setAgentTypes] = useState<AgentTypes>({});
  const [isSpawning, setIsSpawning] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState<string[]>(['core']);
  const [selectedCategory, setSelectedCategory] = useState<string>('core');

  // Load available agent types
  useEffect(() => {
    loadAgentTypes();
  }, []);

  // Set task ID when provided
  useEffect(() => {
    if (currentTaskId) {
      setConfig(prev => ({ ...prev, archonTaskId: currentTaskId }));
    }
  }, [currentTaskId]);

  const loadAgentTypes = async () => {
    const response = await claudeFlowService.getAgentTypes();
    if (response.status === 'success' && response.data) {
      setAgentTypes(response.data);
    }
  };

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => 
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const selectAgent = (agent: string) => {
    setConfig(prev => ({
      ...prev,
      agents: prev.agents.includes(agent)
        ? prev.agents.filter(a => a !== agent)
        : [...prev.agents, agent]
    }));
  };

  const handleSpawnAgents = async () => {
    if (!config.objective.trim() || config.agents.length === 0) {
      return;
    }

    setIsSpawning(true);
    try {
      const response = await claudeFlowService.spawnAgents(config);
      
      if (response.status === 'success') {
        if (onAgentsSpawned) {
          onAgentsSpawned(response.data);
        }
        // Reset form
        setConfig(prev => ({
          ...prev,
          objective: '',
          agents: []
        }));
      } else {
        console.error('Failed to spawn agents:', response.error);
      }
    } catch (error) {
      console.error('Error spawning agents:', error);
    } finally {
      setIsSpawning(false);
    }
  };

  const getSelectedAgentCount = () => config.agents.length;
  const getCategoryAgentCount = (category: string) => {
    return agentTypes[category]?.filter(agent => config.agents.includes(agent)).length || 0;
  };

  return (
    <div className={className}>
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
          <Bot className="w-5 h-5 text-blue-500 filter drop-shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Spawn Agents</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">Deploy AI agents for specific objectives</p>
        </div>
      </div>

      {/* Objective Input */}
      <div className="mb-6">
        <label className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
          <Target className="w-4 h-4" />
          Objective
        </label>
        <div className="backdrop-blur-md bg-gradient-to-b from-white/80 to-white/60 dark:from-white/10 dark:to-black/30 border border-gray-200 dark:border-gray-700 rounded-md px-3 py-2 transition-all duration-200 focus-within:border-blue-500 focus-within:shadow-[0_0_15px_rgba(59,130,246,0.4)]">
          <textarea
            value={config.objective}
            onChange={(e) => setConfig({ ...config, objective: e.target.value })}
            placeholder="Describe what you want the agents to accomplish..."
            className="w-full bg-transparent text-gray-800 dark:text-white placeholder:text-gray-400 dark:placeholder:text-gray-600 focus:outline-none resize-none"
            rows={3}
          />
        </div>
      </div>

      {/* Strategy Selection */}
      <div className="mb-6">
        <label className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
          <Layers className="w-4 h-4" />
          Strategy
        </label>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(STRATEGY_DESCRIPTIONS).map(([strategy, description]) => (
            <label
              key={strategy}
              className={`p-3 rounded-xl cursor-pointer transition-all duration-200 backdrop-blur-sm ${
                config.strategy === strategy
                  ? 'bg-gradient-to-br from-blue-500/20 to-blue-600/10 border border-blue-500/30 shadow-lg'
                  : 'bg-gradient-to-br from-gray-500/10 to-gray-600/5 border border-gray-200 dark:border-gray-700 hover:from-blue-500/10 hover:to-blue-600/5 hover:border-blue-500/20'
              }`}
            >
              <input
                type="radio"
                name="strategy"
                value={strategy}
                checked={config.strategy === strategy}
                onChange={(e) => setConfig({ ...config, strategy: e.target.value as any })}
                className="sr-only"
              />
              <div className="font-medium text-sm text-gray-800 dark:text-white capitalize mb-1">
                {strategy}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">{description}</div>
            </label>
          ))}
        </div>
      </div>

      {/* Agent Selection */}
      <div className="mb-6">
        <label className="flex items-center justify-between text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">
          <span className="flex items-center gap-2">
            <Bot className="w-4 h-4" />
            Select Agents ({getSelectedAgentCount()} selected)
          </span>
        </label>

        <div className="space-y-2 max-h-60 overflow-y-auto rounded-xl bg-gradient-to-br from-gray-500/10 to-gray-600/5 backdrop-blur-sm border border-gray-200 dark:border-gray-700">
          {Object.entries(agentTypes).map(([category, agents]) => {
            const isExpanded = expandedCategories.includes(category);
            const selectedCount = getCategoryAgentCount(category);
            
            return (
              <div key={category} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                <button
                  onClick={() => toggleCategory(category)}
                  className="w-full flex items-center justify-between p-3 text-left hover:bg-blue-500/10 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-800 dark:text-white capitalize">
                      {category}
                    </span>
                    {selectedCount > 0 && (
                      <span className="px-2 py-1 text-xs bg-blue-500/20 text-blue-700 dark:text-blue-300 rounded-full border border-blue-500/30">
                        {selectedCount}
                      </span>
                    )}
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                  )}
                </button>

                {isExpanded && (
                  <div className="px-3 pb-3">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      {agents.map((agent) => (
                        <label
                          key={agent}
                          className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer text-sm transition-all duration-200 ${
                            config.agents.includes(agent)
                              ? 'bg-blue-500/20 text-blue-700 dark:text-blue-300 border border-blue-500/30'
                              : 'hover:bg-gray-100 dark:hover:bg-gray-800 border border-transparent'
                          }`}
                        >
                          <div className="relative">
                            <input
                              type="checkbox"
                              checked={config.agents.includes(agent)}
                              onChange={() => selectAgent(agent)}
                              className="sr-only peer"
                            />
                            <div className="relative w-4 h-4 rounded transition-all duration-200 cursor-pointer
                              bg-gradient-to-b from-white/80 to-white/60 dark:from-white/5 dark:to-black/40
                              border border-gray-300 dark:border-gray-700
                              peer-checked:border-blue-500 dark:peer-checked:border-blue-500/50
                              peer-checked:bg-gradient-to-b peer-checked:from-blue-500/20 peer-checked:to-blue-600/20
                              hover:border-blue-500/50 dark:hover:border-blue-500/30
                              peer-checked:shadow-[0_0_8px_rgba(59,130,246,0.2)] dark:peer-checked:shadow-[0_0_10px_rgba(59,130,246,0.3)]"
                            >
                              <svg
                                className={`w-3 h-3 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-200 text-blue-500 pointer-events-none ${
                                  config.agents.includes(agent) ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
                                }`}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                          </div>
                          <span className="truncate">{agent}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Archon Integration Info */}
      {currentTaskId && (
        <div className="mb-6 p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-600/5 backdrop-blur-sm border border-blue-500/20 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium text-gray-800 dark:text-white">Archon Integration</span>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Agents will be linked to task: <code className="bg-blue-500/20 px-2 py-1 rounded text-blue-700 dark:text-blue-300 font-mono text-xs">{currentTaskId}</code>
          </p>
        </div>
      )}

      {/* Action Button */}
      <Button
        onClick={handleSpawnAgents}
        disabled={!config.objective.trim() || config.agents.length === 0 || isSpawning}
        variant="primary"
        accentColor="blue"
        size="lg"
        className="w-full"
        icon={isSpawning ? (
          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        ) : (
          <Play className="w-4 h-4" />
        )}
      >
        {isSpawning ? 'Spawning Agents...' : `Spawn ${getSelectedAgentCount()} Agents`}
      </Button>

      {/* Quick Tips */}
      <div className="mt-6 p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-600/5 backdrop-blur-sm border border-blue-500/20 shadow-lg">
        <h5 className="text-sm font-medium text-gray-800 dark:text-white mb-2">Tips for Better Results</h5>
        <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
          <li>• Be specific about your objective and expected outcomes</li>
          <li>• Choose agents that complement each other's capabilities</li>
          <li>• Use 3-5 agents for most tasks to balance coverage and coordination</li>
          <li>• Enable Archon integration for automatic task updates</li>
        </ul>
      </div>
    </div>
  );
};