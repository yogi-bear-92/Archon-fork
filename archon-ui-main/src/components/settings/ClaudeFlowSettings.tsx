/**
 * Claude Flow Settings Component
 * 
 * Provides settings and controls for Claude Flow integration in Archon.
 */

import React, { useState, useEffect } from 'react';
import { Bot, Activity, Settings, Zap, AlertTriangle } from 'lucide-react';
import { SwarmControl, AgentSpawner } from '../ClaudeFlow';
import { claudeFlowService } from '../../services/claudeFlowService';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

interface ClaudeFlowSettingsProps {
  className?: string;
}

interface HealthStatus {
  status: string;
  service: string;
  timestamp: string;
}

export const ClaudeFlowSettings: React.FC<ClaudeFlowSettingsProps> = ({ 
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState<'swarm' | 'agents' | 'metrics'>('swarm');
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [isHealthy, setIsHealthy] = useState(false);
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    checkHealth();
    loadMetrics();
    
    // Set up periodic health checks
    const healthInterval = setInterval(checkHealth, 30000); // Every 30 seconds
    const metricsInterval = setInterval(loadMetrics, 60000); // Every minute
    
    return () => {
      clearInterval(healthInterval);
      clearInterval(metricsInterval);
    };
  }, []);

  const checkHealth = async () => {
    const response = await claudeFlowService.healthCheck();
    if (response.status === 'success' && response.data) {
      setHealthStatus(response.data);
      setIsHealthy(true);
    } else {
      setIsHealthy(false);
    }
  };

  const loadMetrics = async () => {
    const response = await claudeFlowService.getAgentMetrics();
    if (response.status === 'success') {
      setMetrics(response.data);
    }
  };

  const tabs = [
    { id: 'swarm' as const, name: 'Swarm Control', icon: Bot },
    { id: 'agents' as const, name: 'Agent Spawner', icon: Zap },
    { id: 'metrics' as const, name: 'Metrics', icon: Activity },
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Health Status */}
      <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-600/5 backdrop-blur-sm border border-blue-500/20 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
            <Settings className="w-6 h-6 text-blue-500 filter drop-shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Claude Flow Integration</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">SPARC methodology and swarm orchestration</p>
            {healthStatus && (
              <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                Last checked: {new Date(healthStatus.timestamp).toLocaleString()}
              </p>
            )}
          </div>
        </div>
        
        {/* Health Status Indicator */}
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg backdrop-blur-sm border ${
          isHealthy 
            ? 'bg-green-500/20 border-green-500/30 text-green-700 dark:text-green-300' 
            : 'bg-red-500/20 border-red-500/30 text-red-700 dark:text-red-300'
        }`}>
          {isHealthy ? (
            <Activity className="w-4 h-4" />
          ) : (
            <AlertTriangle className="w-4 h-4" />
          )}
          <span className="text-sm font-medium">
            {isHealthy ? 'Service Healthy' : 'Service Unavailable'}
          </span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <Card accentColor="blue" className="p-0">
        <div className="border-b border-blue-500/20 bg-gradient-to-r from-blue-500/5 to-blue-600/5">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.name}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {!isHealthy && (
            <div className="mb-6 p-4 rounded-xl bg-gradient-to-br from-red-500/10 to-red-600/5 backdrop-blur-sm border border-red-500/20 shadow-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-5 h-5 text-red-500 filter drop-shadow-[0_0_8px_rgba(239,68,68,0.6)]" />
                <h3 className="text-sm font-medium text-red-700 dark:text-red-300">Service Unavailable</h3>
              </div>
              <p className="text-sm text-red-600 dark:text-red-400">
                Claude Flow service is not responding. Please check that the Archon backend 
                is running and Claude Flow is properly configured.
              </p>
            </div>
          )}

          {/* Tab Content */}
          {activeTab === 'swarm' && (
            <SwarmControl 
              onSwarmInitialized={(sessionId) => {
                console.log('Swarm initialized:', sessionId);
              }}
            />
          )}

          {activeTab === 'agents' && (
            <AgentSpawner 
              onAgentsSpawned={(result) => {
                console.log('Agents spawned:', result);
              }}
            />
          )}

          {activeTab === 'metrics' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Performance Metrics</h3>
                
                {metrics ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(metrics.metrics || {}).map(([category, data]: [string, any]) => (
                      <div key={category} className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-600/5 backdrop-blur-sm border border-blue-500/20">
                        <h4 className="font-medium text-gray-800 dark:text-white mb-2 capitalize">
                          {category.replace('-', ' ')}
                        </h4>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {typeof data === 'object' ? (
                            <pre className="text-xs overflow-auto max-h-32 font-mono">
                              {JSON.stringify(data, null, 2)}
                            </pre>
                          ) : (
                            <span>{String(data)}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Activity className="w-8 h-8 mx-auto mb-2 text-blue-400 opacity-50" />
                    <p className="text-gray-600 dark:text-gray-400">No metrics available</p>
                    <p className="text-sm text-gray-500 dark:text-gray-500">Initialize a swarm to start collecting metrics</p>
                  </div>
                )}
              </div>

              {/* Refresh Button */}
              <Button
                onClick={loadMetrics}
                variant="outline"
                accentColor="blue"
                icon={<Activity className="w-4 h-4" />}
              >
                Refresh Metrics
              </Button>
            </div>
          )}
        </div>
      </Card>

      {/* Feature Information */}
      <div className="p-6 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-600/5 backdrop-blur-sm border border-blue-500/20 shadow-lg">
        <div className="flex items-start gap-3">
          <Bot className="w-6 h-6 text-blue-500 filter drop-shadow-[0_0_8px_rgba(59,130,246,0.6)] mt-0.5" />
          <div>
            <h3 className="text-lg font-medium text-gray-800 dark:text-white mb-2">
              About Claude Flow Integration
            </h3>
            <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
              <p>
                Claude Flow brings enterprise-grade AI orchestration to Archon with:
              </p>
              <ul className="list-disc ml-6 space-y-1">
                <li><strong>SPARC Methodology:</strong> Systematic development with Specification, Pseudocode, Architecture, Refinement, and Completion phases</li>
                <li><strong>Swarm Coordination:</strong> Multi-agent systems with adaptive topologies and intelligent coordination</li>
                <li><strong>Neural Learning:</strong> Pattern recognition and optimization based on successful workflows</li>
                <li><strong>Archon Integration:</strong> Native task management and knowledge base integration</li>
                <li><strong>Performance Monitoring:</strong> Real-time metrics and bottleneck analysis</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};