/**
 * Claude Flow Swarm Control Component
 * 
 * Provides UI controls for initializing and managing Claude Flow swarms
 * with integrated Archon task management.
 */

import React, { useState, useEffect } from 'react';
import { Play, Settings, Activity, Users, Zap } from 'lucide-react';
import { claudeFlowService, SwarmConfig, SwarmStatus } from '../../services/claudeFlowService';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';

interface SwarmControlProps {
  onSwarmInitialized?: (sessionId: string) => void;
  className?: string;
}

export const SwarmControl: React.FC<SwarmControlProps> = ({ 
  onSwarmInitialized, 
  className = '' 
}) => {
  const [config, setConfig] = useState<SwarmConfig>({
    topology: 'adaptive',
    maxAgents: 10,
    archonIntegration: true,
  });
  
  const [status, setStatus] = useState<SwarmStatus | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isActive, setIsActive] = useState(false);
  
  // Load swarm status on component mount
  useEffect(() => {
    loadSwarmStatus();
  }, []);

  const loadSwarmStatus = async () => {
    const response = await claudeFlowService.getSwarmStatus();
    if (response.status === 'success' && response.data) {
      setStatus(response.data);
      setIsActive(response.data.configPresent && response.data.memoryAvailable);
    }
  };

  const handleInitializeSwarm = async () => {
    setIsInitializing(true);
    try {
      const response = await claudeFlowService.initializeSwarm(config);
      
      if (response.status === 'success') {
        await loadSwarmStatus();
        setIsActive(true);
        
        if (onSwarmInitialized && response.data?.session_id) {
          onSwarmInitialized(response.data.session_id);
        }
      } else {
        console.error('Failed to initialize swarm:', response.error);
      }
    } catch (error) {
      console.error('Error initializing swarm:', error);
    } finally {
      setIsInitializing(false);
    }
  };

  return (
    <div className={className}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-500/30">
            <Users className="w-5 h-5 text-blue-500 filter drop-shadow-[0_0_8px_rgba(59,130,246,0.6)]" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Claude Flow Swarm</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">Multi-agent orchestration with Archon integration</p>
          </div>
        </div>
        <div>
          {isActive && (
            <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-green-500/20 border border-green-500/30">
              <Activity className="w-4 h-4 text-green-500" />
              <span className="text-sm font-medium text-green-700 dark:text-green-300">Active</span>
            </div>
          )}
        </div>
      </div>

      {/* Configuration */}
      <div className="space-y-4 mb-6">
        <div>
          <Select
            label="Topology"
            value={config.topology}
            onChange={(e) => setConfig({
              ...config,
              topology: e.target.value as SwarmConfig['topology']
            })}
            accentColor="blue"
            disabled={isActive}
            options={[
              { value: 'adaptive', label: 'Adaptive (Recommended)' },
              { value: 'mesh', label: 'Mesh Network' },
              { value: 'hierarchical', label: 'Hierarchical' }
            ]}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
            Max Agents: <span className="font-semibold text-blue-600 dark:text-blue-400">{config.maxAgents}</span>
          </label>
          <input
            type="range"
            min="2"
            max="20"
            value={config.maxAgents}
            onChange={(e) => setConfig({
              ...config,
              maxAgents: parseInt(e.target.value)
            })}
            className="w-full h-2 bg-gradient-to-r from-gray-200 to-gray-300 dark:from-gray-700 dark:to-gray-600 rounded-lg appearance-none cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:shadow-[0_0_15px_rgba(59,130,246,0.4)]"
            style={{
              background: `linear-gradient(to right, rgb(59 130 246 / 0.4) 0%, rgb(59 130 246 / 0.4) ${((config.maxAgents - 2) / 18) * 100}%, rgb(156 163 175) ${((config.maxAgents - 2) / 18) * 100}%, rgb(156 163 175) 100%)`
            }}
            disabled={isActive}
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
            <span>2</span>
            <span>20</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <input
              type="checkbox"
              id="archon-integration"
              checked={config.archonIntegration}
              onChange={(e) => setConfig({
                ...config,
                archonIntegration: e.target.checked
              })}
              className="sr-only peer"
              disabled={isActive}
            />
            <label
              htmlFor="archon-integration"
              className="relative w-5 h-5 rounded-md transition-all duration-200 cursor-pointer
                bg-gradient-to-b from-white/80 to-white/60 dark:from-white/5 dark:to-black/40
                border border-gray-300 dark:border-gray-700
                peer-checked:border-blue-500 dark:peer-checked:border-blue-500/50
                peer-checked:bg-gradient-to-b peer-checked:from-blue-500/20 peer-checked:to-blue-600/20
                hover:border-blue-500/50 dark:hover:border-blue-500/30
                peer-checked:shadow-[0_0_10px_rgba(59,130,246,0.2)] dark:peer-checked:shadow-[0_0_15px_rgba(59,130,246,0.3)]
                peer-disabled:opacity-50 peer-disabled:cursor-not-allowed"
            >
              <svg
                className={`w-3.5 h-3.5 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 transition-all duration-200 text-blue-500 pointer-events-none ${
                  config.archonIntegration ? 'opacity-100 scale-100' : 'opacity-0 scale-50'
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </label>
          </div>
          <label htmlFor="archon-integration" className="text-sm text-gray-700 dark:text-gray-300 cursor-pointer">
            Enable Archon task integration
          </label>
        </div>
      </div>

      {/* Status Display */}
      {status && (
        <div className="p-4 mb-4 rounded-xl bg-gradient-to-br from-gray-500/10 to-gray-600/5 backdrop-blur-sm border border-gray-500/20">
          <h4 className="text-sm font-medium text-gray-800 dark:text-white mb-3">Swarm Status</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                status.memoryAvailable 
                  ? 'bg-green-400 shadow-[0_0_8px_rgba(34,197,94,0.6)]' 
                  : 'bg-red-400 shadow-[0_0_8px_rgba(239,68,68,0.6)]'
              }`}></div>
              <span className="text-gray-600 dark:text-gray-400">Memory</span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                status.configPresent 
                  ? 'bg-green-400 shadow-[0_0_8px_rgba(34,197,94,0.6)]' 
                  : 'bg-red-400 shadow-[0_0_8px_rgba(239,68,68,0.6)]'
              }`}></div>
              <span className="text-gray-600 dark:text-gray-400">Configuration</span>
            </div>
          </div>
          {status.timestamp && (
            <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
              Last updated: {new Date(status.timestamp).toLocaleString()}
            </p>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <Button
          onClick={handleInitializeSwarm}
          disabled={isInitializing || isActive}
          variant="primary"
          accentColor="blue"
          icon={isInitializing ? (
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <Play className="w-4 h-4" />
          )}
        >
          {isActive ? 'Swarm Active' : isInitializing ? 'Initializing...' : 'Initialize Swarm'}
        </Button>

        <Button
          onClick={loadSwarmStatus}
          variant="outline"
          accentColor="blue"
          icon={<Activity className="w-4 h-4" />}
        >
          Refresh Status
        </Button>
      </div>

      {/* Quick Tips */}
      <div className="mt-6 p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-blue-600/5 backdrop-blur-sm border border-blue-500/20 shadow-lg">
        <div className="flex items-start gap-3">
          <Zap className="w-5 h-5 text-blue-500 filter drop-shadow-[0_0_8px_rgba(59,130,246,0.6)] mt-0.5" />
          <div>
            <h5 className="text-sm font-medium text-gray-800 dark:text-white mb-1">Claude Flow Integration</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              With Archon integration enabled, agents will automatically coordinate with your 
              tasks and projects, using RAG queries for context-aware development.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};