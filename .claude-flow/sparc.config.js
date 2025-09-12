/**
 * SPARC Configuration for Archon Integration
 * Configures SPARC methodology with Archon PRP framework
 */

module.exports = {
  // SPARC workflow phases with ARCHON-FIRST integration
  phases: {
    specification: {
      agent: 'specification',
      description: 'Requirements analysis and specification',
      outputs: ['requirements.md', 'user-stories.md', 'acceptance-criteria.md'],
      archonIntegration: {
        required: true,
        taskManagementFirst: true,
        documentAgent: true,
        ragQuery: true,
        prePhase: [
          "archon:manage_task(action='list', filter_by='project')",
          "archon:perform_rag_query(query='requirements analysis patterns')"
        ],
        postPhase: [
          "archon:manage_task(action='update', status='review')"
        ]
      }
    },
    pseudocode: {
      agent: 'pseudocode', 
      description: 'Algorithm design and pseudocode',
      outputs: ['algorithms.md', 'data-structures.md', 'flow-diagrams.md'],
      dependencies: ['specification']
    },
    architecture: {
      agent: 'architecture',
      description: 'System design and architecture',
      outputs: ['architecture.md', 'system-design.md', 'api-spec.md'],
      dependencies: ['specification', 'pseudocode'],
      archonIntegration: {
        fastApiPatterns: true,
        pydanticaiIntegration: true,
        supabaseSchema: true
      }
    },
    refinement: {
      agent: 'refinement',
      description: 'TDD implementation and refinement',
      outputs: ['tests/', 'src/', 'documentation/'],
      dependencies: ['architecture'],
      archonIntegration: {
        prpCycles: 4,
        taskAgent: true,
        tddLondonSchool: true
      }
    },
    completion: {
      agent: 'sparc-coder',
      description: 'Final integration and completion',
      outputs: ['deployment/', 'monitoring/', 'docs/'],
      dependencies: ['refinement']
    }
  },

  // Agent coordination patterns
  coordination: {
    sparc_tdd: {
      topology: 'hierarchical',
      coordinator: 'sparc-coord',
      agents: ['specification', 'pseudocode', 'architecture', 'tdd-london-swarm', 'sparc-coder']
    },
    archon_prp: {
      topology: 'mesh',
      coordinator: 'archon_prp',
      agents: ['backend-dev', 'ml-developer', 'system-architect']
    },
    full_integration: {
      topology: 'adaptive',
      coordinator: 'adaptive-coordinator',
      agents: ['sparc-coord', 'archon_prp', 'code-review-swarm', 'performance-monitor']
    }
  },

  // Archon-specific configurations
  archon: {
    backend: {
      framework: 'fastapi',
      port: 8080,
      testFramework: 'pytest',
      linting: ['ruff', 'mypy']
    },
    agents: {
      port: 8052,
      framework: 'pydanticai',
      types: ['Document_Agent', 'RAG_Agent', 'Task_Agent']
    },
    mcp: {
      port: 8051,
      protocol: 'http',
      tools: 14
    },
    database: {
      type: 'supabase',
      extensions: ['pgvector'],
      migrations: ['1_initial_setup.sql', '2_archon_projects.sql', '3_mcp_client_management.sql']
    },
    frontend: {
      framework: 'react',
      port: 3737,
      build: 'vite',
      testing: 'vitest'
    }
  },

  // Performance targets
  performance: {
    queryTime: '200-300ms',
    tokenReduction: '32.3%',
    speedImprovement: '2.8-4.4x',
    sweSolveRate: '84.8%'
  },

  // Integration hooks
  hooks: {
    preTask: '.claude-flow/hooks.js pre-task',
    postEdit: '.claude-flow/hooks.js post-edit',
    postTask: '.claude-flow/hooks.js post-task',
    sessionEnd: '.claude-flow/hooks.js session-end'
  },

  // Memory and neural features
  neural: {
    enabled: true,
    patternLearning: true,
    crossSessionMemory: true,
    adaptiveSpawning: true,
    models: 27
  }
};