#!/usr/bin/env python3
"""
Simple validation script for the Master Agent implementation.
This script validates that all components can be imported and basic functionality works.
"""

import sys
import os

# Add the python/src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'src'))

def test_imports():
    """Test that all master agent components can be imported."""
    print("Testing imports...")
    
    try:
        from agents.master.master_agent import (
            MasterAgent,
            MasterAgentConfig,
            MasterAgentDependencies,
            QueryRequest,
            QueryResponse,
            QueryType,
            ProcessingStrategy
        )
        print("✅ Master agent imports successful")
    except ImportError as e:
        print(f"❌ Master agent import failed: {e}")
        return False
    
    try:
        from agents.master.capability_matrix import AgentCapabilityMatrix
        print("✅ Capability matrix import successful")
    except ImportError as e:
        print(f"❌ Capability matrix import failed: {e}")
        return False
    
    try:
        from agents.master.coordination_hooks import ClaudeFlowCoordinator
        print("✅ Coordination hooks import successful")
    except ImportError as e:
        print(f"❌ Coordination hooks import failed: {e}")
        return False
    
    try:
        from agents.master.fallback_strategies import FallbackManager
        print("✅ Fallback strategies import successful")
    except ImportError as e:
        print(f"❌ Fallback strategies import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from agents.master.master_agent import (
            MasterAgentConfig,
            QueryRequest,
            QueryType
        )
        from agents.master.capability_matrix import AgentCapabilityMatrix
        from agents.master.coordination_hooks import ClaudeFlowCoordinator
        from agents.master.fallback_strategies import FallbackManager
        
        # Test configuration
        config = MasterAgentConfig(
            model="openai:gpt-4o",
            rag_enabled=False,  # Disable for testing
            max_retries=1
        )
        print("✅ Configuration creation successful")
        
        # Test query request
        request = QueryRequest(
            query="Test query for validation",
            query_type=QueryType.CODING,
            max_agents=2
        )
        print("✅ Query request creation successful")
        
        # Test capability matrix
        matrix = AgentCapabilityMatrix()
        capabilities = matrix.get_capabilities()
        print(f"✅ Capability matrix loaded: {len(capabilities)} agents")
        
        # Test specific agent capability
        coder_cap = matrix.get_agent_capability("coder")
        if coder_cap:
            print("✅ Agent capability lookup successful")
        else:
            print("❌ Agent capability lookup failed")
            return False
        
        # Test query type routing
        coding_caps = matrix.get_capabilities_for_query_type(QueryType.CODING, max_results=3)
        print(f"✅ Query type routing successful: {len(coding_caps)} agents for coding")
        
        # Test coordinator
        coordinator = ClaudeFlowCoordinator()
        active_sessions = coordinator.get_active_sessions()
        print(f"✅ Coordinator initialization successful: {len(active_sessions)} active sessions")
        
        # Test fallback manager
        manager = FallbackManager()
        stats = manager.get_fallback_stats()
        print(f"✅ Fallback manager initialization successful: {len(manager.local_knowledge)} knowledge topics")
        
        # Test performance metrics update
        update_success = matrix.update_performance_metrics(
            agent_type="coder",
            success=True,
            response_time=15.5
        )
        print(f"✅ Performance metrics update successful: {update_success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enums_and_types():
    """Test enums and type definitions."""
    print("\nTesting enums and types...")
    
    try:
        from agents.master.capability_matrix import QueryType
        from agents.master.master_agent import ProcessingStrategy
        
        # Test QueryType enum
        assert QueryType.CODING == "coding"
        assert QueryType.RESEARCH == "research"
        assert QueryType.ANALYSIS == "analysis"
        print("✅ QueryType enum working correctly")
        
        # Test ProcessingStrategy enum
        assert ProcessingStrategy.SINGLE_AGENT == "single_agent"
        assert ProcessingStrategy.MULTI_AGENT == "multi_agent"
        assert ProcessingStrategy.RAG_ENHANCED == "rag_enhanced"
        print("✅ ProcessingStrategy enum working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Enums and types test failed: {e}")
        return False

def test_agent_capability_details():
    """Test detailed agent capability functionality."""
    print("\nTesting agent capability details...")
    
    try:
        from agents.master.capability_matrix import AgentCapabilityMatrix, QueryType
        
        matrix = AgentCapabilityMatrix()
        
        # Test getting capabilities for different query types
        query_types = [QueryType.CODING, QueryType.RESEARCH, QueryType.ANALYSIS, QueryType.COORDINATION]
        
        for query_type in query_types:
            caps = matrix.get_capabilities_for_query_type(query_type, max_results=2)
            print(f"✅ {query_type.value} query type: {len(caps)} capabilities")
        
        # Test coordination-compatible agents
        coord_agents = matrix.get_coordination_compatible_agents()
        print(f"✅ Coordination-compatible agents: {len(coord_agents)}")
        
        # Test domain-specific agents
        programming_agents = matrix.get_agents_by_domain("programming")
        print(f"✅ Programming domain agents: {len(programming_agents)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent capability details test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Master Agent Validation")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality Tests", test_basic_functionality),
        ("Enums and Types Tests", test_enums_and_types),
        ("Agent Capability Details Tests", test_agent_capability_details)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Master Agent implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)