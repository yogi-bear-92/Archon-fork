import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
        import psutil
"""
Unit tests for individual Serena MCP tool integrations.

Tests each Serena MCP tool in isolation to ensure correct functionality,
parameter handling, error cases, and response formatting.
"""

    SerenaTestData,
    serena_project_data,
    mock_serena_mcp_client,
    mock_async_http_client
)
    MockSerenaTools,
    MockHttpClient,
    time_test
)

class TestSerenaDirectoryOperations:
    """Test directory listing and file discovery operations."""

    @pytest.mark.asyncio
    async def test_list_dir_basic(self, mock_serena_mcp_client):
        """Test basic directory listing functionality."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('list_dir', {
            "dirs": ["src", "tests", "docs"],
            "files": ["README.md", "requirements.txt", "setup.py"]
        })

        result = await mock_tools.list_dir(".", recursive=False)

        assert result["dirs"] == ["src", "tests", "docs"]
        assert result["files"] == ["README.md", "requirements.txt", "setup.py"]

        # Check call history
        history = mock_tools.get_call_history()
        assert len(history) == 1
        assert history[0]['tool'] == 'list_dir'
        assert history[0]['args'] == (".", False, -1)

    @pytest.mark.asyncio
    async def test_list_dir_recursive(self, mock_serena_mcp_client):
        """Test recursive directory listing."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('list_dir', {
            "dirs": ["src", "src/components", "src/utils"],
            "files": [
                "src/main.py",
                "src/components/ui.py",
                "src/utils/helpers.py"
            ]
        })

        result = await mock_tools.list_dir(".", recursive=True)

        assert "src/components" in result["dirs"]
        assert "src/components/ui.py" in result["files"]

        history = mock_tools.get_call_history()
        assert history[0]['args'][1] is True  # recursive=True

    @pytest.mark.asyncio
    async def test_find_file_pattern_matching(self, mock_serena_mcp_client):
        """Test file pattern matching."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('find_file', {
            "found_files": [
                "src/test_utils.py",
                "tests/test_main.py",
                "tests/integration/test_api.py"
            ]
        })

        result = await mock_tools.find_file("test_*.py", ".")

        assert len(result["found_files"]) == 3
        assert all("test_" in filename for filename in result["found_files"])

        history = mock_tools.get_call_history()
        assert history[0]['args'] == ("test_*.py", ".")

    @pytest.mark.asyncio
    async def test_find_file_no_matches(self, mock_serena_mcp_client):
        """Test file search with no matches."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('find_file', {"found_files": []})

        result = await mock_tools.find_file("nonexistent_*.xyz", ".")

        assert result["found_files"] == []

    @pytest.mark.asyncio
    async def test_list_dir_error_handling(self, mock_serena_mcp_client):
        """Test error handling in directory listing."""
        mock_tools = MockSerenaTools()
        mock_tools.set_error_condition('list_dir', FileNotFoundError("Directory not found"))

        with pytest.raises(FileNotFoundError, match="Directory not found"):
            await mock_tools.list_dir("/nonexistent/path")

        metrics = mock_tools.get_metrics()
        assert metrics['error_count'] == 1
        assert metrics['success_rate'] == 0.0

class TestSerenaSemanticAnalysis:
    """Test semantic analysis and symbol operations."""

    @pytest.mark.asyncio
    async def test_get_symbols_overview(self, serena_project_data):
        """Test getting symbols overview for a file."""
        mock_tools = MockSerenaTools()
        expected_symbols = SerenaTestData.create_semantic_analysis_results()
        mock_tools.set_response('get_symbols_overview', {
            "symbols": expected_symbols["functions"] + expected_symbols["classes"],
            "total_symbols": expected_symbols["symbols_found"]
        })

        result = await mock_tools.get_symbols_overview("src/main.py")

        assert result["total_symbols"] == expected_symbols["symbols_found"]
        assert len(result["symbols"]) > 0

        # Check that we have both functions and classes
        symbol_names = [s["name"] for s in result["symbols"]]
        assert "calculate_fibonacci" in symbol_names
        assert "MathUtils" in symbol_names

    @pytest.mark.asyncio
    async def test_find_symbol_by_name(self, mock_serena_mcp_client):
        """Test finding symbols by name pattern."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('find_symbol', [
            {
                "name": "calculate_fibonacci",
                "type": "function",
                "location": {"file": "src/main.py", "line": 2, "column": 1},
                "signature": "calculate_fibonacci(n: int) -> int",
                "docstring": "Calculate the nth Fibonacci number using recursion."
            }
        ])

        result = await mock_tools.find_symbol("calculate_fibonacci", relative_path="src")

        assert len(result) == 1
        symbol = result[0]
        assert symbol["name"] == "calculate_fibonacci"
        assert symbol["type"] == "function"
        assert "fibonacci" in symbol["docstring"].lower()

        history = mock_tools.get_call_history()
        assert history[0]['kwargs']['relative_path'] == "src"

    @pytest.mark.asyncio
    async def test_find_symbol_with_depth(self, mock_serena_mcp_client):
        """Test finding symbols with specified depth."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('find_symbol', [
            {
                "name": "MathUtils",
                "type": "class",
                "location": {"file": "src/main.py", "line": 8},
                "children": [
                    {"name": "is_prime", "type": "method"},
                    {"name": "factorial", "type": "method"}
                ]
            }
        ])

        result = await mock_tools.find_symbol("MathUtils", depth=1)

        assert len(result) == 1
        class_symbol = result[0]
        assert class_symbol["name"] == "MathUtils"
        assert len(class_symbol["children"]) == 2

        method_names = [child["name"] for child in class_symbol["children"]]
        assert "is_prime" in method_names
        assert "factorial" in method_names

    @pytest.mark.asyncio
    async def test_find_referencing_symbols(self, mock_serena_mcp_client):
        """Test finding symbols that reference a given symbol."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('find_referencing_symbols', [
            {
                "referencing_symbol": "main",
                "location": {"file": "src/app.py", "line": 15},
                "context": "result = calculate_fibonacci(10)",
                "reference_type": "function_call"
            },
            {
                "referencing_symbol": "test_fibonacci",
                "location": {"file": "tests/test_main.py", "line": 5},
                "context": "assert calculate_fibonacci(5) == 5",
                "reference_type": "function_call"
            }
        ])

        result = await mock_tools.find_referencing_symbols(
            "calculate_fibonacci",
            "src/main.py"
        )

        assert len(result) == 2
        assert all(ref["reference_type"] == "function_call" for ref in result)

        referencing_files = [ref["location"]["file"] for ref in result]
        assert "src/app.py" in referencing_files
        assert "tests/test_main.py" in referencing_files

    @pytest.mark.asyncio
    async def test_search_for_pattern_regex(self, mock_serena_mcp_client):
        """Test pattern search with regex."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('search_for_pattern', {
            "matches": {
                "src/main.py": [
                    {
                        "line_number": 5,
                        "content": "return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)",
                        "context_before": ["if n <= 1:", "    return n"],
                        "context_after": []
                    }
                ],
                "tests/test_main.py": [
                    {
                        "line_number": 10,
                        "content": "result = calculate_fibonacci(5)",
                        "context_before": ["def test_fibonacci():"],
                        "context_after": ["assert result == 5"]
                    }
                ]
            }
        })

        result = await mock_tools.search_for_pattern(
            r"calculate_fibonacci\(\d+\)",
            context_lines_before=2,
            context_lines_after=1
        )

        assert len(result["matches"]) == 2
        assert "src/main.py" in result["matches"]
        assert "tests/test_main.py" in result["matches"]

        # Check context lines were included
        main_match = result["matches"]["src/main.py"][0]
        assert len(main_match["context_before"]) == 2
        assert len(main_match["context_after"]) == 0  # None available in this case

        history = mock_tools.get_call_history()
        call_kwargs = history[0]['kwargs']
        assert call_kwargs['context_lines_before'] == 2
        assert call_kwargs['context_lines_after'] == 1

class TestSerenaCodeModification:
    """Test code modification operations."""

    @pytest.mark.asyncio
    async def test_replace_symbol_body(self, mock_serena_mcp_client):
        """Test replacing symbol body."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('replace_symbol_body', {
            "status": "success",
            "modified_file": "src/main.py",
            "symbol_name": "calculate_fibonacci",
            "lines_changed": 3
        })

        new_body = '''def calculate_fibonacci(n: int) -> int:
    """Optimized Fibonacci using memoization."""
    memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        result = n
    else:
        result = calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    memo[n] = result
    return result'''

        result = await mock_tools.replace_symbol_body(
            "calculate_fibonacci",
            "src/main.py",
            new_body
        )

        assert result["status"] == "success"
        assert result["modified_file"] == "src/main.py"
        assert result["lines_changed"] == 3

        history = mock_tools.get_call_history()
        assert history[0]['args'][0] == "calculate_fibonacci"
        assert history[0]['args'][1] == "src/main.py"
        assert "memoization" in history[0]['args'][2]

    @pytest.mark.asyncio
    async def test_insert_after_symbol(self, mock_serena_mcp_client):
        """Test inserting code after a symbol."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('insert_after_symbol', {
            "status": "success",
            "modified_file": "src/main.py",
            "insertion_line": 15,
            "lines_inserted": 5
        })

        new_function = '''

def fibonacci_iterative(n: int) -> int:
    """Calculate Fibonacci number iteratively."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b'''

        result = await mock_tools.insert_after_symbol(
            "calculate_fibonacci",
            "src/main.py",
            new_function
        )

        assert result["status"] == "success"
        assert result["insertion_line"] == 15
        assert result["lines_inserted"] == 5

    @pytest.mark.asyncio
    async def test_insert_before_symbol(self, mock_serena_mcp_client):
        """Test inserting code before a symbol."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('insert_before_symbol', {
            "status": "success",
            "modified_file": "src/main.py",
            "insertion_line": 1,
            "lines_inserted": 3
        })

        imports = '''from typing import Dict
'''

        result = await mock_tools.insert_before_symbol(
            "calculate_fibonacci",
            "src/main.py",
            imports
        )

        assert result["status"] == "success"
        assert result["insertion_line"] == 1
        assert result["lines_inserted"] == 3

class TestSerenaMemoryOperations:
    """Test memory persistence and retrieval operations."""

    @pytest.mark.asyncio
    async def test_write_memory(self, mock_serena_mcp_client):
        """Test writing data to memory."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('write_memory', {
            "status": "success",
            "memory_name": "semantic_analysis_cache",
            "content_size": 1024
        })

        memory_content = json.dumps({
            "project_id": "test_project",
            "analyzed_files": ["src/main.py", "src/utils.py"],
            "symbols_cache": {
                "functions": ["calculate_fibonacci", "parse_config"],
                "classes": ["MathUtils", "DataProcessor"]
            },
            "last_analysis": "2024-01-15T10:30:00Z"
        })

        result = await mock_tools.write_memory("semantic_analysis_cache", memory_content)

        assert result["status"] == "success"
        assert result["memory_name"] == "semantic_analysis_cache"
        assert result["content_size"] > 0

        history = mock_tools.get_call_history()
        stored_content = json.loads(history[0]['args'][1])
        assert stored_content["project_id"] == "test_project"
        assert len(stored_content["analyzed_files"]) == 2

    @pytest.mark.asyncio
    async def test_read_memory(self, mock_serena_mcp_client):
        """Test reading data from memory."""
        mock_tools = MockSerenaTools()

        stored_data = {
            "project_context": {
                "current_task": "implement_fibonacci_optimization",
                "semantic_hints": ["recursive_to_iterative", "memoization_pattern"],
                "related_symbols": ["calculate_fibonacci", "MathUtils.factorial"]
            }
        }

        mock_tools.set_response('read_memory', {
            "status": "success",
            "memory_name": "project_context",
            "content": stored_data
        })

        result = await mock_tools.read_memory("project_context")

        assert result["status"] == "success"
        assert result["memory_name"] == "project_context"

        content = result["content"]
        assert content["project_context"]["current_task"] == "implement_fibonacci_optimization"
        assert "memoization_pattern" in content["project_context"]["semantic_hints"]

    @pytest.mark.asyncio
    async def test_list_memories(self, mock_serena_mcp_client):
        """Test listing available memories."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('list_memories', {
            "memories": [
                "semantic_analysis_cache",
                "project_context",
                "coordination_state",
                "performance_metrics"
            ],
            "total_count": 4
        })

        result = await mock_tools.list_memories()

        assert result["total_count"] == 4
        assert "semantic_analysis_cache" in result["memories"]
        assert "project_context" in result["memories"]

    @pytest.mark.asyncio
    async def test_delete_memory(self, mock_serena_mcp_client):
        """Test deleting memory."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('delete_memory', {
            "status": "success",
            "memory_name": "outdated_cache",
            "freed_space": 2048
        })

        result = await mock_tools.delete_memory("outdated_cache")

        assert result["status"] == "success"
        assert result["memory_name"] == "outdated_cache"
        assert result["freed_space"] > 0

class TestSerenaMetaCognition:
    """Test meta-cognitive and reflection operations."""

    @pytest.mark.asyncio
    async def test_think_about_collected_information(self, mock_serena_mcp_client):
        """Test reflection on collected information."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('think_about_collected_information', {
            "status": "analyzed",
            "information_quality": "comprehensive",
            "gaps_identified": [
                "missing_test_coverage_analysis",
                "performance_benchmarks_needed"
            ],
            "recommendations": [
                "analyze_test_files_for_coverage",
                "run_performance_profiling"
            ],
            "confidence_score": 0.82
        })

        result = await mock_tools.think_about_collected_information()

        assert result["status"] == "analyzed"
        assert result["information_quality"] == "comprehensive"
        assert len(result["gaps_identified"]) == 2
        assert len(result["recommendations"]) == 2
        assert 0.0 <= result["confidence_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_think_about_task_adherence(self, mock_serena_mcp_client):
        """Test reflection on task adherence."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('think_about_task_adherence', {
            "status": "on_track",
            "adherence_score": 0.91,
            "deviations": [],
            "progress_assessment": {
                "completed_objectives": ["semantic_analysis", "symbol_mapping"],
                "current_objective": "code_optimization",
                "remaining_objectives": ["testing", "documentation"]
            },
            "recommendations": ["continue_current_approach"]
        })

        result = await mock_tools.think_about_task_adherence()

        assert result["status"] == "on_track"
        assert result["adherence_score"] > 0.9
        assert len(result["deviations"]) == 0
        assert "semantic_analysis" in result["progress_assessment"]["completed_objectives"]

    @pytest.mark.asyncio
    async def test_think_about_whether_you_are_done(self, mock_serena_mcp_client):
        """Test completion assessment."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('think_about_whether_you_are_done', {
            "status": "not_complete",
            "completion_percentage": 75,
            "remaining_tasks": [
                "optimize_recursive_functions",
                "add_error_handling",
                "write_comprehensive_tests"
            ],
            "completion_criteria": {
                "all_symbols_analyzed": True,
                "code_optimized": False,
                "tests_written": False,
                "documentation_updated": False
            },
            "estimated_remaining_time": "30_minutes"
        })

        result = await mock_tools.think_about_whether_you_are_done()

        assert result["status"] == "not_complete"
        assert result["completion_percentage"] == 75
        assert len(result["remaining_tasks"]) == 3

        criteria = result["completion_criteria"]
        assert criteria["all_symbols_analyzed"] is True
        assert criteria["code_optimized"] is False

class TestSerenaOnboarding:
    """Test onboarding and initialization operations."""

    @pytest.mark.asyncio
    async def test_check_onboarding_performed(self, mock_serena_mcp_client):
        """Test checking onboarding status."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('check_onboarding_performed', {
            "onboarding_performed": True,
            "available_memories": [
                "project_overview",
                "development_commands",
                "mcp_tools_overview",
                "code_style_conventions"
            ],
            "last_onboarding": "2024-01-15T09:00:00Z"
        })

        result = await mock_tools.check_onboarding_performed()

        assert result["onboarding_performed"] is True
        assert len(result["available_memories"]) == 4
        assert "project_overview" in result["available_memories"]

    @pytest.mark.asyncio
    async def test_onboarding_process(self, mock_serena_mcp_client):
        """Test onboarding process execution."""
        mock_tools = MockSerenaTools()
        mock_tools.set_response('onboarding', {
            "status": "initiated",
            "onboarding_steps": [
                "analyze_project_structure",
                "identify_main_components",
                "establish_coding_patterns",
                "create_development_memory"
            ],
            "estimated_duration": "5_minutes",
            "next_action": "begin_project_analysis"
        })

        result = await mock_tools.onboarding()

        assert result["status"] == "initiated"
        assert len(result["onboarding_steps"]) == 4
        assert "analyze_project_structure" in result["onboarding_steps"]

class TestSerenaErrorHandling:
    """Test error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_serena_mcp_client):
        """Test handling of timeout conditions."""
        mock_tools = MockSerenaTools()
        mock_tools.set_delay('get_symbols_overview', 5000)  # 5 second delay
        mock_tools.set_error_condition('get_symbols_overview', asyncio.TimeoutError("Operation timed out"))

        with pytest.raises(asyncio.TimeoutError):
            await mock_tools.get_symbols_overview("large_file.py")

        metrics = mock_tools.get_metrics()
        assert metrics['error_count'] == 1

    @pytest.mark.asyncio
    async def test_invalid_parameters(self, mock_serena_mcp_client):
        """Test handling of invalid parameters."""
        mock_tools = MockSerenaTools()
        mock_tools.set_error_condition('find_symbol', ValueError("Invalid name_path pattern"))

        with pytest.raises(ValueError, match="Invalid name_path pattern"):
            await mock_tools.find_symbol("")  # Empty name path

    @pytest.mark.asyncio
    async def test_file_not_found(self, mock_serena_mcp_client):
        """Test handling of missing files."""
        mock_tools = MockSerenaTools()
        mock_tools.set_error_condition('get_symbols_overview', FileNotFoundError("File not found"))

        with pytest.raises(FileNotFoundError):
            await mock_tools.get_symbols_overview("nonexistent.py")

    @pytest.mark.asyncio
    async def test_recovery_mechanisms(self, mock_serena_mcp_client):
        """Test error recovery and graceful degradation."""
        mock_tools = MockSerenaTools()

        # First call fails, second succeeds
        call_count = 0

        async def failing_then_succeeding_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return {"status": "success", "data": "recovered"}

        # Patch the simulate call method
        mock_tools._simulate_call = failing_then_succeeding_call

        # First call should fail
        with pytest.raises(ConnectionError):
            await mock_tools.list_dir(".")

        # Second call should succeed
        result = await mock_tools.list_dir(".")
        assert result["status"] == "success"
        assert result["data"] == "recovered"

class TestSerenaPerformanceCharacteristics:
    """Test performance characteristics of Serena tools."""

    @pytest.mark.asyncio
    @time_test
    async def test_tool_response_times(self, mock_serena_mcp_client):
        """Test that tool responses are within acceptable time limits."""
        mock_tools = MockSerenaTools()
        mock_tools.set_delay('get_symbols_overview', 100)  # 100ms delay

        start_time = time.time()
        result = await mock_tools.get_symbols_overview("src/main.py")
        end_time = time.time()

        execution_time_ms = (end_time - start_time) * 1000

        # Should complete within reasonable time (allowing for test overhead)
        assert execution_time_ms >= 100  # At least the configured delay
        assert execution_time_ms < 200   # But not too much overhead

        metrics = mock_tools.get_metrics()
        assert metrics['total_calls'] == 1
        assert metrics['success_rate'] == 1.0

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, mock_serena_mcp_client):
        """Test concurrent execution of multiple tools."""
        mock_tools = MockSerenaTools()

        # Configure different responses for different tools
        mock_tools.set_response('list_dir', {"dirs": ["src"]})
        mock_tools.set_response('find_file', {"found_files": ["test.py"]})
        mock_tools.set_response('get_symbols_overview', {"symbols": []})

        # Execute tools concurrently
        start_time = time.time()
        results = await asyncio.gather(
            mock_tools.list_dir("."),
            mock_tools.find_file("*.py", "."),
            mock_tools.get_symbols_overview("src/main.py")
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # All three calls should complete
        assert len(results) == 3
        assert results[0]["dirs"] == ["src"]
        assert results[1]["found_files"] == ["test.py"]
        assert results[2]["symbols"] == []

        # Concurrent execution should be faster than sequential
        assert execution_time < 0.1  # Should complete quickly with no delays

        # All calls should be recorded
        metrics = mock_tools.get_metrics()
        assert metrics['total_calls'] == 3

    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self, mock_serena_mcp_client):
        """Test memory usage patterns during tool operations."""

        mock_tools = MockSerenaTools()

        # Configure response with large data
        large_response = {
            "symbols": [{"name": f"symbol_{i}", "data": "x" * 1000} for i in range(100)]
        }
        mock_tools.set_response('get_symbols_overview', large_response)

        # Measure memory before and after
        initial_memory = psutil.Process().memory_info().rss

        result = await mock_tools.get_symbols_overview("large_file.py")

        final_memory = psutil.Process().memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Should handle large responses reasonably
        assert len(result["symbols"]) == 100
        # Memory increase should be reasonable (less than 10MB for this test)
        assert memory_increase < 10
