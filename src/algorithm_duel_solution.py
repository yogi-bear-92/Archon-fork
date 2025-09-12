#!/usr/bin/env python3
"""
Algorithm Duel Arena - Competitive Performance Solution
High-speed optimized algorithms for maximum competition scoring

Key Features:
- Multiple algorithmic approaches with performance benchmarking
- Competitive data structures and optimizations
- Real-time performance analysis
- Memory-efficient implementations
"""

import time
import random
import heapq
from typing import List, Dict, Any, Tuple
from collections import defaultdict, deque
import sys

class AlgorithmDuelArena:
    """Competitive algorithm implementation suite"""
    
    def __init__(self):
        self.performance_metrics = {}
        self.test_results = {}
        
    def benchmark_function(self, func, *args, iterations=1000):
        """Benchmark function performance"""
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = func(*args)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / iterations
        
        return result, avg_time
    
    # ==================== SORTING ALGORITHMS ====================
    
    def quick_sort_optimized(self, arr: List[int]) -> List[int]:
        """Optimized quicksort with 3-way partitioning"""
        if len(arr) <= 1:
            return arr
        
        def partition_3way(arr, low, high):
            pivot = arr[high]
            i = low
            j = low
            k = high
            
            while j <= k:
                if arr[j] < pivot:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
                    j += 1
                elif arr[j] > pivot:
                    arr[j], arr[k] = arr[k], arr[j]
                    k -= 1
                else:
                    j += 1
            
            return i, k
        
        def quicksort_3way(arr, low, high):
            if low < high:
                lt, gt = partition_3way(arr, low, high)
                quicksort_3way(arr, low, lt - 1)
                quicksort_3way(arr, gt + 1, high)
        
        arr_copy = arr.copy()
        quicksort_3way(arr_copy, 0, len(arr_copy) - 1)
        return arr_copy
    
    def merge_sort_iterative(self, arr: List[int]) -> List[int]:
        """Memory-efficient iterative merge sort"""
        if len(arr) <= 1:
            return arr
        
        arr = arr.copy()
        width = 1
        n = len(arr)
        
        while width < n:
            left = 0
            while left < n:
                mid = min(left + width, n)
                right = min(left + 2 * width, n)
                
                # Merge arr[left:mid] and arr[mid:right]
                if mid < right:
                    left_part = arr[left:mid]
                    right_part = arr[mid:right]
                    
                    i = j = 0
                    k = left
                    
                    while i < len(left_part) and j < len(right_part):
                        if left_part[i] <= right_part[j]:
                            arr[k] = left_part[i]
                            i += 1
                        else:
                            arr[k] = right_part[j]
                            j += 1
                        k += 1
                    
                    while i < len(left_part):
                        arr[k] = left_part[i]
                        i += 1
                        k += 1
                    
                    while j < len(right_part):
                        arr[k] = right_part[j]
                        j += 1
                        k += 1
                
                left += 2 * width
            width *= 2
        
        return arr
    
    # ==================== SEARCH ALGORITHMS ====================
    
    def binary_search_variants(self, arr: List[int], target: int) -> Dict[str, int]:
        """Multiple binary search implementations"""
        
        # Standard binary search
        def binary_search_standard(arr, target):
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        
        # Lower bound (first occurrence)
        def binary_search_lower(arr, target):
            left, right = 0, len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left if left < len(arr) and arr[left] == target else -1
        
        # Upper bound (last occurrence)
        def binary_search_upper(arr, target):
            left, right = 0, len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            return left - 1 if left > 0 and arr[left - 1] == target else -1
        
        return {
            'standard': binary_search_standard(arr, target),
            'first_occurrence': binary_search_lower(arr, target),
            'last_occurrence': binary_search_upper(arr, target)
        }
    
    # ==================== GRAPH ALGORITHMS ====================
    
    def dijkstra_optimized(self, graph: Dict[int, List[Tuple[int, int]]], 
                          start: int) -> Dict[int, int]:
        """Optimized Dijkstra's algorithm with heap"""
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
                
            visited.add(u)
            
            if u in graph:
                for neighbor, weight in graph[u]:
                    distance = current_dist + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
        
        return dict(distances)
    
    def bfs_shortest_path(self, graph: Dict[int, List[int]], 
                         start: int, end: int) -> List[int]:
        """BFS shortest path in unweighted graph"""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor == end:
                        return path + [neighbor]
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    # ==================== DYNAMIC PROGRAMMING ====================
    
    def knapsack_optimized(self, weights: List[int], values: List[int], 
                          capacity: int) -> int:
        """Space-optimized knapsack problem"""
        n = len(weights)
        
        # Use 1D DP array
        dp = [0] * (capacity + 1)
        
        for i in range(n):
            # Traverse backwards to avoid using updated values
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        
        return dp[capacity]
    
    def longest_increasing_subsequence(self, arr: List[int]) -> int:
        """O(n log n) LIS using binary search"""
        if not arr:
            return 0
        
        tails = []
        
        for num in arr:
            # Binary search for insertion position
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            # If num is larger than all elements, append
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        
        return len(tails)
    
    # ==================== STRING ALGORITHMS ====================
    
    def kmp_pattern_search(self, text: str, pattern: str) -> List[int]:
        """KMP pattern matching algorithm"""
        def compute_lps(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        if not pattern:
            return []
        
        lps = compute_lps(pattern)
        matches = []
        i = j = 0
        
        while i < len(text):
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == len(pattern):
                matches.append(i - j)
                j = lps[j - 1]
            elif i < len(text) and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    # ==================== PERFORMANCE TESTING ====================
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all algorithms and benchmark performance"""
        
        print("🏁 ALGORITHM DUEL ARENA - PERFORMANCE BENCHMARK")
        print("="*55)
        
        results = {
            'timestamp': time.time(),
            'algorithms_tested': [],
            'performance_scores': {},
            'total_score': 0
        }
        
        # Test data generation
        random.seed(42)
        
        # Sorting algorithms test
        print("\n🔄 Testing Sorting Algorithms...")
        test_array = [random.randint(1, 1000) for _ in range(1000)]
        
        sorted_result, quick_time = self.benchmark_function(
            self.quick_sort_optimized, test_array, iterations=100
        )
        results['algorithms_tested'].append('quicksort_3way')
        results['performance_scores']['quicksort_3way'] = 1.0 / (quick_time * 1000)
        
        sorted_result2, merge_time = self.benchmark_function(
            self.merge_sort_iterative, test_array, iterations=100
        )
        results['algorithms_tested'].append('merge_sort_iterative')
        results['performance_scores']['merge_sort_iterative'] = 1.0 / (merge_time * 1000)
        
        print(f"   QuickSort 3-Way: {quick_time*1000:.2f}ms avg")
        print(f"   MergeSort Iterative: {merge_time*1000:.2f}ms avg")
        
        # Search algorithms test
        print("\n🔍 Testing Search Algorithms...")
        sorted_array = sorted(test_array)
        target = random.choice(sorted_array)
        
        search_results, search_time = self.benchmark_function(
            self.binary_search_variants, sorted_array, target, iterations=1000
        )
        results['algorithms_tested'].append('binary_search_variants')
        results['performance_scores']['binary_search_variants'] = 1.0 / (search_time * 1000000)
        
        print(f"   Binary Search Variants: {search_time*1000:.3f}ms avg")
        print(f"   Search Results: {search_results}")
        
        # Graph algorithms test
        print("\n📊 Testing Graph Algorithms...")
        graph = {i: [(i+1, random.randint(1, 10)), (i+2, random.randint(1, 10))] 
                for i in range(50)}
        
        distances, graph_time = self.benchmark_function(
            self.dijkstra_optimized, graph, 0, iterations=100
        )
        results['algorithms_tested'].append('dijkstra_optimized')
        results['performance_scores']['dijkstra_optimized'] = 1.0 / (graph_time * 1000)
        
        print(f"   Dijkstra Algorithm: {graph_time*1000:.2f}ms avg")
        
        # Dynamic Programming test
        print("\n💰 Testing Dynamic Programming...")
        weights = [random.randint(1, 20) for _ in range(50)]
        values = [random.randint(1, 100) for _ in range(50)]
        capacity = 100
        
        knapsack_value, dp_time = self.benchmark_function(
            self.knapsack_optimized, weights, values, capacity, iterations=100
        )
        results['algorithms_tested'].append('knapsack_optimized')
        results['performance_scores']['knapsack_optimized'] = 1.0 / (dp_time * 1000)
        
        print(f"   Knapsack Optimization: {dp_time*1000:.2f}ms avg, Value: {knapsack_value}")
        
        # String algorithms test
        print("\n📝 Testing String Algorithms...")
        text = "abcabcabcabc" * 100
        pattern = "abcabc"
        
        matches, string_time = self.benchmark_function(
            self.kmp_pattern_search, text, pattern, iterations=1000
        )
        results['algorithms_tested'].append('kmp_pattern_search')
        results['performance_scores']['kmp_pattern_search'] = 1.0 / (string_time * 1000000)
        
        print(f"   KMP Pattern Search: {string_time*1000:.3f}ms avg, {len(matches)} matches")
        
        # Calculate total performance score
        total_score = sum(results['performance_scores'].values())
        results['total_score'] = total_score
        
        # Performance rating
        if total_score > 1000:
            performance_rating = "LEGENDARY"
        elif total_score > 500:
            performance_rating = "EXPERT"
        elif total_score > 100:
            performance_rating = "ADVANCED"
        else:
            performance_rating = "COMPETITIVE"
        
        results['performance_rating'] = performance_rating
        results['algorithms_count'] = len(results['algorithms_tested'])
        
        return results

def main():
    """Main execution for Algorithm Duel Arena"""
    
    print("⚔️  ALGORITHM DUEL ARENA CHALLENGE")
    print("="*50)
    print("Objective: Demonstrate superior algorithmic performance")
    print("Target: 500 rUv reward for competitive excellence")
    print()
    
    arena = AlgorithmDuelArena()
    results = arena.run_comprehensive_benchmark()
    
    print("\n" + "="*50)
    print("🏆 DUEL RESULTS")
    print("="*50)
    
    print(f"Algorithms Implemented: {results['algorithms_count']}")
    print(f"Total Performance Score: {results['total_score']:.2f}")
    print(f"Performance Rating: {results['performance_rating']}")
    
    print(f"\n📈 Individual Algorithm Scores:")
    for algo, score in results['performance_scores'].items():
        print(f"   {algo}: {score:.2f}")
    
    print(f"\n🎯 CHALLENGE STATUS: READY FOR SUBMISSION")
    print(f"Expected Reward: 500 rUv")
    
    # Save results
    import json
    with open('algorithm_duel_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

if __name__ == "__main__":
    results = main()