// Algorithm Duel Arena Challenge Solution
// Challenge ID: 655e031b-97c4-4ac3-9b08-f72d3eba911b
// Reward: 500 rUv + 10 rUv participation
// Requirements: Competitive algorithm performance comparison and duel system

class AlgorithmDuelArena {
  constructor() {
    this.algorithms = new Map();
    this.duelHistory = [];
    this.performanceMetrics = {
      executionTime: [],
      memoryUsage: [],
      accuracy: [],
      efficiency: []
    };
    this.leaderboard = [];
    
    console.log("⚔️ Initializing Algorithm Duel Arena...");
  }

  // Register algorithms for dueling
  registerAlgorithm(name, algorithm, category = "general") {
    const algorithmData = {
      name,
      algorithm,
      category,
      wins: 0,
      losses: 0,
      draws: 0,
      totalDuels: 0,
      averageExecutionTime: 0,
      averageAccuracy: 0,
      rating: 1000 // Starting ELO rating
    };
    
    this.algorithms.set(name, algorithmData);
    console.log(`📝 Registered algorithm: ${name} (${category})`);
    return algorithmData;
  }

  // Execute algorithm with performance measurement
  async executeAlgorithm(algorithmName, testData) {
    const algorithm = this.algorithms.get(algorithmName);
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmName} not found`);
    }

    const startTime = performance.now();
    const startMemory = process.memoryUsage().heapUsed;
    
    try {
      const result = await algorithm.algorithm(testData);
      const endTime = performance.now();
      const endMemory = process.memoryUsage().heapUsed;
      
      const executionTime = endTime - startTime;
      const memoryUsage = endMemory - startMemory;
      
      return {
        result,
        executionTime,
        memoryUsage,
        success: true,
        error: null
      };
    } catch (error) {
      const endTime = performance.now();
      return {
        result: null,
        executionTime: endTime - startTime,
        memoryUsage: 0,
        success: false,
        error: error.message
      };
    }
  }

  // Calculate algorithm performance score
  calculatePerformanceScore(executionData, expectedResult, testCase) {
    const { executionTime, memoryUsage, success, result } = executionData;
    
    if (!success) {
      return {
        score: 0,
        accuracy: 0,
        efficiency: 0,
        penalty: 1000
      };
    }

    // Calculate accuracy based on result correctness
    const accuracy = this.calculateAccuracy(result, expectedResult);
    
    // Calculate efficiency (higher is better)
    const timeScore = Math.max(0, 1000 - executionTime); // Lower time = higher score
    const memoryScore = Math.max(0, 1000 - (memoryUsage / 1024)); // Lower memory = higher score
    const efficiency = (timeScore + memoryScore) / 2;
    
    // Overall performance score
    const score = (accuracy * 0.6) + (efficiency * 0.4);
    
    return {
      score: Math.round(score * 100) / 100,
      accuracy: Math.round(accuracy * 100) / 100,
      efficiency: Math.round(efficiency * 100) / 100,
      penalty: 0
    };
  }

  // Calculate accuracy based on result comparison
  calculateAccuracy(result, expected) {
    if (result === expected) return 100;
    
    // Handle different data types
    if (typeof result === 'number' && typeof expected === 'number') {
      const diff = Math.abs(result - expected);
      const maxDiff = Math.max(Math.abs(result), Math.abs(expected));
      return Math.max(0, 100 - (diff / maxDiff) * 100);
    }
    
    if (Array.isArray(result) && Array.isArray(expected)) {
      if (result.length !== expected.length) return 0;
      let matches = 0;
      for (let i = 0; i < result.length; i++) {
        if (result[i] === expected[i]) matches++;
      }
      return (matches / result.length) * 100;
    }
    
    if (typeof result === 'string' && typeof expected === 'string') {
      const similarity = this.calculateStringSimilarity(result, expected);
      return similarity * 100;
    }
    
    return 0;
  }

  // Calculate string similarity using Levenshtein distance
  calculateStringSimilarity(str1, str2) {
    const matrix = [];
    const len1 = str1.length;
    const len2 = str2.length;
    
    for (let i = 0; i <= len2; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= len1; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= len2; i++) {
      for (let j = 1; j <= len1; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    const maxLen = Math.max(len1, len2);
    return maxLen === 0 ? 1 : (maxLen - matrix[len2][len1]) / maxLen;
  }

  // Execute a duel between two algorithms
  async duel(algorithm1Name, algorithm2Name, testCases) {
    console.log(`⚔️ Starting duel: ${algorithm1Name} vs ${algorithm2Name}`);
    
    const algorithm1 = this.algorithms.get(algorithm1Name);
    const algorithm2 = this.algorithms.get(algorithm2Name);
    
    if (!algorithm1 || !algorithm2) {
      throw new Error("One or both algorithms not found");
    }

    const duelResults = {
      algorithm1: algorithm1Name,
      algorithm2: algorithm2Name,
      testCases: testCases.length,
      results: [],
      algorithm1Score: 0,
      algorithm2Score: 0,
      winner: null,
      timestamp: Date.now()
    };

    // Run algorithms on each test case
    for (let i = 0; i < testCases.length; i++) {
      const testCase = testCases[i];
      console.log(`🧪 Test case ${i + 1}/${testCases.length}: ${testCase.description}`);
      
      // Execute both algorithms
      const [result1, result2] = await Promise.all([
        this.executeAlgorithm(algorithm1Name, testCase.input),
        this.executeAlgorithm(algorithm2Name, testCase.input)
      ]);
      
      // Calculate performance scores
      const score1 = this.calculatePerformanceScore(result1, testCase.expected, testCase);
      const score2 = this.calculatePerformanceScore(result2, testCase.expected, testCase);
      
      // Determine winner of this test case
      let testCaseWinner = null;
      if (score1.score > score2.score) {
        testCaseWinner = algorithm1Name;
        duelResults.algorithm1Score += 1;
      } else if (score2.score > score1.score) {
        testCaseWinner = algorithm2Name;
        duelResults.algorithm2Score += 1;
      }
      
      const testResult = {
        testCase: i + 1,
        description: testCase.description,
        algorithm1: {
          result: result1.result,
          executionTime: result1.executionTime,
          memoryUsage: result1.memoryUsage,
          score: score1.score,
          accuracy: score1.accuracy,
          efficiency: score1.efficiency,
          success: result1.success
        },
        algorithm2: {
          result: result2.result,
          executionTime: result2.executionTime,
          memoryUsage: result2.memoryUsage,
          score: score2.score,
          accuracy: score2.accuracy,
          efficiency: score2.efficiency,
          success: result2.success
        },
        winner: testCaseWinner
      };
      
      duelResults.results.push(testResult);
    }
    
    // Determine overall winner
    if (duelResults.algorithm1Score > duelResults.algorithm2Score) {
      duelResults.winner = algorithm1Name;
      algorithm1.wins++;
      algorithm2.losses++;
    } else if (duelResults.algorithm2Score > duelResults.algorithm1Score) {
      duelResults.winner = algorithm2Name;
      algorithm2.wins++;
      algorithm1.losses++;
    } else {
      duelResults.winner = "draw";
      algorithm1.draws++;
      algorithm2.draws++;
    }
    
    algorithm1.totalDuels++;
    algorithm2.totalDuels++;
    
    // Update ELO ratings
    this.updateELORatings(algorithm1, algorithm2, duelResults.winner);
    
    // Store duel history
    this.duelHistory.push(duelResults);
    
    console.log(`🏆 Duel complete! Winner: ${duelResults.winner}`);
    console.log(`📊 Final Score: ${algorithm1Name} ${duelResults.algorithm1Score} - ${duelResults.algorithm2Score} ${algorithm2Name}`);
    
    return duelResults;
  }

  // Update ELO ratings based on duel result
  updateELORatings(algorithm1, algorithm2, winner) {
    const K = 32; // ELO K-factor
    
    const expected1 = 1 / (1 + Math.pow(10, (algorithm2.rating - algorithm1.rating) / 400));
    const expected2 = 1 / (1 + Math.pow(10, (algorithm1.rating - algorithm2.rating) / 400));
    
    let actual1, actual2;
    if (winner === algorithm1.name) {
      actual1 = 1;
      actual2 = 0;
    } else if (winner === algorithm2.name) {
      actual1 = 0;
      actual2 = 1;
    } else {
      actual1 = 0.5;
      actual2 = 0.5;
    }
    
    algorithm1.rating = Math.round(algorithm1.rating + K * (actual1 - expected1));
    algorithm2.rating = Math.round(algorithm2.rating + K * (actual2 - expected2));
  }

  // Generate comprehensive leaderboard
  generateLeaderboard() {
    const algorithms = Array.from(this.algorithms.values());
    
    // Sort by ELO rating (descending)
    algorithms.sort((a, b) => b.rating - a.rating);
    
    this.leaderboard = algorithms.map((algo, index) => ({
      rank: index + 1,
      name: algo.name,
      category: algo.category,
      rating: algo.rating,
      wins: algo.wins,
      losses: algo.losses,
      draws: algo.draws,
      winRate: algo.totalDuels > 0 ? Math.round((algo.wins / algo.totalDuels) * 100) : 0,
      totalDuels: algo.totalDuels
    }));
    
    return this.leaderboard;
  }

  // Run tournament with multiple algorithms
  async runTournament(algorithmNames, testCases, format = "round-robin") {
    console.log(`🏆 Starting ${format} tournament with ${algorithmNames.length} algorithms`);
    
    const tournamentResults = {
      format,
      participants: algorithmNames,
      testCases: testCases.length,
      matches: [],
      standings: [],
      timestamp: Date.now()
    };
    
    if (format === "round-robin") {
      // Round-robin: every algorithm duels every other algorithm
      for (let i = 0; i < algorithmNames.length; i++) {
        for (let j = i + 1; j < algorithmNames.length; j++) {
          const duelResult = await this.duel(algorithmNames[i], algorithmNames[j], testCases);
          tournamentResults.matches.push(duelResult);
        }
      }
    } else if (format === "single-elimination") {
      // Single elimination tournament
      let remaining = [...algorithmNames];
      let round = 1;
      
      while (remaining.length > 1) {
        console.log(`🥊 Round ${round}: ${remaining.length} algorithms remaining`);
        const nextRound = [];
        
        for (let i = 0; i < remaining.length; i += 2) {
          if (i + 1 < remaining.length) {
            const duelResult = await this.duel(remaining[i], remaining[i + 1], testCases);
            tournamentResults.matches.push(duelResult);
            nextRound.push(duelResult.winner);
          } else {
            nextRound.push(remaining[i]); // Bye
          }
        }
        
        remaining = nextRound;
        round++;
      }
      
      tournamentResults.winner = remaining[0];
    }
    
    // Generate final standings
    tournamentResults.standings = this.generateLeaderboard();
    
    console.log(`🏆 Tournament complete! Winner: ${tournamentResults.winner || 'See standings'}`);
    
    return tournamentResults;
  }

  // Generate performance analytics
  generateAnalytics() {
    const algorithms = Array.from(this.algorithms.values());
    
    const analytics = {
      totalAlgorithms: algorithms.length,
      totalDuels: this.duelHistory.length,
      averageExecutionTime: 0,
      averageAccuracy: 0,
      topPerformer: null,
      mostActive: null,
      categoryBreakdown: {},
      performanceTrends: this.analyzePerformanceTrends()
    };
    
    if (algorithms.length > 0) {
      // Calculate averages
      analytics.averageExecutionTime = algorithms.reduce((sum, algo) => sum + algo.averageExecutionTime, 0) / algorithms.length;
      analytics.averageAccuracy = algorithms.reduce((sum, algo) => sum + algo.averageAccuracy, 0) / algorithms.length;
      
      // Find top performer (highest rating)
      analytics.topPerformer = algorithms.reduce((best, current) => 
        current.rating > best.rating ? current : best
      );
      
      // Find most active (most duels)
      analytics.mostActive = algorithms.reduce((most, current) => 
        current.totalDuels > most.totalDuels ? current : most
      );
      
      // Category breakdown
      algorithms.forEach(algo => {
        if (!analytics.categoryBreakdown[algo.category]) {
          analytics.categoryBreakdown[algo.category] = 0;
        }
        analytics.categoryBreakdown[algo.category]++;
      });
    }
    
    return analytics;
  }

  // Analyze performance trends over time
  analyzePerformanceTrends() {
    if (this.duelHistory.length < 2) return null;
    
    const trends = {
      averageExecutionTime: [],
      averageAccuracy: [],
      duelFrequency: []
    };
    
    // Group duels by time periods (simplified)
    const timePeriods = 10;
    const periodSize = Math.ceil(this.duelHistory.length / timePeriods);
    
    for (let i = 0; i < timePeriods; i++) {
      const start = i * periodSize;
      const end = Math.min((i + 1) * periodSize, this.duelHistory.length);
      const periodDuels = this.duelHistory.slice(start, end);
      
      if (periodDuels.length > 0) {
        const avgTime = periodDuels.reduce((sum, duel) => {
          const totalTime = duel.results.reduce((s, r) => s + r.algorithm1.executionTime + r.algorithm2.executionTime, 0);
          return sum + (totalTime / (duel.results.length * 2));
        }, 0) / periodDuels.length;
        
        const avgAccuracy = periodDuels.reduce((sum, duel) => {
          const totalAccuracy = duel.results.reduce((s, r) => s + r.algorithm1.accuracy + r.algorithm2.accuracy, 0);
          return sum + (totalAccuracy / (duel.results.length * 2));
        }, 0) / periodDuels.length;
        
        trends.averageExecutionTime.push(Math.round(avgTime * 100) / 100);
        trends.averageAccuracy.push(Math.round(avgAccuracy * 100) / 100);
        trends.duelFrequency.push(periodDuels.length);
      }
    }
    
    return trends;
  }
}

// Sample algorithms for testing
const sampleAlgorithms = {
  // Sorting algorithms
  bubbleSort: async (data) => {
    const arr = [...data];
    const n = arr.length;
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - i - 1; j++) {
        if (arr[j] > arr[j + 1]) {
          [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        }
      }
    }
    return arr;
  },
  
  quickSort: async (data) => {
    const arr = [...data];
    const quickSortHelper = (arr, low, high) => {
      if (low < high) {
        const pivotIndex = partition(arr, low, high);
        quickSortHelper(arr, low, pivotIndex - 1);
        quickSortHelper(arr, pivotIndex + 1, high);
      }
    };
    
    const partition = (arr, low, high) => {
      const pivot = arr[high];
      let i = low - 1;
      for (let j = low; j < high; j++) {
        if (arr[j] < pivot) {
          i++;
          [arr[i], arr[j]] = [arr[j], arr[i]];
        }
      }
      [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
      return i + 1;
    };
    
    quickSortHelper(arr, 0, arr.length - 1);
    return arr;
  },
  
  // Search algorithms
  linearSearch: async (data) => {
    const [arr, target] = data;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] === target) return i;
    }
    return -1;
  },
  
  binarySearch: async (data) => {
    const [arr, target] = data;
    let left = 0, right = arr.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (arr[mid] === target) return mid;
      if (arr[mid] < target) left = mid + 1;
      else right = mid - 1;
    }
    return -1;
  },
  
  // Mathematical algorithms
  fibonacci: async (data) => {
    const n = data;
    if (n <= 1) return n;
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
      const temp = a + b;
      a = b;
      b = temp;
    }
    return b;
  },
  
  fibonacciRecursive: async (data) => {
    const n = data;
    if (n <= 1) return n;
    return (await sampleAlgorithms.fibonacciRecursive(n - 1)) + 
           (await sampleAlgorithms.fibonacciRecursive(n - 2));
  }
};

// Execute the Algorithm Duel Arena Challenge
async function executeAlgorithmDuelArena() {
  try {
    console.log("🚀 Starting Algorithm Duel Arena Challenge...");
    
    const arena = new AlgorithmDuelArena();
    
    // Register sample algorithms
    arena.registerAlgorithm("Bubble Sort", sampleAlgorithms.bubbleSort, "sorting");
    arena.registerAlgorithm("Quick Sort", sampleAlgorithms.quickSort, "sorting");
    arena.registerAlgorithm("Linear Search", sampleAlgorithms.linearSearch, "search");
    arena.registerAlgorithm("Binary Search", sampleAlgorithms.binarySearch, "search");
    arena.registerAlgorithm("Fibonacci Iterative", sampleAlgorithms.fibonacci, "mathematical");
    arena.registerAlgorithm("Fibonacci Recursive", sampleAlgorithms.fibonacciRecursive, "mathematical");
    
    // Define test cases
    const sortingTestCases = [
      {
        description: "Small array (10 elements)",
        input: [64, 34, 25, 12, 22, 11, 90, 5, 77, 30],
        expected: [5, 11, 12, 22, 25, 30, 34, 64, 77, 90]
      },
      {
        description: "Medium array (100 elements)",
        input: Array.from({length: 100}, () => Math.floor(Math.random() * 1000)),
        expected: null // Will be calculated
      },
      {
        description: "Large array (1000 elements)",
        input: Array.from({length: 1000}, () => Math.floor(Math.random() * 10000)),
        expected: null // Will be calculated
      }
    ];
    
    // Calculate expected results for random arrays
    sortingTestCases.forEach(testCase => {
      if (testCase.expected === null) {
        testCase.expected = [...testCase.input].sort((a, b) => a - b);
      }
    });
    
    const searchTestCases = [
      {
        description: "Find element in sorted array",
        input: [[1, 3, 5, 7, 9, 11, 13, 15], 7],
        expected: 3
      },
      {
        description: "Find non-existent element",
        input: [[1, 3, 5, 7, 9, 11, 13, 15], 8],
        expected: -1
      }
    ];
    
    const fibonacciTestCases = [
      {
        description: "Fibonacci(10)",
        input: 10,
        expected: 55
      },
      {
        description: "Fibonacci(20)",
        input: 20,
        expected: 6765
      },
      {
        description: "Fibonacci(30)",
        input: 30,
        expected: 832040
      }
    ];
    
    // Run duels
    console.log("⚔️ Running sorting algorithm duels...");
    await arena.duel("Bubble Sort", "Quick Sort", sortingTestCases);
    
    console.log("⚔️ Running search algorithm duels...");
    await arena.duel("Linear Search", "Binary Search", searchTestCases);
    
    console.log("⚔️ Running fibonacci algorithm duels...");
    await arena.duel("Fibonacci Iterative", "Fibonacci Recursive", fibonacciTestCases);
    
    // Run tournament
    console.log("🏆 Running round-robin tournament...");
    const tournament = await arena.runTournament(
      ["Bubble Sort", "Quick Sort", "Linear Search", "Binary Search"],
      [...sortingTestCases.slice(0, 2), ...searchTestCases]
    );
    
    // Generate analytics
    const analytics = arena.generateAnalytics();
    const leaderboard = arena.generateLeaderboard();
    
    const challengeResult = {
      challengeId: "655e031b-97c4-4ac3-9b08-f72d3eba911b",
      status: "completed",
      arena: {
        totalAlgorithms: analytics.totalAlgorithms,
        totalDuels: analytics.totalDuels,
        leaderboard: leaderboard.slice(0, 5), // Top 5
        analytics: analytics
      },
      tournament: tournament,
      performance: {
        averageExecutionTime: analytics.averageExecutionTime,
        averageAccuracy: analytics.averageAccuracy,
        topPerformer: analytics.topPerformer?.name,
        mostActive: analytics.mostActive?.name
      },
      timestamp: new Date().toISOString(),
      message: "Algorithm Duel Arena challenge completed successfully!"
    };
    
    console.log("🏆 Algorithm Duel Arena Challenge Result:");
    console.log(JSON.stringify(challengeResult, null, 2));
    
    return challengeResult;
    
  } catch (error) {
    console.error("❌ Algorithm Duel Arena failed:", error);
    throw error;
  }
}

// Execute the challenge
executeAlgorithmDuelArena()
  .then(result => {
    console.log("✅ Challenge completed successfully!");
    console.log(`🏆 Tournament Winner: ${result.tournament.winner || 'See leaderboard'}`);
    console.log(`📊 Top Performer: ${result.performance.topPerformer}`);
  })
  .catch(error => {
    console.error("💥 Challenge execution failed:", error);
  });

export { AlgorithmDuelArena, executeAlgorithmDuelArena };
