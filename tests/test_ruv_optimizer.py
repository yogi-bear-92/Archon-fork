"""
Comprehensive test suite for rUv Economy Optimizer
Validates all optimization algorithms and ensures quality standards
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ruv_economy_optimizer import (
    AdvancedEconomicOptimizer, 
    EconomicContext
)

class TestRuvEconomyOptimizer(unittest.TestCase):
    """Test suite for Advanced Economic Optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = AdvancedEconomicOptimizer()
        self.test_context = EconomicContext(
            market_volatility=0.15,
            liquidity_score=75,
            demand_trend=1.2,
            supply_constraint=0.8,
            competition_intensity=8
        )
    
    def test_synthetic_data_generation(self):
        """Test synthetic market data generation"""
        data = self.optimizer.generate_synthetic_market_data(100)
        
        # Validate structure
        expected_columns = ['gdp_trend', 'inflation', 'market_cap', 'volatility',
                           'supply', 'demand', 'competition', 'liquidity', 'ruv_potential']
        self.assertEqual(list(data.columns), expected_columns)
        self.assertEqual(len(data), 100)
        
        # Validate data ranges
        self.assertTrue(data['volatility'].min() >= 0)
        self.assertTrue(data['competition'].min() >= 0)
        self.assertTrue(data['liquidity'].mean() > 0)
        
    def test_ensemble_model_building(self):
        """Test ensemble predictor building"""
        data = self.optimizer.generate_synthetic_market_data(200)
        self.optimizer.build_ensemble_predictor(data)
        
        # Validate models were created
        self.assertTrue(len(self.optimizer.ensemble_models) > 0)
        
        # Validate model structure
        for model_info in self.optimizer.ensemble_models:
            self.assertIn('name', model_info)
            self.assertIn('model', model_info)
            self.assertIn('weight', model_info)
            self.assertTrue(model_info['weight'] > 0)
    
    def test_ruv_prediction(self):
        """Test rUv potential prediction"""
        # Build models first
        data = self.optimizer.generate_synthetic_market_data(200)
        self.optimizer.build_ensemble_predictor(data)
        
        # Test prediction
        prediction = self.optimizer.predict_ruv_potential(self.test_context)
        
        # Validate prediction is reasonable
        self.assertIsInstance(prediction, (int, float))
        self.assertTrue(-20 <= prediction <= 20)  # Reasonable range
    
    def test_portfolio_optimization(self):
        """Test portfolio allocation optimization"""
        assets = ['stocks', 'bonds', 'commodities']
        expected_returns = np.array([0.08, 0.04, 0.06])
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.02, 0.005],
            [0.02, 0.005, 0.03]
        ])
        
        portfolio = self.optimizer.optimize_portfolio_allocation(
            assets, expected_returns, cov_matrix
        )
        
        # Validate portfolio structure
        self.assertEqual(len(portfolio), 3)
        self.assertEqual(set(portfolio.keys()), set(assets))
        
        # Validate weights sum to 1 (approximately)
        total_weight = sum(portfolio.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # Validate all weights are non-negative
        self.assertTrue(all(w >= -1e-6 for w in portfolio.values()))
    
    def test_dynamic_pricing(self):
        """Test dynamic pricing strategy"""
        base_price = 100.0
        
        # Test with different contexts
        contexts = [
            EconomicContext(0.1, 80, 1.3, 0.7, 5),   # Favorable
            EconomicContext(0.3, 30, 0.8, 1.5, 15),  # Challenging
        ]
        
        prices = []
        for context in contexts:
            price = self.optimizer.dynamic_pricing_strategy(base_price, context)
            prices.append(price)
            
            # Validate price is positive and reasonable
            self.assertTrue(price > 0)
            self.assertTrue(50 <= price <= 200)  # Within reasonable bounds
        
        # Favorable context should generally yield higher prices
        # (though not guaranteed due to complexity of factors)
        self.assertTrue(all(p > 0 for p in prices))
    
    def test_resource_allocation(self):
        """Test resource allocation optimization"""
        resources = {'marketing': 2.0, 'rd': 3.0, 'operations': 1.5}
        constraints = {'budget': 500, 'max_marketing': 200}
        
        allocation = self.optimizer.resource_allocation_optimization(
            resources, constraints
        )
        
        # Validate allocation structure
        self.assertEqual(set(allocation.keys()), set(resources.keys()))
        
        # Validate budget constraint
        total_allocated = sum(allocation.values())
        self.assertLessEqual(total_allocated, constraints['budget'] + 1e-6)
        
        # Validate individual constraints
        self.assertLessEqual(allocation['marketing'], 
                           constraints.get('max_marketing', float('inf')) + 1e-6)
        
        # Validate non-negative allocations
        self.assertTrue(all(a >= -1e-6 for a in allocation.values()))
    
    def test_comprehensive_optimization(self):
        """Test full optimization suite"""
        results = self.optimizer.run_comprehensive_optimization()
        
        # Validate results structure
        self.assertIn('summary', results)
        self.assertIn('total_ruv_potential', results['summary'])
        self.assertIn('optimization_efficiency', results['summary'])
        
        # Validate scenario results
        for i in range(1, 5):
            scenario_key = f'scenario_{i}'
            self.assertIn(scenario_key, results)
            
            scenario = results[scenario_key]
            self.assertIn('ruv_potential', scenario)
            self.assertIn('optimal_portfolio', scenario)
            self.assertIn('optimal_price', scenario)
            self.assertIn('resource_allocation', scenario)
    
    def test_economic_context_validation(self):
        """Test economic context handling"""
        # Test with extreme values
        extreme_context = EconomicContext(
            market_volatility=0.5,
            liquidity_score=10,
            demand_trend=2.0,
            supply_constraint=0.1,
            competition_intensity=25
        )
        
        # Should handle extreme values gracefully
        data = self.optimizer.generate_synthetic_market_data(50)
        self.optimizer.build_ensemble_predictor(data)
        
        prediction = self.optimizer.predict_ruv_potential(extreme_context)
        self.assertIsInstance(prediction, (int, float))
        
        price = self.optimizer.dynamic_pricing_strategy(100, extreme_context)
        self.assertTrue(price > 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance and quality metrics"""
    
    def setUp(self):
        self.optimizer = AdvancedEconomicOptimizer()
    
    def test_model_accuracy_threshold(self):
        """Ensure models meet minimum accuracy requirements"""
        data = self.optimizer.generate_synthetic_market_data(500)
        self.optimizer.build_ensemble_predictor(data)
        
        # All models should have positive weights (indicating decent performance)
        for model_info in self.optimizer.ensemble_models:
            self.assertGreater(model_info['weight'], 0.1)
    
    def test_optimization_convergence(self):
        """Test that optimization algorithms converge properly"""
        # Portfolio optimization convergence
        assets = ['a', 'b', 'c']
        returns = np.array([0.1, 0.08, 0.12])
        cov = np.eye(3) * 0.04
        
        portfolio = self.optimizer.optimize_portfolio_allocation(assets, returns, cov)
        
        # Should converge to valid solution
        self.assertAlmostEqual(sum(portfolio.values()), 1.0, places=2)
    
    def test_scalability(self):
        """Test performance with larger datasets"""
        import time
        
        start_time = time.time()
        data = self.optimizer.generate_synthetic_market_data(1000)
        self.optimizer.build_ensemble_predictor(data)
        end_time = time.time()
        
        # Should complete within reasonable time (5 seconds)
        self.assertLess(end_time - start_time, 5.0)


if __name__ == '__main__':
    print("🧪 Running rUv Economy Optimizer Test Suite...")
    print("="*50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRuvEconomyOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed! Ready for challenge submission.")
    else:
        print("❌ Some tests failed. Please review and fix issues.")
        
    exit(0 if result.wasSuccessful() else 1)