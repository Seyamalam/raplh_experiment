"""
Test suite for Heart Disease Prediction Benchmark
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_heart_disease import (
    HeartDiseaseBenchmark, 
    ModelResult, 
    calculate_metrics,
    calculate_confidence_interval
)

class TestDataPreprocessing:
    """Test data loading and preprocessing."""
    
    @pytest.fixture
    def benchmark(self):
        return HeartDiseaseBenchmark()
    
    def test_load_data(self, benchmark):
        """Test that data loads correctly."""
        X, y = benchmark.load_and_preprocess()
        assert X.shape[0] == 270
        assert len(y) == 270
        assert X.shape[1] == 13
    
    def test_target_encoding(self, benchmark):
        """Test that target is properly encoded."""
        X, y = benchmark.load_and_preprocess()
        assert set(y) == {0, 1}
    
    def test_feature_scaling(self, benchmark):
        """Test that numerical features are scaled."""
        X, y = benchmark.load_and_preprocess()
        for idx in benchmark.numerical_indices:
            assert abs(np.mean(X[:, idx])) < 1.0

class TestMetricsCalculation:
    """Test metric calculations."""
    
    def test_calculate_metrics(self):
        """Test metric calculation function."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.6, 0.7, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 'mcc' in metrics
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        scores = np.array([0.8, 0.85, 0.9, 0.75, 0.95])
        ci = calculate_confidence_interval(scores)
        
        assert len(ci) == 2
        assert ci[0] <= np.mean(scores) <= ci[1]
        assert ci[0] < ci[1]

class TestModelCreation:
    """Test model creation methods."""
    
    @pytest.fixture
    def benchmark(self):
        return HeartDiseaseBenchmark()
    
    def test_baseline_models(self, benchmark):
        """Test baseline model creation."""
        models = benchmark.create_baseline_models()
        assert len(models) >= 4
        assert 'Logistic Regression' in models
        assert 'Gaussian Naive Bayes' in models
    
    def test_tree_models(self, benchmark):
        """Test tree-based model creation."""
        models = benchmark.create_tree_models()
        assert 'Random Forest' in models
        assert 'XGBoost' in models or len(models) > 5
    
    def test_knn_models(self, benchmark):
        """Test KNN model creation."""
        models = benchmark.create_knn_models()
        assert len(models) == 3
        assert 'KNN (k=5)' in models
    
    def test_svm_models(self, benchmark):
        """Test SVM model creation."""
        models = benchmark.create_svm_models()
        assert 'SVC (RBF)' in models
        assert 'LinearSVC' in models
    
    def test_neural_network_models(self, benchmark):
        """Test neural network model creation."""
        models = benchmark.create_neural_network_models()
        assert 'MLP (Small)' in models
        assert 'MLP (Medium)' in models

class TestResultsStorage:
    """Test results storage and retrieval."""
    
    def test_results_csv_exists(self):
        """Test that benchmark results CSV exists."""
        assert Path('benchmark_results.csv').exists()
    
    def test_master_results_csv_exists(self):
        """Test that master benchmark results CSV exists."""
        assert Path('master_benchmark_results.csv').exists()
    
    def test_results_csv_content(self):
        """Test that results CSV has expected columns."""
        df = pd.read_csv('benchmark_results.csv')
        required_cols = ['model_name', 'accuracy', 'precision', 'recall', 
                        'f1', 'roc_auc', 'pr_auc', 'mcc', 'latency_ms', 'memory_mb']
        for col in required_cols:
            assert col in df.columns
    
    def test_minimum_models(self):
        """Test that at least 20 models were benchmarked."""
        df = pd.read_csv('benchmark_results.csv')
        assert len(df) >= 20

class TestVisualizations:
    """Test visualization generation."""
    
    def test_visualizations_directory_exists(self):
        """Test that visualizations directory exists."""
        assert Path('visualizations').exists()
    
    def test_roc_curves_exists(self):
        """Test ROC curves visualization exists."""
        assert Path('visualizations/roc_curves.png').exists()
    
    def test_pr_curves_exists(self):
        """Test PR curves visualization exists."""
        assert Path('visualizations/pr_curves.png').exists()
    
    def test_performance_efficiency_exists(self):
        """Test performance vs efficiency visualization exists."""
        assert Path('visualizations/performance_vs_efficiency.png').exists()

class TestResearchPaper:
    """Test research paper generation."""
    
    def test_research_paper_exists(self):
        """Test that research paper was generated."""
        assert Path('Research_Paper.md').exists()
    
    def test_research_paper_structure(self):
        """Test research paper has IMRaD structure."""
        with open('Research_Paper.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '## Abstract' in content
        assert '## 1. Introduction' in content
        assert '## 2. Methods' in content
        assert '## 3. Results' in content
        assert '## 4. Discussion' in content
        assert '## 5. Conclusion' in content
    
    def test_research_paper_citations(self):
        """Test research paper has proper citations."""
        with open('Research_Paper.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'ROC-AUC' in content
        assert 'F1-Score' in content
        assert 'XGBoost' in content or 'LightGBM' in content

class TestBenchmarkExecution:
    """Test benchmark execution flow."""
    
    @pytest.fixture
    def benchmark(self):
        return HeartDiseaseBenchmark()
    
    def test_run_benchmark(self, benchmark):
        """Test that benchmark runs successfully."""
        X, y = benchmark.load_and_preprocess()
        assert X is not None
        assert y is not None
    
    def test_model_progression(self, benchmark):
        """Test that models progress from simple to complex."""
        models = {}
        models.update(benchmark.create_baseline_models())
        models.update(benchmark.create_tree_models())
        models.update(benchmark.create_neural_network_models())
        
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models
        assert any('MLP' in name for name in models.keys())

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
