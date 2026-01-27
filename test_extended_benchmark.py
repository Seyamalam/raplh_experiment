"""
Test suite for Extended Heart Disease Prediction Benchmark
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_heart_disease_extended import (
    ExtendedHeartDiseaseBenchmark,
    NestedCVResult,
    StackedEnsembleResult,
    DataScarcityResult,
    PipelineResult
)


class TestExtendedBenchmarkData:
    """Test data loading in extended benchmark."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_load_data(self, benchmark):
        """Test that data loads correctly."""
        X, y = benchmark.load_data()
        assert X.shape[0] == 270
        assert len(y) == 270
        assert X.shape[1] == 13
    
    def test_target_encoding(self, benchmark):
        """Test that target is properly encoded."""
        X, y = benchmark.load_data()
        assert set(y) == {0, 1}


class TestNestedCVResult:
    """Test nested CV result dataclass."""
    
    def test_nested_cv_result_creation(self):
        """Test creating a nested CV result."""
        result = NestedCVResult(
            model_name="XGBoost",
            outer_score=0.892,
            outer_score_ci=(0.852, 0.932),
            inner_best_params={'n_estimators': 100},
            n_features_selected=7,
            nested_fold_scores=[0.88, 0.89, 0.90, 0.91, 0.89]
        )
        assert result.model_name == "XGBoost"
        assert result.outer_score == 0.892
        assert result.n_features_selected == 7
        assert len(result.nested_fold_scores) == 5


class TestStackedEnsembleResult:
    """Test stacked ensemble result dataclass."""
    
    def test_stacked_ensemble_result_creation(self):
        """Test creating a stacked ensemble result."""
        result = StackedEnsembleResult(
            model_name="Stacking (MLP)",
            base_estimators=["RF", "GB", "XGB"],
            meta_learner="MLP",
            accuracy=0.89,
            roc_auc=0.904,
            f1=0.87,
            latency_ms=2.34,
            memory_mb=15.2,
            hyperparameters={'final_estimator': 'MLP'}
        )
        assert result.model_name == "Stacking (MLP)"
        assert result.roc_auc == 0.904
        assert len(result.base_estimators) == 3


class TestDataScarcityResult:
    """Test data scarcity result dataclass."""
    
    def test_data_scarcity_result_creation(self):
        """Test creating a data scarcity result."""
        result = DataScarcityResult(
            model_name="XGBoost",
            train_percentage=0.25,
            sample_size=68,
            accuracy=0.84,
            roc_auc=0.86,
            f1=0.82,
            std_accuracy=0.05,
            std_roc_auc=0.04,
            fold_scores=[[0.82, 0.85, 0.87], [0.84, 0.86, 0.88], [0.80, 0.83, 0.85]]
        )
        assert result.train_percentage == 0.25
        assert result.sample_size == 68
        assert result.std_accuracy == 0.05


class TestPipelineResult:
    """Test pipeline result dataclass."""
    
    def test_pipeline_result_creation(self):
        """Test creating a pipeline result."""
        result = PipelineResult(
            pipeline_name="RF + SelectKBest",
            feature_selection_method="SelectKBest",
            n_features=7,
            accuracy=0.87,
            roc_auc=0.91,
            f1=0.85,
            best_params={'classifier__n_estimators': 100},
            latency_ms=0.52,
            memory_mb=8.5
        )
        assert result.pipeline_name == "RF + SelectKBest"
        assert result.n_features == 7
        assert result.roc_auc == 0.91


class TestTop5Models:
    """Test top 5 models retrieval."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_get_top_5_models(self, benchmark):
        """Test that top 5 models are retrieved."""
        models = benchmark.get_top_5_models()
        assert len(models) >= 5
        assert 'Random Forest' in models
        assert 'Gradient Boosting' in models


class TestAdvancedStackedEnsembles:
    """Test advanced stacked ensemble creation."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_create_stacked_ensembles(self, benchmark):
        """Test that stacked ensembles are created."""
        ensembles = benchmark.create_advanced_stacked_ensembles()
        assert len(ensembles) >= 3
        assert any('Stacking' in name for name in ensembles.keys())
        assert any('MLP' in name or 'Ridge' in name for name in ensembles.keys())
    
    def test_stacked_ensemble_has_meta_learner(self, benchmark):
        """Test that stacked ensembles have meta-learners."""
        ensembles = benchmark.create_advanced_stacked_ensembles()
        for name, ensemble in ensembles.items():
            assert hasattr(ensemble, 'final_estimator')
            assert ensemble.final_estimator is not None


class TestSklearnPipelines:
    """Test scikit-learn pipeline creation."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_create_pipelines(self, benchmark):
        """Test that pipelines are created."""
        pipelines = benchmark.create_sklearn_pipelines()
        assert len(pipelines) >= 3
        assert any('SelectKBest' in name for name in pipelines.keys())
    
    def test_pipeline_has_steps(self, benchmark):
        """Test that pipelines have required steps."""
        pipelines = benchmark.create_sklearn_pipelines()
        for name, pipeline in pipelines.items():
            assert hasattr(pipeline, 'named_steps')
            assert 'classifier' in pipeline.named_steps


class TestNestedCVBenchmark:
    """Test nested cross-validation benchmark."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_run_nested_cv_single_model(self, benchmark):
        """Test running nested CV for a single model."""
        X, y = benchmark.load_data()
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        param_grid = {'n_estimators': [50, 100]}
        
        result = benchmark.run_nested_cross_validation(
            "Test_RF", model, X, y, param_grid
        )
        
        assert result.model_name == "Test_RF"
        assert result.outer_score > 0.7
        assert result.outer_score < 1.0
        assert len(result.nested_fold_scores) == 5


class TestDataScarcityStudy:
    """Test data scarcity study."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_run_scarcity_study_single_model(self, benchmark):
        """Test running scarcity study for a single model."""
        X, y = benchmark.load_data()
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        benchmark.scarcity_results = []
        benchmark.run_data_scarcity_study(X, y)
        
        assert len(benchmark.scarcity_results) > 0
        for result in benchmark.scarcity_results:
            assert result.train_percentage in [0.10, 0.25, 0.50]
            assert result.accuracy > 0.5
            assert result.roc_auc > 0.5


class TestPipelineBenchmark:
    """Test pipeline benchmarking."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_benchmark_pipelines(self, benchmark):
        """Test running pipeline benchmark."""
        X, y = benchmark.load_data()
        
        benchmark.pipeline_results = []
        results = benchmark.benchmark_pipelines(X, y)
        
        assert len(results) > 0
        for result in results:
            assert result.n_features > 0
            assert result.n_features <= 13
            assert result.accuracy > 0.5
            assert result.roc_auc > 0.5


class TestVisualizationGeneration:
    """Test visualization generation."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_generate_nested_cv_visualization(self, benchmark):
        """Test nested CV visualization generation."""
        Path('visualizations').mkdir(exist_ok=True, parents=True)
        
        benchmark.nested_results = [
            NestedCVResult(
                model_name="XGBoost",
                outer_score=0.892,
                outer_score_ci=(0.852, 0.932),
                inner_best_params={},
                n_features_selected=7,
                nested_fold_scores=[0.88, 0.89, 0.90, 0.91, 0.89]
            ),
            NestedCVResult(
                model_name="Random Forest",
                outer_score=0.871,
                outer_score_ci=(0.831, 0.911),
                inner_best_params={},
                n_features_selected=7,
                nested_fold_scores=[0.85, 0.87, 0.88, 0.89, 0.86]
            )
        ]
        
        benchmark.generate_nested_cv_visualization()
        
        assert Path('visualizations/nested_cv_results.png').exists()
    
    def test_generate_pipeline_comparison(self, benchmark):
        """Test pipeline comparison visualization generation."""
        Path('visualizations').mkdir(exist_ok=True, parents=True)
        
        benchmark.pipeline_results = [
            PipelineResult(
                pipeline_name="RF + SelectKBest",
                feature_selection_method="SelectKBest",
                n_features=7,
                accuracy=0.87,
                roc_auc=0.91,
                f1=0.85,
                best_params={},
                latency_ms=0.52,
                memory_mb=8.5
            ),
            PipelineResult(
                pipeline_name="XGB + SelectKBest",
                feature_selection_method="SelectKBest",
                n_features=7,
                accuracy=0.86,
                roc_auc=0.90,
                f1=0.84,
                best_params={},
                latency_ms=0.48,
                memory_mb=9.2
            )
        ]
        
        benchmark.generate_pipeline_comparison()
        
        assert Path('visualizations/pipeline_comparison.png').exists()


class TestResultsSaving:
    """Test results saving functionality."""
    
    @pytest.fixture
    def benchmark(self):
        return ExtendedHeartDiseaseBenchmark()
    
    def test_save_extended_results(self, benchmark):
        """Test saving extended results."""
        benchmark.nested_results = [
            NestedCVResult("XGBoost", 0.892, (0.85, 0.93), {}, 7, [0.88, 0.89, 0.90, 0.91, 0.89])
        ]
        benchmark.stacked_results = [
            StackedEnsembleResult("Stacking (MLP)", ["RF", "GB"], "MLP", 0.89, 0.904, 0.87, 2.34, 15.2, {})
        ]
        benchmark.scarcity_results = [
            DataScarcityResult("XGBoost", 0.25, 68, 0.84, 0.86, 0.82, 0.05, 0.04, [])
        ]
        benchmark.pipeline_results = [
            PipelineResult("RF + SelectKBest", "SelectKBest", 7, 0.87, 0.91, 0.85, {}, 0.52, 8.5)
        ]
        
        benchmark.save_extended_results()
        
        assert Path('extended_results/nested_cv_results.csv').exists()
        assert Path('extended_results/stacked_ensemble_results.csv').exists()
        assert Path('extended_results/data_scarcity_results.csv').exists()
        assert Path('extended_results/pipeline_results.csv').exists()


class TestResearchPaperUpdate:
    """Test research paper update."""
    
    def test_update_research_paper(self):
        """Test updating research paper with new section."""
        original_content = ""
        with open('Research_Paper.md', 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        update_research_paper()
        
        with open('Research_Paper.md', 'r', encoding='utf-8') as f:
            updated_content = f.read()
        
        assert '## 6. Model Robustness and Pipelining Efficiency' in updated_content
        assert 'Nested Cross-Validation' in updated_content
        assert 'Data Scarcity Study' in updated_content
        assert 'End-to-End Scikit-Learn Pipelines' in updated_content
        
        assert len(updated_content) > len(original_content)


class TestFullExtendedBenchmark:
    """Test full extended benchmark execution."""
    
    def test_run_full_extended_benchmark(self):
        """Test running the complete extended benchmark."""
        benchmark = ExtendedHeartDiseaseBenchmark()
        
        benchmark.run_full_extended_benchmark()
        
        assert len(benchmark.nested_results) > 0
        assert len(benchmark.scarcity_results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
