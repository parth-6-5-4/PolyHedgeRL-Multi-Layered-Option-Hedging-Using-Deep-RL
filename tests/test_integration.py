# tests/test_integration.py
"""
Integration test for the end-to-end PolyHedgeRL pipeline.
"""
import pytest
import os
from src.config.settings import get_config, create_directories
from scripts.train_agent import main as train_main
from scripts.evaluate_performance import main as eval_main

@pytest.mark.integration
def test_full_pipeline(tmp_path, monkeypatch):
    # Redirect model and report paths to temp directory
    paths = get_config('paths')
    monkeypatch.setitem(paths, 'models_dir', str(tmp_path/'models'))
    monkeypatch.setitem(paths, 'reports_dir', str(tmp_path/'reports'))
    create_directories()

    # Train for very few timesteps
    import sys
    sys.argv = ['train_agent.py', '--timesteps', '100']
    train_main()

    # Check model saved
    model_dir = paths['models_dir']
    assert os.listdir(model_dir), 'No models saved'

    # Evaluate
    sys.argv = ['evaluate_performance.py', '--model_path', os.path.join(model_dir, get_config('model')['best_model_name']), '--episodes', '5']
    eval_main()

    # Check report generated
    report_dir = paths['reports_dir']
    assert os.listdir(report_dir), 'No reports generated'

if __name__ == '__main__':
    pytest.main([__file__])