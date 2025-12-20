#!/usr/bin/env python3
"""
Stock Filter Optimizer - Main Entry Point

Usage:
    python main.py collect    # Collect S&P 500 data
    python main.py train      # Train ML models
    python main.py optimize   # Optimize filter criteria
    python main.py backtest   # Run backtesting
    python main.py serve      # Start web server
    python main.py all        # Run complete pipeline
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def collect_data(args):
    """Collect S&P 500 historical data"""
    from src.data.collector import DataCollector

    print("=" * 60)
    print("COLLECTING S&P 500 DATA")
    print("=" * 60)

    collector = DataCollector()
    df = collector.collect_all_sp500(period=args.period)

    if not df.empty:
        stats = collector.get_data_stats(df)
        print(f"\nCollection complete!")
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Number of tickers: {stats['num_tickers']}")
        print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")


def train_model(args):
    """Train ML models"""
    from src.ml.train import ModelTrainer

    print("=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)

    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        breakout_threshold=args.threshold,
        lookforward_days=args.days,
        test_size=args.test_size
    )

    # Train
    trainer.train(X_train, y_train)

    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)

    # Get feature importance
    importance = trainer.ensemble.get_feature_importance()

    # Save
    trainer.save_model()
    trainer.save_results(metrics, importance)

    print("\n" + "=" * 60)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance.head(20).to_string(index=False))

    return trainer, X_test, y_test


def optimize_filters(args):
    """Optimize filter criteria"""
    from src.ml.train import ModelTrainer
    from src.ml.optimizer import FilterOptimizer
    import pandas as pd

    print("=" * 60)
    print("OPTIMIZING FILTER CRITERIA")
    print("=" * 60)

    trainer = ModelTrainer()

    # Load or train model
    try:
        # Try to load existing model
        import os
        models_dir = trainer.models_dir
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            trainer.load_model(latest_model)
            print(f"Loaded model: {latest_model}")
        else:
            raise FileNotFoundError()
    except:
        print("No trained model found. Training new model...")
        X_train, X_test, y_train, y_test = trainer.prepare_data()
        trainer.train(X_train, y_train)

    # Load processed data
    data_path = trainer.data_dir / "sp500_processed.parquet"
    if not data_path.exists():
        print("Processed data not found. Run training first.")
        return

    df = pd.read_parquet(data_path)

    # Prepare X and y from processed data
    from src.features.engineer import FeatureEngineer
    engineer = FeatureEngineer()
    X, y, _ = engineer.prepare_training_data(df, target_col='breakout')

    # Get feature importance
    importance = trainer.ensemble.get_feature_importance()

    # Optimize
    optimizer = FilterOptimizer()
    criteria = optimizer.optimize(
        X, y, importance,
        top_n_features=args.top_n,
        min_precision=args.min_precision,
        target_recall=args.target_recall
    )

    # Evaluate
    performance = optimizer.evaluate_filter_performance(X, y)

    print("\n" + "=" * 60)
    print("OPTIMAL FILTER CRITERIA")
    print("=" * 60)
    for rule in optimizer.get_filter_rules():
        print(f"  {rule}")

    print("\n" + "=" * 60)
    print("FILTER PERFORMANCE")
    print("=" * 60)
    print(f"  Base breakout rate: {performance['base_breakout_rate']:.2%}")
    print(f"  Filtered breakout rate: {performance['filtered_breakout_rate']:.2%}")
    print(f"  Precision lift: {performance['precision_lift']:.1%}")
    print(f"  Recall: {performance['recall']:.2%}")
    print(f"  Filter pass rate: {performance['filter_pass_rate']:.2%}")

    # Save
    results_path = trainer.results_dir / "optimal_filters.json"
    optimizer.save(str(results_path))
    print(f"\nSaved optimal filters to {results_path}")

    # Get screener config
    screener_config = optimizer.get_screener_config()

    print("\n" + "=" * 60)
    print("SCREENER CONFIGURATION")
    print("=" * 60)
    print("\nTechnical Criteria:")
    for c in screener_config['technical_criteria'][:10]:
        print(f"  {c['rule']}")

    return optimizer, X, y


def run_backtest(args):
    """Run backtesting"""
    from src.ml.optimizer import FilterOptimizer
    from src.backtest.engine import BacktestEngine
    import pandas as pd
    from pathlib import Path

    print("=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    base_path = Path(__file__).parent

    # Load optimizer
    filter_path = base_path / "results" / "optimal_filters.json"
    if not filter_path.exists():
        print("No optimized filters found. Run optimize first.")
        return

    optimizer = FilterOptimizer.load(str(filter_path))

    # Load data
    data_path = base_path / "data" / "sp500_processed.parquet"
    if not data_path.exists():
        print("Processed data not found. Run training first.")
        return

    df = pd.read_parquet(data_path)

    # Prepare features
    from src.features.engineer import FeatureEngineer
    engineer = FeatureEngineer()
    X, y, _ = engineer.prepare_training_data(df, target_col='breakout')

    # Apply filters
    filter_mask = optimizer.apply_filters(X)

    # Align with df
    aligned_mask = pd.Series(False, index=df.index)
    common_idx = df.index.intersection(filter_mask.index)
    aligned_mask.loc[common_idx] = filter_mask.loc[common_idx]

    # Run backtest
    engine = BacktestEngine(
        breakout_threshold=args.threshold,
        lookforward_days=args.days
    )
    results = engine.run_backtest(df, aligned_mask, period_months=args.period_months)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Number of periods: {results['num_periods']}")
    print(f"  Date range: {results['date_range']['start']} to {results['date_range']['end']}")
    print(f"\n  Average Precision: {results['precision']['mean']:.2%} (+/- {results['precision']['std']:.2%})")
    print(f"  Average Recall: {results['recall']['mean']:.2%} (+/- {results['recall']['std']:.2%})")
    print(f"  Average Win Rate: {results['win_rate']['mean']:.2%}")
    print(f"  Total Signals: {results['signals_per_period']['total']}")
    print(f"  Total Breakouts Captured: {results['total_breakouts_captured']}")


def train_gpu(args):
    """Train ML models with GPU acceleration"""
    from src.ml.train_gpu import run_gpu_pipeline

    run_gpu_pipeline(
        breakout_threshold=args.threshold,
        lookforward_days=args.days,
        use_lstm=args.lstm,
        xgb_estimators=args.xgb_trees,
        lgb_estimators=args.lgb_trees,
        lstm_epochs=args.lstm_epochs
    )


def serve_web(args):
    """Start web server"""
    print("=" * 60)
    print("STARTING WEB SERVER")
    print("=" * 60)
    print(f"Open http://localhost:{args.port} in your browser")
    print("Press Ctrl+C to stop")

    import uvicorn
    from src.web.app import app
    uvicorn.run(app, host=args.host, port=args.port)


def run_all(args):
    """Run complete pipeline"""
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 60)

    # Collect data
    collect_data(args)

    # Train model
    train_model(args)

    # Optimize filters
    optimize_filters(args)

    # Run backtest
    run_backtest(args)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("Run 'python main.py serve' to start the web dashboard")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Filter Optimizer - Find optimal stock screening criteria"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect S&P 500 data')
    collect_parser.add_argument('--period', default='10y', help='Data period (default: 10y)')

    # Train command (CPU)
    train_parser = subparsers.add_parser('train', help='Train ML models (CPU)')
    train_parser.add_argument('--threshold', type=float, default=0.20, help='Breakout threshold (default: 0.20)')
    train_parser.add_argument('--days', type=int, default=63, help='Lookforward days (default: 63)')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')

    # Train GPU command
    train_gpu_parser = subparsers.add_parser('train-gpu', help='Train ML models with GPU/CUDA acceleration')
    train_gpu_parser.add_argument('--threshold', type=float, default=0.20, help='Breakout threshold (default: 0.20)')
    train_gpu_parser.add_argument('--days', type=int, default=63, help='Lookforward days (default: 63)')
    train_gpu_parser.add_argument('--xgb-trees', type=int, default=1000, help='XGBoost trees (default: 1000)')
    train_gpu_parser.add_argument('--lgb-trees', type=int, default=1000, help='LightGBM trees (default: 1000)')
    train_gpu_parser.add_argument('--lstm-epochs', type=int, default=50, help='LSTM epochs (default: 50)')
    train_gpu_parser.add_argument('--lstm', action='store_true', default=True, help='Use LSTM (default: True)')
    train_gpu_parser.add_argument('--no-lstm', dest='lstm', action='store_false', help='Disable LSTM')

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize filter criteria')
    optimize_parser.add_argument('--top-n', type=int, default=30, help='Top N features to consider')
    optimize_parser.add_argument('--min-precision', type=float, default=0.4, help='Minimum precision')
    optimize_parser.add_argument('--target-recall', type=float, default=0.3, help='Target recall')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--threshold', type=float, default=0.20, help='Breakout threshold')
    backtest_parser.add_argument('--days', type=int, default=63, help='Lookforward days')
    backtest_parser.add_argument('--period-months', type=int, default=3, help='Backtest period months')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')

    # All command
    all_parser = subparsers.add_parser('all', help='Run complete pipeline')
    all_parser.add_argument('--period', default='10y', help='Data period')
    all_parser.add_argument('--threshold', type=float, default=0.20, help='Breakout threshold')
    all_parser.add_argument('--days', type=int, default=63, help='Lookforward days')
    all_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    all_parser.add_argument('--top-n', type=int, default=30, help='Top N features')
    all_parser.add_argument('--min-precision', type=float, default=0.4, help='Minimum precision')
    all_parser.add_argument('--target-recall', type=float, default=0.3, help='Target recall')
    all_parser.add_argument('--period-months', type=int, default=3, help='Backtest period months')

    args = parser.parse_args()

    if args.command == 'collect':
        collect_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'train-gpu':
        train_gpu(args)
    elif args.command == 'optimize':
        optimize_filters(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'serve':
        serve_web(args)
    elif args.command == 'all':
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
