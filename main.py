import argparse
from pathlib import Path
from importlib import import_module

def main():
    parser = argparse.ArgumentParser(description="Run machine learning examples")
    parser.add_argument(
        "--model",
        choices=["sentiment", "spam", "recommend", "timeseries"],
        required=True,
        help="Which model to run"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to data file (default uses built-in samples)"
    )
    
    args = parser.parse_args()
    
    # Import and run the selected model
    module = import_module(f"ml_models.{args.model}_analysis" 
                          if args.model == "sentiment" else 
                          f"ml_models.{args.model}_detection" 
                          if args.model == "spam" else 
                          f"ml_models.{args.model}")
    
    if args.data:
        module.run(args.data)
    else:
        module.run()

if __name__ == "__main__":
    main()