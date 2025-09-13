
from src.preprocessing.preprocess_cli import run_preprocessing
from src.experiments.exp_cli import run_experiments

def main():
    run_preprocessing()
    run_experiments()

if __name__ == "__main__":
    main()