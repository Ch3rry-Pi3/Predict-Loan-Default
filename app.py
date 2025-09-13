from src.preprocessing.import_data import import_data
from src.preprocessing.clean_data import clean_data
from src.feature_engineering import feature_engineering
from src.experiments.exp_cli import run_cli

def main():
    import_data()
    clean_data()
    feature_engineering()
    run_cli()

if __name__ == "__main__":
    main()