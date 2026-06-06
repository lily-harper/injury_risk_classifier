from src.individual_models import (run_dummy_models,
                                   run_logistic_regression,
                                   run_decision_tree,
                                   run_naive_bayes)
from src.pipeline import run_model_comparisons 

def main():
    run_dummy_models.main()
    run_logistic_regression.main()
    run_decision_tree.main()
    run_naive_bayes.main()
    run_model_comparisons.main()


if __name__ == "__main__":
    main()