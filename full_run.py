import numpy as np

import config as cfg
import cross_val as cv


def run():
    data = np.loadtxt(f"{cfg.DATASET}_dataset.txt")

    # CROSS VALIDATION
    global_best_hyper, global_scores, avg_scores = cv.cross_validation(data, 10, 0.1)
    print(global_best_hyper)
    print(global_scores)
    print(avg_scores)

    # PARAM TUNING
    best_hyper, F1_score, err_all_hyperparams = cv.param_tuning(data, 10, 0.1)

    print(best_hyper)
    print(F1_score)
    print(err_all_hyperparams)


if __name__ == "__main__":
    # Read command line arguments into args dict
    run()
