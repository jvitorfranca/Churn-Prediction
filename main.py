import autosklearn.classification as asc
import autosklearn as asc_t
import sklearn as sk
import pandas as pd
import numpy as np
import scores
import utilities as ut


def main():

    time = 60

    file = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    file = ut.prune_data(file)

    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(file.drop('Churn', axis=1), file['Churn'], test_size=0.3)

    dt = asc.AutoSklearnClassifier(
                time_left_for_this_task=time+10,
                per_run_time_limit=time,
                ensemble_size=1,
                initial_configurations_via_metalearning=0
                )

    kappas = asc_t.metrics.make_scorer('kappa', sk.metrics.cohen_kappa_score)

    dt.fit(X_train, Y_train, metric=kappas)


    print("------------------------ \t cv_results_ \t -------------------------------")
    i = 0
    for key in dt.cv_results_.items():
        if i < 5:
            print(key)
            print("\n")
        else:
            break
        i = i + 1
    print("------------------------ \t end cv_results_ \t ---------------------------")

    # scores.predict_and_save(dt, X_test, Y_test, verbose=True, file="RF.txt")

if __name__ == "__main__":
    main()
