import autosklearn.classification as asc
import autosklearn as asc_t
import sklearn as sk
import pandas as pd
import numpy as np
import scores
import utilities as ut
import pickle


def main():

    time = 60*10

    runs = 6

    for j in range(0, runs):

        file = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

        file = ut.prune_data(file)

        X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(file.drop('Churn', axis=1), file['Churn'], test_size=0.3)

        automl = asc.AutoSklearnClassifier(
                    time_left_for_this_task=time+10,
                    per_run_time_limit=time,
                    initial_configurations_via_metalearning=0,
                    # resampling_strategy='cv',
                    # resampling_strategy_arguments={'folds': 5},
                    )

        # Choosing kappa as the metric
        kappas = asc_t.metrics.make_scorer('kappa', sk.metrics.cohen_kappa_score)

        automl.fit(X_train, Y_train, metric=kappas)

        # automl.refit(X_train, Y_train)

        # Getting the rank of models
        i = 0
        for key in automl.cv_results_.items():
            if i < 5:
                with open("log.txt", "a") as arch:
                    arch.write("{}\n\n".format(key))
            else:
                break
            i = i + 1

        # Predicting the model
        scores.predict_and_save(automl, X_test, Y_test, verbose=True, file="prediction.txt")

        # Save the model into binary code
        filename = str(j)+'finalized_model.sav'
        pickle.dump(automl, open(filename, 'wb'))

if __name__ == "__main__":
    main()
