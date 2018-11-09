import autosklearn.classification as asc
import autosklearn as asc_t
import sklearn as sk
import pandas as pd
import numpy as np
import scores
import utilities as ut
import pickle
import smoteenn
import sys


def main():

    time = 60*5

    runs = 5

    option = int(input("1. Imbalanced\n2. SMOTE + ENN\n"))

    for j in range(0, runs):

        file = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

        file = ut.prune_data(file)

        X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(file.drop('Churn', axis=1), file['Churn'], test_size=0.3)

        # Choosing kappa as the metric
        kappas = asc_t.metrics.make_scorer('kappa', sk.metrics.cohen_kappa_score)

        if option == 1:

            automl = asc.AutoSklearnClassifier(
                        time_left_for_this_task=time+10,
                        per_run_time_limit=time,
                        initial_configurations_via_metalearning=0,
                        # resampling_strategy='cv',
                        # resampling_strategy_arguments={'folds': 5},
                        )

            automl.fit(X_train, Y_train, metric=kappas)

            # automl.refit(X_train, Y_train)

            # Getting the rank of models
            scores.save_log(automl, "log_imbal.txt")

            # Predicting the model
            scores.predict_and_save(automl, X_test, Y_test, verbose=True, file="prediction.txt")

            # Save the model into binary code
            filename = str(j)+'imbal_model.sav'
            pickle.dump(automl, open(filename, 'wb'))

        elif option == 2:

            # Applying SMOTE + ENN
            X_resampled, Y_resampled = smoteenn.run(X_train, Y_train, smote_kind="svm")

            resample = asc.AutoSklearnClassifier(
                        time_left_for_this_task=time+10,
                        per_run_time_limit=time,
                        initial_configurations_via_metalearning=0
                        )

            resample.fit(X_resampled, Y_resampled, metric=kappas)

            # Getting the rank of models
            scores.save_log(resample, "log_bal.txt")

            scores.predict_and_save(resample, X_test, Y_test, verbose=True, file="prediction_balanced.txt")

            # Save the model into binary code
            filename = str(j)+'bal_model.sav'
            pickle.dump(resample, open(filename, 'wb'))

        else:

            print("Invalid Option")
            sys.exit()


if __name__ == "__main__":
    main()
