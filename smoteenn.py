from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.svm import SVC


def run(X=None, Y=None, random_state=42, smote_ratio="minority", smote_kind="regular", enn_ratio="all", enn_kind_sel="all", enn_n_neighbors=3, save_dist=False, file=None):

    sm = None

    if smote_kind == "svm":
        sm = SMOTE(random_state=random_state, ratio=smote_ratio, kind=smote_kind, svm_estimator=SVC())
    else:
        sm = SMOTE(random_state=random_state, ratio=smote_ratio, kind=smote_kind,)

    enn = EditedNearestNeighbours(random_state=random_state, ratio=enn_ratio, kind_sel=enn_kind_sel, n_neighbors=enn_n_neighbors)

    X_resampled, Y_resampled = sm.fit_sample(X, Y)

    if(save_dist):
        with open(file, "a") as arch:
           arch.write("SMOTE: " + str(Counter(Y_resampled)) + " ")

    X_st, Y_st = enn.fit_sample(X_resampled, Y_resampled)

    if(save_dist):
        with open(file, "a") as arch:
            arch.write("ENN:" + str(Counter(y_st))+"\n")

    return X_st, Y_st
