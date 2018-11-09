import sklearn as sk
import autosklearn as asc

def predict_and_save(classifier, X_test, Y_test, verbose=False, file=None):
    if(verbose):
        print('Classification results...')

    predictions = classifier.predict(X_test)
    # print(classification_report_imbalanced(Y_test, predictions))
    cm = sk.metrics.confusion_matrix(Y_test, predictions)

    tn, fp, fn, tp = cm.ravel()
    pos = tp + fn + 0.0
    neg = fp + tn + 0.0

    acc = float(tp + tn)/float(pos + neg)
    prec = float(tp)/float(tp + fp)
    sens = float(tp)/float(tp + fn)
    spec = float(tn)/float((tn + fp))
    fscore = float(2*tp)/float(2*tp + fp + fn)

    kappa = sk.metrics.cohen_kappa_score(Y_test, predictions)

    if(verbose):
        print("Acc\t\tPrec\t\tSens\t\tSpec\t\tFscore\t\tKappa\t\tTP\tFN\tFP\tTN")
        print("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}".format(acc,prec,sens,spec,fscore,kappa,tp,fn,fp,tn))

    if(file != None):
        with open(file, "a") as arch:
            arch.write("{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:2f}\t{:d}\t{:d}\t{:d}\t{:d}\n".format(acc,prec,sens,spec,fscore,kappa,tp,fn,fp,tn))

def save_log(classifier, file):

    # Getting the rank of models
    i = 0
    for key in classifier.cv_results_.items():
        if i < 5:
            with open(file, "a") as arch:
                arch.write("{}\n\n".format(key))
        else:
            break
        i = i + 1
