import sklearn.metrics as sk
import numpy as np

def get_fnpr(labels,scores,fixed_tpr=0.95):
    fpr, tpr, thresholds = sk.roc_curve(labels, scores)
    
    ### fit a 3rd order polynomial
    z = np.polyfit(tpr, fpr, 3)
    f = np.poly1d(z)
    
    return f(fixed_tpr)

def get_metrics(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    scores = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(scores), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, scores)
    aupr=sk.average_precision_score(labels, scores)
    fnpr=get_fnpr(labels,scores,fixed_tpr=0.95)

    return {'auroc':auroc,'aupr':aupr,'fnpr':fnpr}

def get_and_print_results(ood_loader):
    out_score = get_ood_scores(ood_loader)
    metrics = get_metrics(out_score, in_score)
    # print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    return metrics

def classification_accuracy(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    scores = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(scores), dtype=np.int32)
    labels[:len(pos)] += 1

    accuracy = sklearn.metrics.accuracy_score(labels,scores)
    f1_score = sklearn.metrics.f1_score(labels,scores)
    precision = sklearn.metrics.precision_score(labels,scores)
    recall = sklearn.metrics.recall_score(labels,scores)

    return accuracy, f1_score, precision,recall