import sklearn.metrics as sk

auroc_list = []


def get_metrics(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc



def get_and_print_results(ood_loader):
    out_score = get_ood_scores(ood_loader)
    auroc = get_metrics(out_score, in_score)
    # print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    return auroc