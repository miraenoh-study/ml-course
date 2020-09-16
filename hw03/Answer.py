import numpy as np

def Accuracy(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the accuracy for prediction.
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - Acc : (scalar, float), Computed accuracy score
    # ========================= EDIT HERE =========================
    Acc = 0.

    n = label.shape[0]
    for i in range(0, n):
        if label[i] == pred[i]:
            Acc += 1.

    Acc /= n

    # print("Acc: ", Acc)

    # =============================================================
    return Acc

def Precision(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Precision for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - precision : (scalar, float), Computed precision score
    # ========================= EDIT HERE =========================
    precision = 0.

    nTP = 0
    nPP = 0

    n = label.shape[0]
    for i in range(0, n):
        if pred[i] == 1:
            nPP += 1
            if label[i] == 1:
                nTP += 1

    if nPP == 0:
        return 1
    else:
        precision = nTP / nPP

    # =============================================================
    return precision

def Recall(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Recall for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - recall : (scalar, float), Computed recall score
    # ========================= EDIT HERE =========================
    recall = 0.

    nTP = 0
    nAP = 0

    n = label.shape[0]
    for i in range(0, n):
        if label[i] == 1:
            nAP += 1
            if pred[i] == 1:
                nTP += 1

    recall = nTP / nAP

    # =============================================================
    return recall

def F_measure(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the F-measure score for prediction.
    #         you can erase the code. (F_score = 0.)
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - F_score : (scalar, float), Computed F-score score
    # ========================= EDIT HERE =========================
    F_score = None

    precision = Precision(label, pred)
    recall = Recall(label, pred)

    F_score = (2 * precision * recall) / (precision + recall)

    # =============================================================
    return F_score

def MAP(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the code to calculate the MAP for prediction.
    #         Notice that, hypo is the real value array in (0, 1)
    #         MAP (at = 10) means MAP @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed MAP score
    # ========================= EDIT HERE =========================
    Map = 0.
    nQuery = label.shape[0]
    nElem = label.shape[1]

    ap = list()

    pred = np.zeros_like(hypo, dtype=np.int32)
    pred[hypo >= 0.5] = 1

    for i in range(0, nQuery):
        nPP = 0
        nTP = 0
        precision = list()
        for j in range(0, nElem):
            if pred[i][j] == 1:
                nPP += 1
                if label[i][j] == 1:
                    nTP += 1
                    if j < at:
                        precision.append(nTP/nPP)
        if nPP != 0:
            ap.append(np.sum(precision) / nPP)


    Map = np.mean(ap)

    # =============================================================
    return Map

def nDCG(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the each code to calculate the nDCG for prediction.
    #         you can erase the code. (dcg, idcg, ndcg = 0.)
    #         Notice that, hypo is the real value array in (0, 1)
    #         nDCG (at = 10 ) means nDCG @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed nDCG score


    def DCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        dcg = 0.
        n =  label.shape[0]
        nPP = 0

        n = label.shape[0]
        for i in range(0, at):
            if hypo[i] == 1:
                nPP += 1
                if label[i] == 1:
                    dcg += (1 / np.log2(nPP+1))

        # =============================================================
        return dcg

    def IDCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        idcg = 0.
        n = label.shape[0]
        nPP = 0
        nTP = 0

        for i in range(0, n):
            if hypo[i] == 1 and label[i] == 1:
                nTP += 1

        nRV = at
        if at > nTP:
            nRV = nTP

        for i in range(0, nRV):
            idcg += (1 / np.log2(i+2))

        # =============================================================
        return idcg
    # ========================= EDIT HERE =========================
    ndcg = None
    ndcgs = list()

    pred = np.zeros_like(hypo, dtype=np.int32)
    pred[hypo >= 0.5] = 1
    nQuery = label.shape[0]
    nElem = label.shape[1]

    for n in range(0, nQuery):
        curLabel = label[n]
        curPred = pred[n]
        dcg = DCG(curLabel, curPred, at)
        idcg = IDCG(curLabel, curPred, at)
        if idcg == 0:
            ndcgs.append(0.)
        else:
            ndcgs.append(dcg / idcg)

    ndcg = np.mean(ndcgs)

    # =============================================================
    return ndcg

# =============================================================== #
# ===================== DO NOT EDIT BELOW ======================= #
# =============================================================== #

def evaluation_test1(label, pred, at = 10):
    result = {}

    result['Accuracy '] = Accuracy(label, pred)
    result['Precision'] = Precision(label, pred)
    result['Recall   '] = Recall(label, pred)
    result['F_measure'] = F_measure(label, pred)

    return result

def evaluation_test2(label, hypo, at = 10):
    result = {}

    result['MAP  @%d'%at] = MAP(label, hypo, at)
    result['nDCG @%d'%at] = nDCG(label, hypo, at)

    return result
