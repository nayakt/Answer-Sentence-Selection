def calc_mean_avg_prec(preds):
    """
    skip all questions w/o correct answers
    and all questions w/ only correct answers
    """
    mean_avg_prec, relQ = 0.0, 0.0
    for pred in preds.values():
        cnt = 0
        for tri in pred:
            cnt += tri[1]
        if cnt == 0 or cnt == len(pred):
            continue
        sorted_pred = sorted(pred, key=lambda res: res[1])
        sorted_pred = sorted(sorted_pred, key=lambda res: res[2], reverse=True)
        avg_prec, rel = 0.0, 0.0
        for i, tri in enumerate(sorted_pred):
            if tri[1] == 1:
                rel += 1.0
                avg_prec += rel / (i + 1)
        avg_prec /= rel
        mean_avg_prec += avg_prec
        relQ += 1.0
    mean_avg_prec /= relQ
    return mean_avg_prec

def calc_mean_reciprocal_rank(preds):
    """
    skip all questions w/o correct answers
    and all questions w/ only correct answers
    """
    mean_reciprocal_rank, relQ = 0.0, 0.0
    for pred in preds.values():
        cnt = 0
        for tri in pred:
            cnt += tri[1]
        if cnt == 0 or cnt == len(pred):
            continue
        sorted_pred = sorted(pred, key=lambda res: res[1])
        sorted_pred = sorted(sorted_pred, key=lambda res: res[2], reverse=True)
        reciprocal_rank, rel = 0.0, 0.0
        for i, tri in enumerate(sorted_pred):
            if tri[1] == 1:
                rel += 1.0
                reciprocal_rank += rel / (i + 1)
                break
        relQ += 1.0
        mean_reciprocal_rank += reciprocal_rank
    mean_reciprocal_rank /= relQ
    return mean_reciprocal_rank

def calc_trigger_fscore(preds, thre=0.1):
    """
    precision, recall, fmeasure for the task of answering triggering
    """
    gt_cnt, pred_cnt, match_cnt = 0.0, 0.0, 0.0
    for pred in preds.values():
        sorted_pred = sorted(pred, key=lambda res: res[2], reverse=True)
        if sorted_pred[0][2] > thre:
            pred_cnt += 1.0
            if sorted_pred[0][1] == 1:
                match_cnt += 1.0
        sorted_gt = sorted(pred, key=lambda res: res[1], reverse=True)
        if sorted_gt[0][1] == 1:
            gt_cnt += 1.0
    prec, reca = match_cnt / pred_cnt, match_cnt / gt_cnt
    return prec, reca, 2*prec*reca / (prec+reca)