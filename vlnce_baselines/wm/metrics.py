"""Deviation-detection metrics (no sklearn dependency)."""

import numpy as np


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    for v in np.unique(scores):
        tie = scores == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if labels.sum() == 0:
        return float('nan')
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def false_alarm_rate(scores, labels, threshold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    neg = labels == 0
    if neg.sum() == 0:
        return float('nan')
    return float((scores[neg] > threshold).mean())


def deviation_report(records, threshold):
    """records: iterable of (p_dev, label, revised) with label in {0, 1, None}."""
    rec = [(p, l) for p, l, _ in records if l is not None]
    out = {'num_verified_steps': len(rec),
           'num_revisions': int(sum(1 for _, _, r in records if r))}
    if len(rec) == 0:
        return out
    scores, labels = zip(*rec)
    out.update({
        'dev_rate': float(np.mean(labels)),
        'dev_auroc': auroc(scores, labels),
        'dev_auprc': auprc(scores, labels),
        'dev_false_alarm': false_alarm_rate(scores, labels, threshold),
    })
    return out
