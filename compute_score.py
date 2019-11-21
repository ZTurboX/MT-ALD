# coding=utf-8

import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score,precision_score,recall_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def score(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        precision=precision_score(y_true=labels, y_pred=preds)
        recall=recall_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "precision":precision,
            "recall":recall,
        }


    def compute_metrics(task_name, preds, labels):
        print("preds:",len(preds))
        print("label:",len(labels))
        assert len(preds) == len(labels)

        if task_name=="aggression":
            return {"score": score(preds, labels)}
        elif task_name=="attack":
            return {"score": score(preds, labels)}
        elif task_name=="toxicity":
            return {"score": score(preds, labels)}
        elif task_name=="multi_task":
            return {"score": score(preds, labels)}
        else:
            raise KeyError(task_name)
