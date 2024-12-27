import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.preprocessing import label_binarize
from .build import EVALUATOR_REGISTRY
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, confusion_matrix, f1_score


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        pred = pred % 2
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = self._correct / self._total
        err = 1 - acc
        macro_f1 = f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # 计算 Macro Average Precision (Macro AP)
        classes = np.unique(self._y_true)
        y_true_bin = label_binarize(self._y_true, classes=classes)
        y_pred_bin = label_binarize(self._y_pred, classes=classes)
        macro_ap = average_precision_score(y_true_bin, y_pred_bin, average="macro")


        y_true = np.array(self._y_true)
        y_pred = np.array(self._y_pred)

        ap = average_precision_score(y_true, y_pred)
        r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0])
        f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])
        acc_ = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["macro_ap"] = macro_ap

        results["acc"] = acc_
        results["r_acc"] = r_acc
        results["f_acc"] = f_acc
        results["f1"] = f1
        results["ap"] = ap
        results["cm"] = cm

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.4f}%\n"
            f"* error: {err:.4f}%\n"
            f"* macro_f1: {macro_f1:.4f}\n"
            f"* macro_ap: {macro_ap:.4f}\n\n%"
            
            f"* acc: {acc:.2f}\n%"
            f"* r_acc: {r_acc:.2f}\n%"
            f"* f_acc: {f_acc:.2f}\n%"
            f"* f1: {f1:.2f}\n%"
            f"* ap: {ap:.2f}\n%"
            f"* cm: {cm}"
        )
        results["cm"] = str(cm)

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
