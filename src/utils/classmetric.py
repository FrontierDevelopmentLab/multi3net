import numpy as np

class ClassMetric(object):
    def __init__(self, num_classes=2, ignore_index=-100):
        self.num_classes = num_classes
        _range = -0.5, num_classes - 0.5
        self.range = np.array((_range, _range), dtype=np.int64)
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    def _update(self, o, t):
        t = t.flatten()
        o = o.flatten()
        # confusion matrix
        n, _, _ = np.histogram2d(t, o, bins=self.num_classes, range=self.range)
        self.hist += n

    def _metrics(self):
        if self.ignore_index != -100:
            self.hist[self.ignore_index] = 0


        # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        N = np.sum(self.hist)
        p0=np.sum(np.diag(self.hist))/N
        pc=np.sum(np.sum(self.hist,axis=0)*np.sum(self.hist,axis=1))/N**2
        kappa = (p0 - pc) / (1 - pc)

        recall = np.diag(self.hist) / np.sum(self.hist, axis=1)
        precision = np.diag(self.hist) / np.sum(self.hist, axis=0)
        f1 = (2 * precision * recall) / (precision + recall)

        # Per class IoU
        iou = np.diag(self.hist) / (
            self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + 1e-12)
        if self.ignore_index != -100:
            iou = np.delete(iou, self.ignore_index)

        # Per class accuracy
        cl_acc = np.diag(self.hist) / (self.hist.sum(1) + 1e-12)
        if self.ignore_index != -100:
            cl_acc = np.delete(cl_acc, self.ignore_index)

        acc = np.diag(self.hist).sum() / (self.hist.sum() + 1e-12)
        return {"mIoU": float(np.mean(iou)),
                "iou_background": iou[0],
                "iou_building": iou[1],
                "precision_background": precision[0],
                "precision_building": precision[1],
                "recall_background": recall[0],
                "recall_building": recall[1],
                "f1_background": f1[0],
                "f1_building": f1[1],
                "mAcc": float(np.nanmean(cl_acc)),
                "Acc": float(acc),
                "Kappa": kappa}

    def __call__(self, target, output):
        for o, t in zip(output, target):
            self._update(o.data.max(0)[1].cpu().numpy(), t.data.cpu().numpy())
        return self._metrics()