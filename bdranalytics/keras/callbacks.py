from keras.callbacks import Callback
from keras.models import Sequential
from sklearn.metrics import roc_auc_score, average_precision_score, \
    precision_score, recall_score


class EpochEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.metrics = {}

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0:
            print(" - ".join(["val_{:s}: {:.4f}".format(k, v)
                              for k, v in self.metrics.items()]))

    def on_epoch_end(self, epoch, logs={}):
        if isinstance(self.model, Sequential):
            predict = self.model.predict_proba
        else:
            predict = self.model.predict

        y_pred = predict(self.X_val, verbose=0)
        y_pred_bin = y_pred > 0.5

        y_true = self.y_val
        self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        self.metrics['pr_auc'] = average_precision_score(
            y_true, y_pred, average="micro")
        self.metrics['recall'] = recall_score(y_true, y_pred_bin)
        self.metrics['precision'] = precision_score(y_true, y_pred_bin)
