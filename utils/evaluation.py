'''
    From https://github.com/ma-xu/Open-Set-Recognition/blob/master/Utils/evaluation.py#L8
'''
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

class Evaluation(object):
    """Evaluation class based on python list"""
    def __init__(self, predict, label, prediction_scores = None):
        self.predict = predict
        self.label = label
        self.prediction_scores = prediction_scores
        self.accuracy = self._accuracy()
        if self.prediction_scores is not None:
            self.area_under_roc_weighted = self._area_under_roc(prediction_scores,average='weighted')
            self.area_under_roc_per_class = self._area_under_roc_per_class(prediction_scores)
    def _accuracy(self) -> float:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))

    def _area_under_roc(self, prediction_scores: np.array = None, multi_class='ovr',average='macro') -> float:
        """
        Area Under Receiver Operating Characteristic Curve

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
            prediction scores for each class. If not specified, will generate its own prediction scores that assume
            100% confidence in selected prediction.
        :param multi_class: {'ovo', 'ovr'}, default='ovo'
            'ovo' computes the average AUC of all possible pairwise combinations of classes.
            'ovr' Computes the AUC of each class against the rest.
        :return: float representing the area under the ROC curve
        """
        label, predict = self.label, self.predict
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class,average=average)

    def _area_under_roc_per_class(self, prediction_scores: np.array = None) -> float:
        """
        Area Under Receiver Operating Characteristic Curve

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
            prediction scores for each class. If not specified, will generate its own prediction scores that assume
            100% confidence in selected prediction.
        :return: float representing the area under the ROC curve
        """
        label, predict = self.label, self.predict
        one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        prediction_scores_t = np.array(prediction_scores)
        aucs = []
        for i in range(self.label.max() + 1):
            aucs.append(roc_auc_score(true_scores[:, i], prediction_scores_t[:, i]))
        return aucs