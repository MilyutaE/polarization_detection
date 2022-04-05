import pandas as pd
import numpy as np
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score


class SimpleMetrics():
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.tp, self.tn, self.fp, self.fn = None, None, None, None
        
    def __call__(self, y_true, y_pred):
        assert len(y_true) == len(y_pred), 'Lens must be the same'
        print(f'New data passed with shape {len(y_true)}')
        
        pairs = dict(zip(range(len(y_true)), zip(y_true, y_pred)))
    
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in pairs.values():
            for j in pairs.values():
                i_true, i_pred = i
                j_true, j_pred = j
                if i_true == j_true:
                    if i_pred == j_pred:
                        tp +=1
                    else:
                        fn +=1
                else:
                    if i_pred == j_pred:
                        fp += 1
                    else:
                        tn += 1
        self.tp, self.tn, self.fp, self.fn = tp, tn, fp, fn
        
    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)
    
    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)
    
    @property
    def f1(self):
        return (self.precision * self.recall * 2) / (self.precision + self.recall)

    def get_report(self):
        return {
            'presicion': self.precision,
            'recall': self.recall,
            'F1': self.f1
        }
    
    
class EntropyMetrics():
    def __init__(self):
        pass
        
    def __call__(self, y_true, y_pred):
        assert len(y_true) == len(y_pred), 'Lens must be the same'
        print(f'New data passed with shape {len(y_true)}')

        self.homogenity = homogeneity_score(y_true, y_pred)
        self.completeness = completeness_score(y_true, y_pred)
        self.v_measure = v_measure_score(y_true, y_pred)

    def get_report(self):
        return {
            'V_measure': self.v_measure,
            'homogenity': self.homogenity,
            'completeness': self.completeness
        }
    
    
def get_report(y_true, y_pred):
    simple_metr = SimpleMetrics()
    simple_metr(y_true, y_pred)
    
    entr_metr = EntropyMetrics()
    entr_metr(y_true, y_pred)
    
    metrics = entr_metr.get_report()
    metrics.update(simple_metr.get_report())
    return metrics
    
    
