
import torch

from PredictionObj import PredictionObj
from imagenet_load import get_label


def is_cat(predicted_class):
    return predicted_class in [281, 282, 283, 284, 285]

def get_cat_breed_from_probs(probs):
    top5_prob, top5_catid = torch.topk(probs, 5)
    for prob, catid in zip(top5_prob, top5_catid):
        if is_cat(catid.item()):
            return PredictionObj(catid.item(), prob.item(), get_label(catid.item()))
    
    return None