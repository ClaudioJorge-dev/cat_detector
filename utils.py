
def is_cat(predicted_class):
    return predicted_class in [281, 282, 283, 284, 285]

def get_cat_breed(predictions):
    for pred in predictions:
        if is_cat(pred.cat_class):
            return pred
    
    return None