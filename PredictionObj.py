

class PredictionObj:
    def __init__(self, cat_class, probability, label):
        self.cat_class = cat_class
        self.probability = probability
        self.label = label   
        
    def __repr__(self):
        return f"{self.label} with probability {self.probability*100:.2f}%"