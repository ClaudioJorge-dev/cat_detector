import urllib.request

def load_imagenet_labels(): 
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    urllib.request.urlretrieve(url, "imagenet_classes.txt")
    
    labels = []
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    
    return labels

labels = load_imagenet_labels()

def get_label(category_id):
    return labels[category_id]

#print(get_label(282))