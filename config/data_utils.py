import pandas as pd
import numpy as np

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma',
    'norm':'Normal'
}

# all_classes= ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
all_classes = ['nv','mel','bkl','bcc','akiec','vasc','df']
# dsMean = [0.49139968, 0.48215827, 0.44653124]
dsMean = [0.4914, 0.4822, 0.4465]
# dsStd = [0.24703233, 0.24348505, 0.26158768]
dsStd = [0.2023, 0.1994, 0.2010]

def estimate_weights_mfb(label,filepath):
    metadata = pd.read_csv(filepath)
    class_weights = np.zeros_like(label, dtype=np.float)
    counts = np.zeros_like(label)
    for i,l in enumerate(label):
        counts[i] = metadata[metadata['dx']==str(l)]['dx'].value_counts()[0]
    counts = counts.astype(np.float)
    #print(counts)
    median_freq = np.median(counts)
    #print(median_freq)
    #print(weights.shape)
    for i, label in enumerate(label):
        #print(label)
        class_weights[i] = median_freq / counts[i]
    return class_weights

classweight= estimate_weights_mfb(all_classes,'../data/HAM10000_metadata.csv')
class_weights = torch.FloatTensor(class_weights)

for i in range(len(all_classes)):
    print(all_classes[i],":", classweight[i])