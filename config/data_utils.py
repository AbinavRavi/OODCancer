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