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
dsMean = [0.49139968, 0.48215827, 0.44653124]
dsStd = [0.24703233, 0.24348505, 0.26158768]