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
dsMean = [0.763038, 0.54564667, 0.57004464]
dsStd = [0.14092727, 0.15261286, 0.1699712]