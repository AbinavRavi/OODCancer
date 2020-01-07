import pickle
import glob

def read_pickle(file_path):
    with open(file_path, 'rb') as handle:
        content = pickle.load(handle)

    return content

all_healthy_file=[]

all_pickle=glob.glob('./click_farm/**')
for pkl in all_pickle:
    all_healthy_file.append(read_pickle(pkl))

with open('./click_farm/final_healthy.pickle', 'wb') as handle:
    pickle.dump(all_healthy, handle, protocol=pickle.HIGHEST_PROTOCOL)

