import pickle
import glob

def read_pickle(file_path):
    with open(file_path, 'rb') as handle:
        content = pickle.load(handle)

    return content

all_healthy_file=[]

all_pickle=glob.glob('./click_farm/**')

for pkl in all_pickle:
    if 'healthy' not in pkl:    
        all_healthy_file+=read_pickle(pkl)
all_healthy_file=list(set(all_healthy_file))

print(f'TOTAL HEALTHY SAMPLES:{len(all_healthy_file)}')

with open('./click_farm/final_healthy.pickle', 'wb') as handle:
    pickle.dump(all_healthy_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


