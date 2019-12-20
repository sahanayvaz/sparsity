import os
import shutil


log_dir = '/cluster/home/sayvaz/sparsity/log_dir'
save_dir = '/cluster/home/sayvaz/sparsity/save_dir'
os.makedirs(save_dir, exist_ok=True)

list_dir = os.listdir(log_dir)

for l in list_dir:
	full_l = os.path.join(log_dir, l)
	l_dir = os.listdir(full_l)
	for j in l_dir:
		save_dir_l = os.path.join(save_dir, l)
		os.makedirs(save_dir_l, exist_ok=True)
		if not '.npz' in j:
			try:
				file_path = os.path.join(full_l, j)
				shutil.move(file_path, save_dir_l)
			except:
				print('passing {}'.format(j))
				pass
print('done')