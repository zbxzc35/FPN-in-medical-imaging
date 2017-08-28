import os
import numpy as np
train_data_path = '/mnt/DSB2017/tianchi_data/all_data_segment2/train_segment2'
train_save_file = 'kaggleluna_full.npy'

val_data_path = '/mnt/DSB2017/tianchi_data/all_data_segment2/val_segment2'
val_save_file = 'valsplit.npy'

def main(data_path = None, save_file = None):
	patients = []
	file_list = os.listdir(data_path)
	for file in file_list:
		patients.append(file.split('_')[0])
		patients = list(set(patients))
	#patients = ['LKDS-00001', 'LKDS-00002', 'LKDS-00003', 'LKDS-00004']
	patients = np.array(patients)
	np.save(save_file, patients)
	print(len(patients))

if __name__ == '__main__':
	main(train_data_path, train_save_file)
	main(val_data_path, val_save_file)