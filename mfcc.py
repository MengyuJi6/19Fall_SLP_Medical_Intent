from tqdm import tqdm
import csv
import librosa
import numpy as np


# def feature_normalize(dataset):
# 	mu = np.mean(dataset, axis=0)
# 	sigma = np.std(dataset, axis=0)
# 	return (dataset - mu) / sigma
# def windows(data, window_size):
# 	start = 0
# 	while start < len(data):
# 		yield int(start), int(start + window_size)
# 		start += (window_size / 2)

if __name__ == "__main__":
	# symptoms = {}
	# 9 id 11 label
	fw = open('mfcc.csv', 'w+')

	with open('overview-of-recordings-processed.csv', 'r+') as f:
		reader = csv.reader(f)
		header = True
		t = tqdm(total=6662)
		# l = []
		for line in reader:
			if header:
				header = False
				continue

			t.update(1)
			audio = './recordings/' + line[9]
			y, sr = librosa.load(audio)

			# l.append(len(y))
			# y = feature_normalize(y)
			# for (start,end) in windows(y, 20480):
			# 	if(len(y[start:end]) == 20480):
			# 		signal = y[start:end]
			# 		# y: audio time series, sr: sampling rate, n_mfcc: number of MFCCs to return
			# 		# librosa.feature.mfcc() function return numpy array with shape (bands, frames)
			# 		# transpose since the model expects time axis(frames) come first
			# 		mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc = 20).T 
			# 		print(mfcc.shape)
		# print(max(l))
					# mfccs.append(mfcc)
					# labels.append(label)
			mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) #, n_mfcc=13
			feature = []
			for coefficient in mfcc:
				feature += [max(coefficient), min(coefficient), np.mean(coefficient), np.std(coefficient), np.median(coefficient)]

			# print(feature)
			# print(len(y))
			# print(mfcc.shape)
			# # break
			# mfcc = mfcc.reshape(640, 1)

			formatted = line[9]
			for i in feature:
				formatted += ',' + str(i)
			formatted += ',' + line[11] + '\n'
			fw.writelines(formatted)

