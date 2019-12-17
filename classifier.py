import csv
import numpy as np
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv1D, MaxPooling1D
import pickle

def one_hot_encode(labels):
	mapping = { 'Mental health': 0, 
				'Skin issue': 1, 
				'Internal medicine': 2, 
				'Surgery department': 3, 
				'Other': 4, 
				'Neurology': 5 }
	one_hot_labels = [[0] * 6 for i in range(len(labels))]
	for i in range(len(one_hot_labels)):
		one_hot_labels[i][mapping[labels[i]]] = 1
	return one_hot_labels

if __name__ == "__main__":
	test_filenames = []
	f = open('test_filenames.txt', 'r+')
	for line in f.readlines():
		line = line.strip()
		test_filenames.append(line)
	f.close()

	files = ['./csv/all_acoustic_features_mean.csv', './csv/all_ngram_features.csv', './csv/all_pos_features.csv']

	print('...read files...')
	acoustic, ngram, pos, labels = [], [], [], []
	acoustic_test, ngram_test, pos_test, labels_test = [], [], [], []
	for file in files:
		with open(file) as f:
			reader = csv.reader(f)
			for line in reader:
				if file == './csv/all_acoustic_features_mean.csv':
					if line[0] in test_filenames:
						acoustic_test.append([float(i) for i in line[1:-4]])
						labels_test.append(line[-1])
					else:
						acoustic.append([float(i) for i in line[1:-4]])
						labels.append(line[-1])
				elif file == './csv/all_ngram_features.csv':
					if line[0] in test_filenames:
						ngram_test.append([float(i) for i in line[1:-1]])
					else:
						ngram.append([float(i) for i in line[1:-1]])
				else:
					if line[0] in test_filenames:
						pos_test.append([float(i) for i in line[1:-1]])
					else:
						pos.append([float(i) for i in line[1:-1]])

	# labels = one_hot_encode(labels)
	# labels_test = one_hot_encode(labels_test)

	train_X = []
	for i in range(len(labels)):
		feature = []
		feature += acoustic[i]
		feature += ngram[i]
		feature += pos[i]
		train_X.append(np.array(feature))

	test_X = []
	for i in range(len(labels_test)):
		feature = []
		feature += acoustic_test[i]
		feature += ngram_test[i]
		feature += pos_test[i]
		test_X.append(np.array(feature))

	train_X, train_y, test_X, test_y = np.array(train_X), np.array(labels), np.array(test_X), np.array(labels_test)
	# train_X, train_y = np.array(train_X), np.array(labels)

	# 6661
	# test_idx = np.random.choice(6661, int(6661*0.15), replace=False)
	# train_idx = np.array([i for i in range(6661) if i not in test_idx])
	# train_X = np.array([features[i] for i in train_idx])
	# train_y = np.array([labels[i] for i in train_idx])
	# test_X = np.array([features[i] for i in test_idx])
	# test_y = np.array([labels[i] for i in test_idx])
	# train_X, test_X = np.array(features[0:4661]), np.array(features[4661:])
	# train_y, test_y = np.array(labels[0:4661]), np.array(labels[4661:])

	imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp.fit(train_X)
	train_X = imp.transform(train_X)
	imp.fit(test_X)
	test_X = imp.transform(test_X)

	# train_X = np.array(train_X).reshape(4998, 206, 1)
	# test_X = np.array(test_X).reshape(1663, 206, 1)
	# train_y, test_y = labels, labels_test

	# classifier = OneVsRestClassifier(DecisionTreeClassifier())
	# classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
	classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=500))
	# classifier = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5, max_iter=6000))
	print('...start training...')
	classifier.fit(train_X, train_y)
	pickle.dump(classifier, open('finalized_model.sav', 'wb'))
	# classifier = pickle.load(open('finalized_model.sav', 'rb'))
	print('...start prediction...')
	predict_y = classifier.predict(test_X)

	count = 0
	for i in range(len(predict_y)):
		if predict_y[i] == test_y[i]:
			count += 1
	accuracy = float(count)/float(len(predict_y))
	print('accuracy: ' + str(accuracy))


	# convolutional neural network feature weight
	# print('...start training...')
	# model = Sequential()
	# model.add(Conv1D(256, 5, padding='same', input_shape=(206, 1))) #1
	# model.add(Activation('relu'))
	# model.add(Conv1D(128, 5, padding='same')) #2
	# model.add(Activation('relu'))
	# model.add(Dropout(0.1))
	# model.add(MaxPooling1D(pool_size=(8)))
	# model.add(Conv1D(128, 5, padding='same')) #3
	# model.add(Activation('relu'))
	# model.add(Conv1D(128, 5, padding='same')) #4
	# model.add(Activation('relu'))
	# model.add(Conv1D(128, 5, padding='same')) #5
	# model.add(Activation('relu'))
	# model.add(Dropout(0.2))
	# model.add(Conv1D(128, 5, padding='same')) #6
	# model.add(Activation('relu'))
	# model.add(Flatten())
	# model.add(Dense(6)) #7
	# model.add(Activation('softmax'))
	# opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
	# print('...start compile...')
	# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	# print('...start fitting...')
	# model.fit(np.array(train_X), np.array(train_y), batch_size=32, epochs=20, validation_split=0.15)


	# print('...evaluation...')
	# score = model.evaluate(np.array(test_X), np.array(test_y), verbose=0)
	# print('' + str(score[0]))
	# print('accuracy: ' + str(score[1]))
