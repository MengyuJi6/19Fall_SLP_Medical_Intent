import flask
import os
from flask import Flask, escape, request
from flask_cors import CORS
import speech_recognition as sr
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import glob
import json
import nltk
from nltk.data import load
import pickle

app = Flask(__name__)
CORS(app)


def speechRecognition(f):
	r = sr.Recognizer()
	hello = sr.AudioFile(f)
	with hello as source:
		audio = r.record(source)
	try:
		s = r.recognize_google(audio, language="en-US")
		return s
	except Exception as e:
		print("Exception: "+str(e))

def extract_mfcc(f):
	y, sr = librosa.load(f)
	mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
	feature = []
	for coefficient in mfcc:
		feature += [max(coefficient), min(coefficient), np.mean(coefficient), np.std(coefficient), np.median(coefficient)]
	return feature

def extractPitch(sound, pitchFloor, pitchCeiling, unit, interpolation):
	pitch = call(sound, "To Pitch", 0.0, pitchFloor, pitchCeiling)
	minPitch = call(pitch, "Get minimum", 0, 0, unit, interpolation)
	maxPitch = call(pitch, "Get maximum", 0, 0, unit, interpolation)
	meanPitch = call(pitch, "Get mean", 0, 0, unit)
	sdPitch = call(pitch, "Get standard deviation", 0, 0, unit)
	return minPitch, maxPitch, meanPitch, sdPitch

def extractIntensity(sound, minPitch, timeStep,interpolation):
	intensity = call(sound, "To Intensity", minPitch, timeStep)
	minIntensity = call(intensity, "Get minimum", 0, 0, interpolation)
	maxIntensity = call(intensity, "Get maximum", 0, 0, interpolation)
	meanIntensity = call(intensity, "Get mean", 0, 0)
	sdIntensity = call(intensity, "Get standard deviation", 0, 0)
	return minIntensity, maxIntensity, meanIntensity, sdIntensity

def extractJitterAndShimmer(sound, pitchFloor, pitchCeiling):
	pitch = call(sound, "To Pitch", 0.0, pitchFloor, pitchCeiling)
	pointProcess = call(pitch, "To PointProcess")
	localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
	localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
	return localJitter, localShimmer

def extractHNR(sound):
	harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
	hnr = call(harmonicity, "Get mean", 0, 0)
	return hnr

def extractSpeakingRate(sound, transcriptionLength):
	totalDuration = call(sound, "Get total duration")
	speakingRate = transcriptionLength / totalDuration
	return speakingRate

def extract_parselmouth(text, f):
	feature = []

	sound = parselmouth.Sound(f)
	transcription_length = len(str(text).split())
	(minPitch, maxPitch, meanPitch, sdPitch) = extractPitch(sound, 75, 600, "Hertz", "Parabolic")
	(minIntensity, maxIntensity, meanIntensity, sdIntensity) = extractIntensity(sound, 75, 0.0, "Parabolic")
	(localJitter, localShimmer) = extractJitterAndShimmer(sound, 75, 600)
	hnr = extractHNR(sound)
	speakingRate = extractSpeakingRate(sound, transcription_length)

	feature += [minPitch,maxPitch,meanPitch,sdPitch,minIntensity,maxIntensity,meanIntensity,sdIntensity,localJitter,localShimmer,hnr,speakingRate]
	return feature

def extract_ngram(text):
	ngrams = []
	f = open(os.path.join('./', 'ngrams.txt'), 'r+')
	for line in f.readlines():
		line = line.strip()
		ngrams.append(line)
	bigrams = ngrams[0:24]
	unigrams = ngrams[24:]
	f.close()
	ngram_to_index = dict((u, i) for i, u in enumerate(ngrams))

	index_encoded = []
	tokens = nltk.word_tokenize(text)
	line_bigrams = nltk.ngrams(tokens, 2)
	for item in line_bigrams:
		bigram = ' '.join(item)
		if bigram in ngram_to_index:
			index_encoded.append(ngram_to_index[bigram])

	line_unigrams = nltk.ngrams(tokens, 1)
	for item in line_unigrams:
		unigram = ' '.join(item)
		if unigram in ngram_to_index:
			index_encoded.append(ngram_to_index[unigram])

	onehot = [0 for _ in range(len(ngrams))]
	for value in index_encoded:
		onehot[value] += 1

	return onehot

def extract_pos_tag(string):
	nltk.download('tagsets')
	nltk.download('averaged_perceptron_tagger')
	tagdict = load('help/tagsets/upenn_tagset.pickle')
	keyList = []
	for key in tagdict.keys():
		keyList.append(key)

	skeleton_dict = {key: 0 for key in keyList}
	ts = nltk.word_tokenize(string)
	td = nltk.pos_tag(ts)

	sdc = skeleton_dict.copy()
	for i in range(len(td)):
		sdc[td[i][1]] = sdc[td[i][1]]+1

	# return list(sdc.items())
	return [v for k, v in sdc.items()]

def classify(feature):
	model = pickle.load(open(os.path.join('./', 'finalized_model.sav'), 'rb'))
	predict_label = model.predict(feature)[0]
	return predict_label

def helper(text, f):
	mfcc_feature = extract_mfcc(f)
	parselmouth_feature = extract_parselmouth(text, f)
	ngram_feature = extract_ngram(text)
	pos_feature = extract_pos_tag(text)
	feature = mfcc_feature + parselmouth_feature + ngram_feature + pos_feature
	label = classify(np.array([feature]))
	return label

@app.route('/getFile',  methods=['GET', 'POST'])
def getFile():
	f = request.files.get('file')
	f.save(os.path.join('./', f.filename))
	f.stream.seek(0)
	translate_res = speechRecognition(f)
	result = helper(translate_res, os.path.join('./', f.filename))
	# os.remove(os.path.join('./', f.filename))
	return json.dumps({"transcription": translate_res, "result2":result, "result3":"hello"})

if __name__ == '__main__':
	app.run()
