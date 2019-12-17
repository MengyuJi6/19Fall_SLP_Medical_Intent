import nltk
import csv
from tqdm import tqdm

def extract_bigrams():
	f_write = open('ngrams.txt', 'w+')

	with open('uniqueText.csv', 'r+') as f_read:
		reader = csv.reader(f_read)
		bigram_dict = dict()
		unigram_dict = dict()
		header = True

		for line in reader:
			if header:
				header = False
				continue

			tokens = nltk.word_tokenize(line[-2])
			all_bigrams = nltk.ngrams(tokens, 2)
			for item in all_bigrams:
				bigram = ' '.join(item)
				if bigram not in bigram_dict:
					bigram_dict[bigram] = 0
				bigram_dict[bigram] += 1

			all_unigrams = nltk.ngrams(tokens, 1)
			for item in all_unigrams:
				unigram = ' '.join(item)
				if unigram not in unigram_dict:
					unigram_dict[unigram] = 0
				unigram_dict[unigram] += 1

		bigrams = [k for (k,v) in bigram_dict.items() if v>20]
		print(len(bigrams))

		unigrams = [k for (k,v) in unigram_dict.items() if v>20]
		print(len(unigrams))

		for item in bigrams:
			f_write.writelines(item+'\n')
		for item in unigrams:
			f_write.writelines(item+'\n')

	f_write.close()
	return

def main():
	ngrams = []
	f = open('ngrams.txt', 'r+')
	for line in f.readlines():
		line = line.strip()
		ngrams.append(line)
	bigrams = ngrams[0:24]
	unigrams = ngrams[24:]
	f.close()

	f_write = open('all_ngram_features.csv', 'w+')
	with open('overview-of-recordings-processed.csv', 'r+') as f_read:
		ngram_to_index = dict((u, i) for i, u in enumerate(ngrams))

		# formatted_str = ''
		# # formatted_str += 'audio_no,utterance_index,subutterance_index,'
		# for n in ngrams:
		# 	formatted_str += n + ','
		# formatted_str += 'prompt\n'
		# f_write.writelines(formatted_str)

		reader = csv.reader(f_read)
		header = True

		for line in reader:
			if header:
				header = False
				continue

			index_encoded = []
			tokens = nltk.word_tokenize(line[-3])
			line_bigrams = nltk.ngrams(tokens, 2)
			for item in line_bigrams:
				bigram = ' '.join(item)
				if bigram in ngram_to_index:
					index_encoded.append(ngram_to_index[bigram])
				# else:
				# 	index_encoded.append(len(ngram_to_index))

			line_unigrams = nltk.ngrams(tokens, 1)
			for item in line_unigrams:
				unigram = ' '.join(item)
				if unigram in ngram_to_index:
					index_encoded.append(ngram_to_index[unigram])
				# else:


			onehot = [0 for _ in range(len(ngrams))]
			for value in index_encoded:
				onehot[value] += 1
			
			formatted_str = line[-4] + ','
			for i in onehot:
				formatted_str += str(i) + ','
			formatted_str += line[-2] + '\n'
			f_write.writelines(formatted_str)

	f_write.close()


if __name__ == '__main__':
	# extract_bigrams()
	main()

