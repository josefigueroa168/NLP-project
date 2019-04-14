import sys
# from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
import pandas as pd


def step_one():
	f = open(sys.argv[2],'r')
	content = f.readlines()

	w = open('for_pos.txt','w')

	for line in content:
		if line.startswith('# ::snt '):
			w.write(line[8:])

		if line.startswith('('):
			j = line.find('/')
			x = line[j+2:]
			tmp = ''
			for i in x:
				if i == ' ':
					break
				if i.isalpha():
					tmp += i
			focuses.append(tmp)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def match_focus(line,index):
	normalized_words = []
	focus = focuses[index]
	lemmatizer = WordNetLemmatizer()

	for word in line:
		l = lemmatizer.lemmatize(word, get_wordnet_pos(word))
		normalized_words.append(l.lower())

	# focus_index = normalized_words.find(focus)
	if focus in normalized_words:
		focus_index = normalized_words.index(focus)
		# print (focus,normalized_words[focus_index])
	else:
		focus_index = -1
	# print (normalized_words)
	
	return normalized_words, focus_index



def step_two():
	f = open("for_pos.txt",'r')
	content = f.readlines()

	index  = 0

	for line in content:

		tokenized = word_tokenize(line)
		pos = pos_tag(tokenized)

		normalized_words, focus_index = match_focus(tokenized,index)
		
		assert(len(pos) == len(normalized_words))

		# word in sentence, normalized words, focus or not (0 or 1), index of word, pos, pos for index-1,
		# pos for index -2, pos for index +1, pos for index+2
		for i in range(len(pos)):
			# print (pos[i])
			tmp = []
			word_tag = pos[i]
			tmp.append(word_tag[0])
			tmp.append(normalized_words[i])
			if focus_index == i:
				tmp.append(1)
			else:
				tmp.append(0)

			tmp.append(i)
			tmp.append(word_tag[1])

			if i >= 1:
				tmp.append(pos[i-1][1])
			else:
				tmp.append('N/A')

			if i > 1:
				tmp.append(pos[i-2][1])
			else:
				tmp.append('N/A')

			if i + 1 < len(pos):
				tmp.append(pos[i+1][1])
			else:
				tmp.append('N/A')

			if i + 2 < len(pos):
				tmp.append(pos[i+2][1])
			else:
				tmp.append('N/A')

			data.append(tmp)
		
		index += 1
		# print (pos)

focuses = []
data = []	

step_one()
step_two()
df = pd.DataFrame(data,columns=['word','normalized_words','focus','index','POS','-1POS','-2POS','+1POS','+2POS'])
fname = sys.argv[2][:-3] + 'csv'
df.to_csv(fname,index = None, header=True)





