import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

def find_main_verb(i):
	# find PRP and VB
	main_verb = ''
	verb = []
	prp = []
	wrb = []
	www = ['WRB','WP','WDT']
	for j in range(len(pos[i])):
		if pos[i][j].startswith('V'):
			verb.append(j)
			continue
		if pos[i][j] == 'PRP':
			prp.append(j)
			continue

		if pos[i][j] in www:
			wrb.append(j)

	if len(verb)!=0:
		re = []
		for j in verb:
			for w in wrb:
				if j - w > 0 and j - w < 5:
					re.append(j)

		for j in re:
			verb.remove(j)

	if len(verb) != 0:
		m = verb[0]
	else:
		return '0'

	c = 10000
	for j in verb:
		for k in prp:
			if j - k < c and j - c > 0:
				m = j
				c = j-k

	main_verb = nword[i][m]
	if (main_verb == 'be' or main_verb == 'do') and verb.index(m) != len(verb)-1:
		m = verb[verb.index(m)+1]
		main_verb = nword[i][m]

	if (main_verb == 'have' or main_verb == 'has') and verb.index(m) != len(verb)-1:
		if m + 1 == verb[verb.index(m)+1]:
			# print ('----------------')
			main_verb = nword[i][m+1]
			m = m+1
	score[m]+=2
	if main_verb == 'be':
		score[m]-=1
	return main_verb

def find_quotation(i):
	indices = [j for j, x in enumerate(sent[i]) if x == "\""]
	
def dosomething(i,score):
	for j in range(len(nword[i])):
		if not nword[i][j].isalpha():
			score[j] -= 50

		if pos[i][j].startswith('V') and nword[i][j] != 'be':
			score[j] += 3

		if pos[i][j].startswith('N'):
			score[j] += 1

		if pos[i][j].startswith('J'):
			score[j] += 2
	return score

def prepro():

	first = True

	for i in range(len(index)):
		if index[i]== 0:
			if first:
				first = False
				focus_word.append(dfocus[i])
			else:
				assert(len(tmp_sent) == len(tmp_nword))
				assert(len(tmp_nword) == len(tmp_pos))
				assert(len(tmp_pred) == len(tmp_pos))

				sent.append(tmp_sent)
				nword.append(tmp_nword)
				pos.append(tmp_pos)	
				pred.append(tmp_pred)			
				focus_word.append(dfocus[i])

			tmp_sent = []
			tmp_nword = []
			tmp_pos = []
			tmp_pred = []

		tmp_pred.append(dpred[i])	
		tmp_sent.append(word[i])
		tmp_nword.append(nor_word[i])
		tmp_pos.append(dfpos[i])

	# print (sent)

# get a sentence back to one 
# output: 1. sentence + focus 2. 0 and 1
#
possible = ['can','could','might','may','perhaps','able']
cause = ['why','since','because','therefore','due','cause','reason','thus']
obligate = ['must','have','has']

mode = 1
result = []
sent = []
nword = []
pos = []
pred = []
flag = False
focus_word=[]
if (mode == 1):
	df = pd.read_csv('lr_out_little_prince.csv')
	df1 = pd.read_csv('data/amr-bank-struct-v1.6-test.csv')
	word = list(df['word'])
	nor_word = list(df['normalized_words'])
	index = list(df['index'])
	dfpos = list(df1['POS'])
	isfocus = list(df['isfocus'])
	dpred = list(df['prediction'])
	dfocus = list(df['focus'])
	assert(len(sent) == len(pred))	
	assert(len(focus_word) == len(sent))
	prepro()

	for i in range(len(sent)):
		flag = False
		score = [0] * len(sent[i])
		if sum(pred[i]) == 0:
			flag = True
		else:
			for j in range(len(pred[i])):
				if pred[i][j] == 'i':
					score[j] = 2

		if flag:
			pass
		mainw = find_main_verb(i)
		if mainw != '0':
			assert(sum(score) != 0)
		# print (mainw)

		if 'but' in nword[i] or 'however' in nword[i]:
			quo = [j for j, x in enumerate(sent[i]) if x == "\""]
			if 'but' in nword[i]:
				ibut = nword[i].index('but')
			else:
				ibut = nword[i].index('however')
			ff = False
			for j in quo:
				if j + 1 == ibut:
					ff = True
					break
			if not ff:
				result.append('contrast')
			continue

		

		if any(w in possible for w in nword[i]):
			result.append("possible")
			continue

		if 'and' in nword[i]:
			quo = [j for j, x in enumerate(sent[i]) if x == "\""]
			
			ibut = nword[i].index('and')
			
			ff = False
			for j in quo:
				if j + 1 == ibut:
					ff = True
					break
			if not ff:
				result.append('and')
			
			continue

		if any(w in cause for w in nword[i]):
			result.append('cause')
			continue

		

		if 'must' in nword[i] or (('have' in nword[i] or 'has' in nword[i]) and 'to' in nword[i]):
			ff = False
			if 'must' in nword[i]:
				result.append('obligate')
				continue
			else:

				ito = [j for j, x in enumerate(sent[i]) if x == "to"]
				ihave = [j for j, x in enumerate(sent[i]) if x == "have" or x == "has"]
				for j in ito:
					if any(j - k > 0 and j - k < 5 for k in ihave):
						result.append('obligate')
						ff = True
						break
				if ff:
					continue


		if 'like' in nword[i] and 'be' in nword[i]:
			result.append('resemble')
			continue

		if "should" in nword[i]:
			result.append('recommend')
			continue

		if sum(score) == 0 or mainw == 'be':
			score = dosomething(i,score)
		m = max(score)
		xx = score.index(m)
		result.append(nword[i][xx])


	assert(len(result) == len(sent))
	correct = 0
	for i in range(len(sent)):
		if result[i] == focus_word[i]:
			correct+=1
		else:	
			print (result[i],focus_word[i],nword[i])
			print ("\n")

	print(correct)

# 83



