import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np

if len(sys.argv) <= 2:
	print ("usage: testRankSentence 3 data.txt")
	exit()
else:
	top_k = int(sys.argv[1])
	file = open(sys.argv[2],'r')
	text = file.read()

text = text.strip().replace('\r\n', ' ')

sentences = sent_tokenize(text)
#print sentences

stop_words = set(stopwords.words('english'))

def token_lower(sentence):
	return [word.lower() for word in word_tokenize(sentence) if (word not in stop_words and word.isalpha())]

tok_fil_sent = list(map(token_lower, sentences))
num_nodes = len(list(tok_fil_sent))

graph = np.zeros((num_nodes,num_nodes))
for i in range(num_nodes):
	for j in range(i+1,num_nodes):
		graph[i,j] = float(len(set(tok_fil_sent[i])&set(tok_fil_sent[j])))/(len(tok_fil_sent[i])+len(tok_fil_sent[j]))
		graph[j,i] = graph[i,j]
#print graph

node_weights = np.ones(num_nodes)


def text_rank_sent(graph,node_weights,d=.85,iter=20):
	weight_sum = np.sum(graph,axis=0)
	while iter >0:
		for i in range(len(node_weights)):
			temp = 0.0
			for j in range(len(node_weights)):
				temp += graph[i,j]*node_weights[j]/weight_sum[j]
			node_weights[i] = 1-d+(d*temp)
		iter-=1

text_rank_sent(graph,node_weights)


top_index = [i for i,j in sorted(enumerate(node_weights), key=lambda x: x[1],reverse=True)[:top_k]]

top_sentences = [sentences[i] for i in top_index]
print (top_sentences)