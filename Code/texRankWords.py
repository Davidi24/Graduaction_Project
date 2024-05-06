import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import nltk
nltk.download('averaged_perceptron_tagger')

if len(sys.argv) <= 2:
	print ("usage: testRankWord 3 data.txt")
	exit()
else:
	k = int(sys.argv[1])
	file = open(sys.argv[2],'r')
	text = file.read()

text = text.strip().replace('\r\n', ' ')

tagged = pos_tag(word_tokenize(text))
#print tagged

stop_words = set(stopwords.words("english"))
#print stop_words
filter_set = set(["NN","NNP","NNS","NNPS","JJ","JJR","JJS"])

tag_filtered = [(word.lower(),tag) for (word,tag) in tagged if (word.lower() not in stop_words and tag not in filter_set and word.isalpha())]
#print tag_filtered	

words_filtered = [word for (word,tag) in tag_filtered]
#print words_filtered

graph = {}
node_weight = {}

def add_edge(graph,node_weight,edge1,edge2,weight=1):
	if edge1 in graph:
		if edge2 not in graph[edge1]:
			graph[edge1].append(edge2)
	if edge1 not in graph:
		graph[edge1] = [edge2]

	if edge2 in graph:
		if edge1 not in graph[edge2]:
			graph[edge2].append(edge1)
	if edge2 not in graph:
		graph[edge2] = [edge1]
	
	node_weight[edge1] = 1
	node_weight[edge2] = 1

#creating graph
span = 3 
i = 0
while i < len(words_filtered):
	j=i+1
	while j < i+span and j < len(words_filtered):
		add_edge(graph,node_weight,words_filtered[i],words_filtered[j]) 
		j+=1
	i+=1

#print graph
#print node_weight


def text_rank_word(graph,node_weight,d=.85,iterations=20):
	while iterations>0:
		for node in node_weight:
			temp=0.0
			for neighbour in graph[node]:
				temp+=node_weight[neighbour]/len(graph[neighbour])
			node_weight[node] = 1-d+(d*temp) 

		iterations-=1


text_rank_word(graph,node_weight,iterations=50)


def my_fun(tup):
	return tup[1]

top_k_words = sorted(node_weight.items(),key=my_fun,reverse=True)[:k]

print (top_k_words)