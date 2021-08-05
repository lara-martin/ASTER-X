from collections import defaultdict
import pickle

nouns = defaultdict(int)
unique_probs = []

with open("all-sci-fi-justEvents.txt",'r') as eventFile:
	for line in eventFile.readlines():
		unique = defaultdict(int)
		for event in line.split("|||"):
			for word in event.split(" "):				
				if "Synset" in word:
					nouns[word]+=1
					unique[word]+=1
				elif "<" in word and word != "<PRP>":
					unique[word]+=1
					label,_ = word.split(">")
					nouns[label+">"]+=1
		unique_prob = float(len(unique.keys())) / float(sum(unique.values()))
		unique_probs.append(unique_prob)

labels = []
counts = []
unique_prob = sum(unique_probs)/float(len(unique_probs))
print(unique_prob)

for key, value in nouns.items():
	labels.append(key)
	counts.append(value)

with open("scifi-nounCount.pickle","wb") as nounFile:
	pickle.dump(labels, nounFile)
	pickle.dump(counts,nounFile)
	pickle.dump(unique_prob, nounFile)

