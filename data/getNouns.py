nouns = set()

with open("all-sci-fi-justEvents.txt",'r') as eventFile:
	for line in eventFile.readlines():
		for event in line.split("|||"):
			for word in event.split(" "):
				if "Synset" in word:
					nouns.add(word)

nouns = list(nouns)
with open("scifi-synsets.txt","w") as nounFile:
	for noun in nouns:
		nounFile.write(noun+"\n")
