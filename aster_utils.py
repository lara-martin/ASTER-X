from collections import defaultdict
import json
from gensim.models import Word2Vec
import pickle

def isVerbNet(thing):
	if "-" in thing:
		pos = thing.index("-")
		if thing[pos+1].isdigit():
			return True
	return False

def swapLastParams(event):
	#print(event)
	temp = event[-1]
	event[-1] = event[-2]
	event[-2] = temp
	return event
	
def rep(word):
	return word.replace("<","").replace(">","").replace("Synset(","").replace(")", "").replace('"\'',"'").replace("'\"","'")

def getPrimaryFrame(frame):
	"""get the primary frame and its parts of speech"""
	primary_frame = frame["description"]["primary"]  # primary syntactic description
	POS_frame = [x.split(".")[0].split("_")[0].split("-")[0] for x in
				 primary_frame.split(" ")]  # list of words, minus the descriptions after the . or _
	POS_frame = list(filter(None, POS_frame))  # get rid of the empties
	# remove special words (how, that, what, to be, up, whether, together, out, down, apart, why, if, when, about)
	temp_POS = POS_frame
	POS_frame = []
	for p in temp_POS:
		if p.isupper():
			POS_frame.append(p)
	return primary_frame, POS_frame  # the primary_frame might not be necessary since it's the string before the descriptions are removed

def getSelectors(vnclass):
	# collect the selectional restrictions for this verb class
	# note the thematic roles in this class
	# fill self.extractedSels --> e.g. extractedSels["Agent"] = {"+animate|+concrete"}
	class_selectors = defaultdict(set)
	for theme in vnclass['themroles']:
		themeType = theme['themrole']
		sels = theme['selrestrs']  # there should only be one
		found = False
		if sels["selrestrs_list"]:  # logic
			selSet = set()
			for sel in sels["selrestrs_list"]:
				found = True
				if sel['value'] and sel['type']:
					selSet.add(sel['value'] + sel['type'])
			if sels["selrestr_logic"] == "&":
				class_selectors[themeType] |= selSet
			else:
				or_set = sels["selrestr_logic"].join(selSet)  # "|".join(selSet)
				class_selectors[themeType] |= set([or_set])
	return class_selectors


class DataHolder:
	def __init__(self, file_example=None, model_name=None):
		
		self.censusNames = pickle.load(open("../data/names-percentage.pkl", 'rb'))
		
		NER_file = [x.split(";") for x in open("../data/sci-fi_ner-lowercase-noPeriod-namechange.txt", 'r').readlines()]
		self.NER_dict = {k.strip(): ("<MISC>" if v.strip()=="O" else "<"+v.strip()+">") for k,v in NER_file} #entity: tag
		rev_NER_dict = defaultdict(set) #tag: set(entities)
		for key, value in self.NER_dict.items():
			rev_NER_dict[value].add(key.title())
		del(NER_file)
		self.rev_NER_dict = rev_NER_dict
		
		f = open('../data/transitive_verbs_only.txt', 'r', encoding='utf-8')
		self.transitive_verbs = [line.strip() for line in f.readlines()]
		f.close()
		f = open('../data/intransitive_verbs.txt', 'r', encoding='utf-8')
		self.intransitive_verbs = [line.strip() for line in f.readlines()]
		f.close()
		
		vn_file = json.load(open("../vn_3.4-edit.json",'r'))["VerbNet"]
		verbnet = {}
		for vclass in vn_file:
			verbnet[vclass["class_id"]] = vclass
		self.verbnet = verbnet
		
		if model_name is None and file_example is not None:
			filename = file_example.split('/')[-1].split('.')[0]
			sample = open(file_example, "r", encoding='utf-8')
			s = sample.read()
			f = s.replace("\n", " ")
			data = []
			words = []
			for i in sent_tokenize(f):
				temp = []
				for j in word_tokenize(i):
					temp.append(j.lower())
					words.append(j.lower())
				data.append(temp)
			self.model = Word2Vec(data, min_count=1, size=100, window=5)
			self.model.save("../data/word2vec_{}.model".format(filename))
			self.num_words = float(len(words))
			fdist = nltk.FreqDist(words)
			fdist['**TOTAL**'] = float(self.num_words)
			f = open('../data/{}_word_freq.json'.format(filename), 'w')
			json.dump(fdist, f)
			f.close()
			self.word_frequencies = fdist

		elif model_name is not None:
			self.model = Word2Vec.load('../data/word2vec_{}_sentences.model'.format(model_name))
			json_file = open('../data/{}_word_freq.json'.format(model_name), 'r')
			self.word_frequencies = json.load(json_file)
			json_file.close()
			self.num_words = float(self.word_frequencies['**TOTAL**'])
		else:
			self.model = None
			self.word_frequencies = None

