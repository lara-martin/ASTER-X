# -*- coding: utf-8 -*-
#python3
import networkx as nx
from numpy import random
from nltk.corpus import wordnet as wn
from collections import defaultdict
from lm import LM
from aster_utils import *

lm = LM()

class MemoryGraph:
	def __init__(self, models):
		""" Creating a new event node that keeps track of the previous event and pointers to all things mentioned in this timestep
		"""
		self.censusNames = models.censusNames
		
		self.graph = nx.DiGraph() #keeps track of when things were mentioned
		self.graph.add_node("-")
		self.prev = -1
		
		self.NEnums = defaultdict(str) #keeps track of what tags correspond to
		self.rev_NER_dict = models.rev_NER_dict
		self.NER_dict = models.NER_dict
	
	

	def add_event(self, gen_event):
		# ADDING EVENT TO MEMORY
		verb = gen_event[1]
		#print("THINGS MENTIONED",thingsMentioned)
		self.prev+=1
		#create a new event node
		eventString = "E"+str(self.prev)
		newEvent = self.graph.add_node(eventString, verb=verb)
#		self.graph.add_edge(eventString, self.prev)
		#connect all recently-mentioned things to this node
		for i, category in enumerate(gen_event):
			if i == 1: continue #verb
			if i == len(gen_event)-1: continue #preposition
			if category == "EmptyParameter": continue
			if isVerbNet(category): continue
			
			weight = 0.4

			#print("add_event")
			if "<" in category and category != "<PRP>":
				weight = 1.0
				if not category in self.graph.nodes():
					self.graph.add_node(category)#, att=event_nouns[thing])
			else:
				if not category in self.graph.nodes():
					self.graph.add_node(category)#, att=event_nouns[thing])

			self.graph.add_edge(eventString, category, weight=weight)
			self.graph.add_edge(category, eventString, weight=weight)


	def find_recent_mentioned_item(self, already_mentioned):
		# FILLING PRONOUNS
		index = self.prev
		neighbors = list()	
		while True:
			if index < 0: return None
			neighbors = list(set(nx.all_neighbors(self.graph, "E"+str(index))))
			#print("NEIGHBORS",neighbors)
			random.shuffle(neighbors)			
			for n in neighbors:
				#print(already_mentioned, n)
				if n not in already_mentioned:
					return n
			index-=1 #go back a state and look there
		return None
		
	############ FOR FILLING ############
	def findFreeNameNumber(self, category="<PERSON>"):
		i = 0
		while True:
			if category+str(i) in self.NEnums:
				i+=1
			else:
				return i
		return False
		
	def checkIfName(self, word, orig_word=None):
		if "<" in word: return None
		if "Synset" in word:
			pos_name = word.split(".")[0].split("'")[1]
			pos_name = pos_name[0].upper()+pos_name[1:]
			if orig_word and pos_name.lower() == orig_word.lower():
				if pos_name in self.censusNames:
					newNum = self.findFreeNameNumber()
					category = "<PERSON>"+str(newNum)
					self.NEnums[category] = pos_name
					return pos_name
		return None

	def pickNE(self,category):
		name_select = random.choice(list(self.rev_NER_dict[category]))
		return name_select
		
	def checkTaginMemory(self, tag):
		#print("TAG", tag)
		#print("NE NUMS", self.NEnums)
		if tag in self.NEnums: return self.NEnums[tag]
		return None
		
	def getNameFromTag(self, tag):
		#print("GETNAMEFROMTAG", tag)
		if "<" in tag:
			category = tag.split(">")[0]+">"
			name = self.pickNE(category)
			self.NEnums[tag] = name
			return name
		elif "Synset" in tag:
			name = self.checkIfName(tag)
			if not name:
				return None
			#already put in NEnums
			return name
		else:
			return None
			
	
	

