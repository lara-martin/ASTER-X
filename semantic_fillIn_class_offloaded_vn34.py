# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import sys
sys.path.insert(1, "../")
import argparse, os, copy, random, subprocess, pickle, re, json, math
from collections import defaultdict
import nltk.corpus
from en.verb import *
from lm import LM
from memoryGraph_scifi2 import MemoryGraph
from gensim.models import Word2Vec
from itertools import combinations
import numpy as np
import string
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from enum import Enum
from aster_utils import *


stop_words = set(stopwords.words('english'))
def remove_punctuation(word):
	for punct in string.punctuation:
		if punct != '_':
			word = word.replace(punct, '')
	word = re.sub("\d+", "", word)
	return word

def cosine(word_vec1, word_vec2):
	num = np.dot(word_vec1, word_vec2)
	dem = (np.linalg.norm(word_vec1)* np.linalg.norm(word_vec2))
	return num/dem


def pastTense(m_name, first, prev):
	if "-" in m_name: m_name = m_name.split("-")[0]
	if "_" in m_name:
		m_name = m_name.split("_")
		if not first: return " ".join(m_name)
		v = ""
		new_m_name = []
		for i, word in enumerate(m_name):
			if i == 0:
				try:
					if prev == "they" or "ORGANIZATION" in prev:
						v = verb_past(word, person=3)
					else:
						v = verb_past(word, person=1)
					if v:
						new_m_name.append(v)
					else:
						new_m_name.append(word)
				except KeyError:
					if word[-1] == "e":
						new_m_name.append(word + "d")
					else:
						new_m_name.append(word + "ed")
			else:
				new_m_name.append(word)
		return " ".join(new_m_name)
	else:
		if not first: return m_name
		try:
			if prev == "they": return verb_past(m_name, person=3)
			return verb_past(m_name, person=1)
		except KeyError:
			if m_name[-1] == "e":
				return m_name + "d"
			else:
				return m_name + "ed"
				
def get_word_name(word):
	if type(word) == Synset:
		word_name = word.name().split('.')[0]
	elif type(word) == dict:
		word_name = word["name"]
	else:
		word_name = word
	return word_name.lower()


def get_noun_candidates(parent):
	candidates = []
	hypo = lambda s: s.hyponyms()
	TREE_hypo = parent.tree(hypo, depth=2)
	for ix in range(1, len(TREE_hypo)):
		cans = [can[0] for can in TREE_hypo[ix][1:]]
		candidates.extend(cans)
	return candidates
	
	
def pickNoun(cat, prev):
	try:
		_,category,_ = cat.split("'")
		category = category.split(".")[0]
	except:
		return cat

	try:
		synset = wn.synsets(category)
		for syn in synset:
			if str(syn) == cat:				
				hypo = list(syn.hyponyms())
				break
		random.shuffle(hypo)
		_,selection,_ = str(hypo.pop()).split("'")
		return selection.split(".")[0].replace("_"," ")
	except:
		_,selection,_ = cat.split("'")
		return selection.split(".")[0].replace("_"," ")
		

class FillInMethods(Enum):
	WordVectors = 'word_vectors',
	RandomFill = 'random',


class FillIn:
	def __init__(self, models, verbose=False, method=FillInMethods.WordVectors, use_context=True, wc=0.4, context_num=1, remove_transitive=False, use_frequency=True, N=10):
		self.LM = LM()
		self.N = N
		self.method = method
		self.verbose = verbose
		self.context_vec = None
		self.wc = wc
		self.use_frequency = use_frequency
		self.use_context = use_context
		self.context_num = context_num
		self.context_update = context_num
		self.remove_transitive=remove_transitive
		self.trans_member_sum=0.0
		self.cnt=0.0
		self.words_in_context=[]
		
		self.transitive_verbs = models.transitive_verbs
		self.intransitive_verbs = models.intransitive_verbs
		self.model = models.model
		self.num_words = models.num_words
		self.word_frequencies = models.word_frequencies
		self.verbnet = models.verbnet



	def get_word_vector(self, word):
		if type(word) == Synset:
			wordname = remove_punctuation(word.name().split('.')[0])
		elif type(word) == dict:
			wordname = remove_punctuation(word["name"])
		else:
			word = word.lower()
			wordname = remove_punctuation(word)
		if '_' in wordname:
			words = [remove_punctuation(w) for w in wordname.split('_')]
			word_components = [self.model.wv[w] for w in words if w in self.model]
			if len(word_components) > 0:
				word_vect = np.mean(word_components, axis=0)
				return word_vect
			else:
				return None
		elif wordname not in self.model:
			return None
		else:
			word_vect = self.model.wv[wordname]
			return word_vect
			
	################################################
	##VERBS##

	def slot_fill_verbs(self, filledSent, inSent, event_nouns, new_nouns):
		final_filled_sent = []
		verb = ""
		first = True
		for i, category in enumerate(filledSent):
			if isVerbNet(category):
				if verb: first = False
				alpha, num = category.split("-", 1)
				picked = ""
				if isVerbNet(category):
					if i< len(filledSent)-1 and filledSent[i+1] != 'EmptyParameter':
						if self.verbose and self.remove_transitive:
							print('Has Direct Object so keep the Transitive Verbs')
						picked = self.pickVerb(category, final_filled_sent[-1], first, nouns=new_nouns, hasDO=True)
					else:
						if self.verbose and self.remove_transitive:
							print('Does NOT have Direct Object so remove the Transitive Verbs')
						picked = self.pickVerb(category, final_filled_sent[-1], first, nouns=new_nouns, hasDO=False)
					word = ""
					if picked:
						word = picked
					else:
						word = category
					if i == len(filledSent) - 1:
						if final_filled_sent[-1] == "they":
							final_filled_sent.append("were")
						else:
							final_filled_sent.append("was")
					final_filled_sent.append(word)
					verb = word
				else:
					final_filled_sent.append(category)
					verb = category
			else:
				final_filled_sent.append(category)
		return final_filled_sent, inSent, event_nouns
		

	def pickVerb(self, verb, prev, first=True, nouns=None, hasDO=False):
		vnclass = ""
		while True:
			try:
				vnclass = self.verbnet[verb]
			except KeyError:
				return pastTense(verb, first, prev)
			
			members = vnclass['members']
			#print("MEMBERS", members)
			
			if self.remove_transitive and not hasDO:
				non_members = []
				for mem in members:
					if not mem["name"] in self.transitive_verbs:
						non_members.append(mem)
				#members = [mem for mem in members if all([members["name"] != trans for trans in self.transitive_verbs])]
				if self.verbose:
					print('Removing Transitive {}'.format(verb))
					print('number of members: {}'.format(len(members)))
					print('number of non-transitive members: {}'.format(len(non_members)))
					self.cnt+=1
					self.trans_member_sum += float(len(non_members))
					print('average #non-transitive members {}'.format(self.trans_member_sum/self.cnt))
				if len(members) == 0:
					members = vnclass['members']
				else: members=non_members
			if members: break
			else:
				verb = verb.rsplit("-",1)[0]
			

		if nouns is None or len(nouns) == 0:
			member = ""
			while member == "":
				member = members.pop()
		elif self.method == FillInMethods.WordVectors:
			member = ""
			best_val = 0
			for noun in nouns:
				if self.use_frequency:
					temp, val = self.get_closest(noun, members)
				else:
					temp, val = self.get_closest_wo_frequency(noun, members)
				if val > best_val:
					best_val = val
					member = temp
				if type(temp) == dict:       
					if val > best_val:
						best_val = val
						member = temp['name']
				else:
					if self.verbose:
						print("{} is not in VerbNet".format(temp))
		else:
			member = ""
			while member == "":
				member = members.pop()

		if member == "":
			while member == "":
				member = members.pop()

		return pastTense(member["name"], first, prev)

	################################################
	
	def Nmax_elements(self, the_list):
		final_list = []
		for i in range(0, self.N):  
			best = ('', -100000)
			for j in range(len(the_list)):
				if self.verbose:
					print('best: {}'.format((get_word_name(best[0]), best[1])))                    
					print('the_list: {}'.format([get_word_name(n) for n, val in the_list]))
				if the_list[j][1] > best[1]:
					best = the_list[j]
					the_list.remove(best)
					final_list.append(best)
		return final_list 


	def get_closest_wo_frequency(self, word, candidates):
		word_vect = self.get_word_vector(word)
		word_name = get_word_name(word)
		best_can = candidates[0]
		best_val = -1
		for can in candidates:
			if type(can) == Synset:
				can_name = can.name().split('.')[0]
			elif type(can) == dict:
				can_name = can["name"]
			else:
				can_name = can
			can_vect = self.get_word_vector(can_name)
			if can_vect is not None and word_vect is not None:
				if self.use_context and self.context_vec is not None:
					context_part = self.wc*(cosine(word_vect, self.context_vec))
					val = context_part + (1-self.wc)*cosine(word_vect, can_vect)                    
				else:
					val = cosine(word_vect, can_vect)                
				if val > best_val and word_name != can_name:
					best_can = can
					best_val = val
		return best_can, best_val

	def get_closest(self, word, candidates):
		word_vect = self.get_word_vector(word)
		distances = []
		for can in candidates:
			can_vect = self.get_word_vector(can)
			if can_vect is not None and word_vect is not None:
				if self.use_context and self.context_vec is not None:
					context_part = self.wc*(cosine(word_vect, self.context_vec))
					val = context_part + (1-self.wc)*cosine(word_vect, can_vect)
				else:
					val = cosine(word_vect, can_vect)
				distances.append((can, val))
		distances.sort(key=lambda x: x[1])
		Ncandidates = distances[0:self.N]
		candidate_frequencies = []
		for can, dist in Ncandidates:
			can_name = get_word_name(can)
			if '_' in can_name:
				word_parts = [self.word_frequencies[word] for word in can_name.split('_') if word in self.word_frequencies.keys()]
				if len(word_parts) > 0:
					val = min(word_parts)#sum(word_parts)/len(word_parts)
					candidate_frequencies.append((can, val))
			elif can_name in self.word_frequencies:
				candidate_frequencies.append((can, self.word_frequencies[can_name]))
		
		if len(candidates) == 0:
			best_can = word
			best_val = 1
		elif len(candidate_frequencies) == 0:
			best_can, best_val = self.get_closest_wo_frequency(word, candidates)
		else:
			denominator = sum([num for can, num in candidate_frequencies])
			probs = [num/denominator for can, num in candidate_frequencies]
			chosen = np.random.choice(range(len(candidate_frequencies)), 1, p=probs)[0]
			best_can = candidate_frequencies[chosen][0]
			best_val = probs[chosen]
			if self.verbose:
				print('best_can: {}'.format(get_word_name(best_can)))
				print('best_prob: {}'.format(best_val))
		return best_can, best_val


	################################################
	##NOUNS##
	
	def slot_fill_noun(self, event):
		if self.verbose:
			print('in slot fill noun')
		nouns = []
		for ix, word in enumerate(event):
			if 'Synset' in word:
				word_split = word.split('.')
				name = word.split('.')[0][8:]
				name_ix = int(word.split('.')[-1][:-2])
				nouns.append((ix, wn.synsets(name)[name_ix - 1]))
		if self.verbose:
			print('nouns: {}'.format(nouns))
		if len(nouns) == 0:
			return event, [], []

		new_nouns = []
		if len(nouns) == 1 and self.method==FillInMethods.RandomFill:
			nn = pickNoun(str(nouns[0][1]), None)
			event[nouns[0][0]] = nn.split(".")[0]
			new_nouns.append(nn)
			return event, nouns, new_nouns
		elif len(nouns) == 1:
			candidates = [can.name().split(".")[0] for can in get_noun_candidates(nouns[0][1])]
			candidate_frequencies = [(self.word_frequencies[can]/self.num_words, can) for can in candidates if can in self.word_frequencies.keys()]
			if len(candidate_frequencies) == 0:
				best_can = pickNoun(str(nouns[0][1]), None).split(".")[0]
			else:
				denominator = sum([num for num, can in candidate_frequencies])
				chosen = np.random.choice(range(len(candidate_frequencies)), 1, p=[num/denominator for num, _ in candidate_frequencies])[0]
				best_can = candidate_frequencies[chosen][1]
			event[nouns[0][0]] = best_can
			new_nouns.append(best_can)
			return event, nouns, new_nouns
			
		while len(nouns) != 0:
			pairs = list(combinations(nouns, 2))
			if len(pairs) == 0:
				# Odd case just use the other already filled nouns as candidates
				candidates = get_noun_candidates(nouns[0][1])
				best_can = None
				best_val = -1
				for n in new_nouns:
					if self.use_frequency:
						can, val = self.get_closest(n, candidates)
					else:
						can, val = self.get_closest_wo_frequency(n, candidates)
					if val > best_val:
						best_can = (n, can)
						best_val = val
				if nouns[0][0] > len(event) and self.verbose:
					print('This is the event: {}'.format(event))
					print('this is the index we are trying: {}'.format(nouns[0][0]))
					print(best_can)
				if best_can is None:
					if self.verbose:
						print('Did not find proper filling. using frequencies')
					candidates = [can.name().split(".")[0] for can in get_noun_candidates(nouns[0][1])]
					candidate_frequencies = [(self.word_frequencies[can]/self.num_words, can) for can in candidates if can in self.word_frequencies.keys()]
					if len(candidate_frequencies) == 0:
						max_candidate = pickNoun(str(nouns[0][1]), None).split(".")[0]
					else:
						denominator = sum([num for num, can in candidate_frequencies])
						chosen = np.random.choice(range(len(candidate_frequencies)), 1, p=[num/denominator for num, _ in candidate_frequencies])[0]
						best_can = candidate_frequencies[chosen][1]
					event[nouns[0][0]] = best_can
					new_nouns.append(best_can)
					nouns.remove(nouns[0])
				else:
					best_can_name = best_can[1].name().split('.')[0]
					event[nouns[0][0]] = best_can_name
					new_nouns.append(best_can[1])
					nouns.remove(nouns[0])
			else:
				if self.verbose:
					print('slot_fill_noun general case')
				filling = pairs[0]
				filled_closeness = -1000000
				fill_with = (None, None)
				for parent1, parent2 in pairs:
					candi1 = get_noun_candidates(parent1[1])
					candi2 = get_noun_candidates(parent2[1])
					best_can = ''
					best_val = -1000000
					# will go over the candidates for synset 1 and find the closest candidate in synset 2
					# take that pair and fill in the event
					for can1 in candi1:
						if self.verbose:
							print('calling get closest for: {}'.format(get_word_name(can1)))
						if self.use_frequency:
							can, val = self.get_closest(can1, candi2)
						else:
							can, val = self.get_closest_wo_frequency(can1, candi2)                        
						if val > best_val:
							best_can = (can1, can)
							best_val = val
					if best_val > filled_closeness:
						filling = (parent1, parent2)
						fill_with = best_can
				parent1, parent2 = filling
				if fill_with[0] is not None and fill_with[1] is not None:
				  event[parent1[0]] = fill_with[0].name().split(".")[0]
				  event[parent2[0]] = fill_with[1].name().split(".")[0]
				  new_nouns.append(fill_with[0])
				  new_nouns.append(fill_with[1])
				  nouns.remove(parent1)
				  nouns.remove(parent2)
				else:
					if self.verbose:
						print("DID NOT FIND SUITABLE FILLING PAIR HERE")
					for parent in filling:
						candidates = [can.name().split(".")[0] for can in get_noun_candidates(parent[1])]
						candidate_frequencies = [(self.word_frequencies[can]/self.num_words, can) for can in candidates if can in self.word_frequencies.keys()]
						if len(candidate_frequencies) == 0:
							max_candidate = pickNoun(str(parent[1]), None).split(".")[0]
						else:
							denominator = sum([num for num, can in candidate_frequencies])
							num, max_candidate = max([(num/denominator, can) for num, can in candidate_frequencies], key=lambda x: x[0])
						event[parent[0]] = max_candidate
						new_nouns.append(max_candidate)
						nouns.remove(parent)
		return event, nouns, new_nouns
		
	################################################
	#main function to call
	
	def fillEvent(self, emptyEvent, memory):		
		event_nouns = defaultdict(str)
		filledSent = []
		verb = ""
		# last two parameters already swapped
		new_nouns = []
		inSent = set()
		first = True
		nouns = [i for i, category in enumerate(emptyEvent) if "Synset(" in category and ".n." in category]
		verbs = [i for i, category in enumerate(emptyEvent) if isVerbNet(category)]
		if self.verbose:
			print('emptyEvent: {}'.format(emptyEvent))
		for i, category in enumerate(emptyEvent):
			#print("WORD", category)
			already_exists = memory.checkTaginMemory(category)
			if already_exists:
				#print("ALREADY EXITS", category, "is", already_exists)
				filledSent.append(already_exists)
				event_nouns[already_exists] = category
			else:
				name = memory.getNameFromTag(category)				
				if name:
					#print("NAME", name)
					if "Synset" in category:
						category = "<PERSON>"
					filledSent.append(name)
					#event_nouns[name] = "<PERSON>"
				elif "Synset(" in category:				
					if ".n." in category:  # it's a noun synset
						#print("SYNSET", category)
						#word = memory.find_recent_mentioned_item(category)
						if self.method == FillInMethods.WordVectors:
							word = category
						elif self.method == FillInMethods.RandomFill:
							word = pickNoun(category, None)
							new_nouns.append(word)
							memory.NEnums[category] = word
					else:
						word = pickNoun(category, None)
					filledSent.append(word)
					event_nouns[word] = category					
				else:  # regular word or verb
					#print("OTHER")
					filledSent.append(category)
			#print("FILLED SENT", filledSent)

		if self.verbose:
			print('filledSent b4 nouns: {}'.format(filledSent))

		if len(nouns) > 0 and self.method == FillInMethods.WordVectors:
			output = self.slot_fill_noun(filledSent)
			filledSent = output[0]
			old_nouns = output[1]
			new_nouns = output[2]
			for i, noun in enumerate(old_nouns):
				if type(noun) == tuple:
					memory.NEnums[str(noun[1])] = new_nouns[i]
				else:
					memory.NEnums[noun] = new_nouns[i]
		if self.verbose:
			print('filledSent after nouns: {}'.format(filledSent))
			print('new nouns: {}'.format(new_nouns))
			
		#Pick verb
		filledSent, inSent, event_nouns = self.slot_fill_verbs(filledSent, inSent, event_nouns, new_nouns)
		if self.verbose:
			print('filledSent after verbs: {}'.format(filledSent))
		#self.memory.add_event(filledSent, emptyEvent, event_nouns)
		if self.use_context:
			if self.verbose:
				print('calculating context_vec')
			sent_vecs = [(get_word_name(word), self.get_word_vector(word)) for word in filledSent if self.get_word_vector(word) is not None and word.lower() not in stop_words]
			if len(sent_vecs) > 0:
				if self.context_update % self.context_num == 0:
					if self.verbose:
						print('{}%{} == 0'.format(self.context_update, self.context_num))
						print('word in context_vector: {}'.format([word for word, sentvect in sent_vecs]))
						print('words in context_vec {}'.format(len(sent_vecs)))
					self.context_vec = np.mean([sentvect for word, sentvect in sent_vecs], axis=0)
					self.words_in_context = sent_vecs
				elif self.context_vec is not None:
					self.words_in_context.extend(sent_vecs)
					if self.verbose:
						print('{}%{} != 0'.format(self.context_update, self.context_num))
						print('word in context_vector: {}'.format([word for word, sentvect in self.words_in_context]))
						print('words in context_vec {}'.format(len(self.words_in_context)))
					self.context_vec = np.mean([sentvect for word, sentvect in self.words_in_context], axis=0)
			self.context_update += 1
			
		filledSent = [word.replace('_', ' ') for word in filledSent]
		if self.verbose:
			print('filledSent: {}'.format(filledSent))
			
		return filledSent, memory
