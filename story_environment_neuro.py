import sys
sys.path.insert(1, "../")
import pickle
from collections import defaultdict
from manipulateState_v5 import *
from memoryGraph_scifi2 import MemoryGraph #TODO: generalized?
import random
from nltk import pos_tag
import copy
from aster_utils import *

def prepToPrint(event):
	return rep(str(swapLastParams(event)))

class storyEnv:
	def __init__(self, verbose=False, state=State()):
		self.state = state
		self.verbose=verbose

	   # print("Initialized new story environment!")

	"""Step fxn that takes in an action and advances the state
	   Contains code adapted from callManipulateState.py
	   Returns State, Reward, done, info
	"""

	def print_verbose(self,item):
		if self.verbose:
			print(item)

	######################################################
	""" Helper functions for checking various validation """
		
	def isValid(self, event, forced_frame, models):
		#now make sure it's valid
		#print("EVENT",event)
		manip = ManipulateState(self.state, models)
		found, reason1 = manip.findFrame(event, forced_frame)
		if not found:
			#print("<valid>false</valid>")
			#print("<reason>"+rep(reason1)+"</reason>")
			return None		
		else:
			valid, reason = manip.checkValidity()
			if valid:
				"""
				print("<valid>true</valid>")
				print("<reason>")
				print(f"<!--Pre- and post-conditions for valid event-->")
				print("<preConditions>")
				tempDict = defaultdict(list)

				for entity, sel in manip.newSels.items(): #was extracted sels
					if sel:
						tempDict[entity].append(sel)
						#self.print_verbose("\t"+entity+" is "+str(sel))
				for predicate in reason.pre_conditions:
					y = "not " if predicate.negated else ""
					tempDict[predicate.subject].append(y+predicate.predicate+"("+rep(",".join(predicate.roles_to_fill))+")")

				for key in tempDict.keys():
					print("<entity>")
					print("<name>"+rep(key)+"</name>")
					print(f"<!--Facts for entity {rep(key)}-->")
					print("<facts>")
					for item in tempDict[key]:
						print("<fact>"+str(item)+"</fact>")
					print("</facts>")
					print("</entity>")
				print("</preConditions>")
				print("<postConditions>")

				tempDict = defaultdict(list)
				for predicate in reason.post_conditions:
					y = "not " if predicate.negated else ""
					tempDict[predicate.subject].append(y+predicate.predicate+"("+rep(",".join(predicate.roles_to_fill))+")")
				
				for key in tempDict.keys():
					print("<entity>")
					print("<name>"+rep(key)+"</name>")
					print(f"<!--Facts for entity {rep(key)}-->")
					print("<facts>")
					for item in tempDict[key]:
						print("<fact>"+str(item)+"</fact>")
					print("</facts>")
					print("</entity>")
				print("</postConditions>")
				print("</reason>")	
				"""	
				return manip
			else:
				#print("<valid>false</valid>")
				#print("<reason>"+rep(reason)+"</reason>")
				return None
		return None
				
	def correctStructure(self, event):
		#check that the event is the right format
		if len(event) != 5: return False
		#make sure nothing is a verbnet class except the 2nd param (and sometimes the 3rd)
		if isVerbNet(event[0]) or isVerbNet(event[3]) or isVerbNet(event[4]):
			#print("<valid>false</valid>")
			#print("<reason> Incorrect structure: Verb found in wrong event parameter </reason>")
			return False
		if not isVerbNet(event[1]):
			#print("<valid>false</valid>")
			#print("<reason> Cannot check validity due to verb \""+ rep(event[1]) +"\" not being in VerbNet </reason>")
			return False
		#subject should not be empty
		if event[0] == "EmptyParameter":
			#print("<valid>false</valid>")
			#print("<reason> Incorrect structure: Event starts with \"EmptyParameter\" (no subject) </reason>")
			return False
		#if not empty, preposition should be preposition
		if event[4] != "EmptyParameter":
			if "Synset" in event[4] or "<" in event[4]:
				#print("<valid>false</valid>")
				#print("<reason> Incorrect structure: Invalid preposition \""+ rep(event[-1])+"\" </reason>")
				return False
		if event[3] != "EmptyParameter":
			if not "Synset" in event[3] and not "<" in event[3]:
				#print("<valid>false</valid>")
				#print("<reason> Incorrect structure: Invalid modifier \""+ rep(event[3])+"\" </reason>")
				return False
		if event[2] != "EmptyParameter":
			if not "Synset" in event[2] and not "<" in event[2] and not isVerbNet(event[2]):
				#print("<valid>false</valid>")
				#print("<reason> Incorrect structure: Invalid direct object \""+ rep(event[2])+"\" </reason>")
				return False
		if not "Synset" in event[0] and not "<" in event[0]:
			#print("<valid>false</valid>")
			#print("<reason> Incorrect structure: Invalid subject \""+ rep(event[0])+"\" </reason>")
			return False
		return True
			
			
	def fillPronouns(self, event, memory):
		#fill in pronouns, no duplicates
		for i, arg in enumerate(event):
			if arg == "<PRP>":
				mem_item = memory.find_recent_mentioned_item(event)
				if mem_item:
					event[i] = mem_item
				else:
					#print("<valid>false</valid>")
					#print("<reason> Not enough entities: Can't fill in all pronouns (PRPs) in event </reason>")
					return None
		return event
		
		
	def seenBefore(self, event, event_history):
		#similarity to previous event
		if event in event_history:
			#print("<valid>false</valid>")
			#print("<reason> Looping prevention: Event was seen in the story already </reason>")
			return True
		return False


	######################################################

	def validate(self, event_candidates, memory, event_history, models, forced_frame = False):
		"""
		Given a set of candidate events, filter out those which are invalid
		:param event_candidates: set of possible events for next action
		:return: list of valid candidate events
		"""
		valid_candidates = []
		#self.print_verbose("Searching for and validating next candidate events...")
		for i, event in enumerate(event_candidates):
			#print("<candidateEvent id=\""+str(i)+"\">")
			#print("<!--Checking validity of candidate event-->")
			#print("<event>"+rep(str(event))+ "</event>")
			#event is [subject, verb, direct object, indirect object, preposition]
			if not self.correctStructure(event):
				#print("</candidateEvent>")
				continue

			new_event = self.fillPronouns(event, memory)
			if not new_event:
				#print("</candidateEvent>")
				continue

			if self.seenBefore(new_event, event_history):
				#print("</candidateEvent>")
				continue

			#now make sure it's valid
			manip = self.isValid(new_event, forced_frame, models)
			if manip:						
				valid_candidates.append((new_event, manip))
			#print("</candidateEvent>")
		return valid_candidates
	

		
	######################################################

	def step(self, action):
		#it should've been guaranteed valid at this point
		(event, manip) = action
		manip.updateState()
		self.state = manip.state
		return self.state.returnDictionary()

	def reset(self):
		self.state = State()
		return self.state
		
	######################################################

	def nonCausal_validate(self, event_candidates, memory, event_history):
		"""
		Given a set of candidate events, filter out those which are invalid
		:param event_candidates: set of possible events for next action
		:return: list of valid candidate events without causality (correct structure, pronouns filled)
		"""
		valid_candidates = []
		for event in event_candidates:
			#event is [subject, verb, direct object, indirect object, preposition]
			if not self.correctStructure(event): continue
			new_event = self.fillPronouns(event, memory)
			if not new_event: continue
			if self.seenBefore(new_event, event_history): continue
			valid_candidates.append(new_event)
		return valid_candidates
		
		
	def onlyFillPronouns(self, event_candidates, memory, event_history):
		"""
		Given a set of candidate events, filter out those which are invalid
		:param event_candidates: set of possible events for next action
		:return: list of valid candidate events with only pronouns filled
		"""
		valid_candidates = []
		for event in event_candidates:
			#event is [subject, verb, direct object, indirect object, preposition]
			new_event = self.fillPronouns(event, memory)
			if not new_event: continue
			if self.seenBefore(new_event, event_history): continue
			valid_candidates.append(new_event)
		return valid_candidates
