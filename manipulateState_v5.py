#  -*- coding: utf-8 -*-
import sys
sys.path.insert(1, "../")
from copy import deepcopy
from collections import defaultdict
import json
from aster_utils import *
from manipulateState_helper import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ManipulateState:
	def __init__(self, state, models):
		""" Creating a new state manipulator"""
		self.verbnet = models.verbnet
		
		self.state = state  # old State object
		self.event = []

		# for frame selections
		self.frameSelected = None
		self.preps = []  # preps extracted from events
		self.numRolesToFill = 0

		self.roles = {}
		self.rev_roles = {}
		self.extractedSels = defaultdict(set)  # new sels extracted from VerbNet
		self.extractedPreds = None #defaultdict(set)  # predicate manipulator object, predicates extracted from VerbNet
		self.filledSels = None # extracted sels with entities filled
		self.newSels = None  # new sels actually ready to be used

	# # # # # # Selectional Restriction Checking and Updating # # # # # # #

	def checkSelrestrs(self):
		NER = {"<PERSON>": "+human",
			   "<LOCATION>": "+location",
			   "<ORGANIZATION>": "+organization",
			   "<DURATION>": "+time",
			   "<DATE>": "+time",
			   "<OBJECT>": "+machine",
			   "<VESSEL>": "+vehicle"}  # <MISC>, <NUMBER>
		updatedSels = defaultdict(set)
		prev = self.state.selRestrictions  # self.state.selRestrictions is old
		# prev[role] = selrestrs

		# self.extractedSels is new; extractedSels[role] = selrestrs
		
		
		# FOR EACH ENTITY NOT IN THE REV_ROLES KEYS
		extras = {self.event[0], self.event[3]}
		if not isVerbNet(self.event[2]): extras.add(self.event[2])
		extras -= self.rev_roles.keys()
		
		for entity in extras:
			if not entity in prev:
				# a new entity!
				#named entity
				if ">" in entity:
					tag, num = entity.split(">")
					tag += ">"
					if tag in NER:
						updatedSels[entity] = set([NER[tag]])
		

		# for each new role
		for entity in self.rev_roles.keys():
			#print("ENTITY:",entity)
			# pull out the restrictions that the role requires in this verb
			role = self.rev_roles[entity]
			role_sels = self.extractedSels[role]
			

			if entity in prev:
				# check against pre-existing selrestrs
				satisfied_restriction, tempSels = checkSel_SubCall(prev[entity], role_sels)
				if not satisfied_restriction:
					return False, (entity, tempSels)
				updatedSels[entity] = tempSels
			else:
				# a new entity!
				#named entity
				if ">" in entity:
					tag, num = entity.split(">")
					tag += ">"
					if tag in NER:
						satisfied_restriction, tempSels = checkSel_SubCall(set([NER[tag]]), role_sels)
						if not satisfied_restriction:
							return False, (entity, tempSels)
						updatedSels[entity] = tempSels
				#other
				elif not entity.startswith("?"):
					updatedSels[entity] = role_sels

		old_entities = prev.keys() - self.rev_roles.keys()
		for entity in list(old_entities):
			updatedSels[entity] = prev[entity]

		#self.filledSels = copy.deepcopy(updatedSels)
		self.newSels = updatedSels  # keep them in the object for now
		return True, None

	def getSelectors(self, vnclass):
		# collect the selectional restrictions for this verb class
		# note the thematic roles in this class
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
		self.extractedSels = class_selectors


	# # # # # # # # # Fill predicates from roles # # # # # # # # # # #

	def fillPredicates(self):
		# fill the predicates' roles with the items in the sentence
		# put everything in the right place in the predicates, including changing predicates to "core predicates"
		# make sure that this verb is legal
		prep_count = 0
		pred_objs = []
		frame = self.frameSelected
		roles = self.roles

		# print("FRAME", frame)
		# print("ROLES", roles)
		
		
		# go through the predicates and find any "irrealis" to get rid of predicates with it
		# (subevent didn't really happen)
		irrealis_event = ""
		for pred in frame['semantics']:
			for arg in pred['args']:
				if arg['arg_type'] == 'Event' and pred['predicate'] == "irrealis":
					irrealis_event = arg['value']
					#print(arg['value']+" IS NOT REAL.")
					break

		# go through the predicates and find any "equals" first so that those will be filled with the same thing
		for pred in frame['semantics']:
			if pred['predicate'] == "equals":
				same_entity = ""
				empty_value = ""
				for arg in pred['args']:  # for each argument in this predicate
					if arg['value'] in roles.keys():
						same_entity = roles[arg['value']]
					else:
						empty_value = arg['value']
				roles[empty_value] = same_entity
				break
				
		start_pred = True
		#print("FRAME ",frame['description']['primary'])
		for pred in frame['semantics']:  # go through each predicate in this frame
			pred_name = pred['predicate'].lower()
			#print("FRAME PREDICATE",pred_name)
			
			# add all the start "has_"s to pre-conditions
			if "has_" in pred_name and start_pred:
				eventType = "pre"
			else:
				eventType = "post"
				start_pred = False
			if shouldRemovePred(pred_name): continue
			negated = False
			subject = ""
			roles_to_fill = []
			if 'bool' in pred and pred['bool'] == "!":
				negated = True
			remove = False
			for arg in pred['args']:  # for each argument in this predicate
				if arg['arg_type'] == "Event":
					#don't use irrealis events
					if irrealis_event != "" and arg['value'] == irrealis_event:
						remove = True
						break
					if len(pred['args']) == 1:
						eventType = "post"
					elif arg['value'] == "e1" or arg['value'] == "E":
						eventType = "pre"
				elif arg['arg_type'] == "Constant":
					roles_to_fill.append(arg['value'])
				elif "?" in arg['value']:
					if subject == "":
						subject = arg['value']
					else:
						roles_to_fill.append(arg['value'])
				elif "prep" == arg['value'].lower() and self.preps:
					roles_to_fill.append(self.preps[0])
				else:
					found = False
					for role in roles.keys():
						# print("ROLE",role)
						if role:
							if role in arg['value']:
								if subject == "":
									subject =roles[role]
								else:
									roles_to_fill.append(roles[role])
								found = True
								break
					if not found and arg['value'] != "prep":
						if subject == "":
							subject = "?" + arg['value']
						else:
							roles_to_fill.append("?" + arg['value'])
			if remove or "?" in subject: continue
			#print("PREDNAME",pred_name)
			#print("SUBJECT",subject)
			#print("ROLESTOFILL",roles_to_fill)
			#print("EVENTTYPE", eventType)
			#print("NEGATED", negated)
			pred_obj = Predicate(pred_name, subject, roles_to_fill, eventType, negated)
			pred_objs.append(pred_obj)

		self.roles = roles
		# print("ROLE'EM",self.roles)
		rev_roles = {}
		for r in self.roles.keys():
			val = self.roles[r]
			rev_roles[val] = r
		self.rev_roles = rev_roles
		# print("REV'EM",self.rev_roles)

		return pred_objs

	# # # # # # # Frame Extraction # # # # # # #

	def findFrameHelper(self, frame, prep, forced=False):
		# check the verb's frame against the event and match up the semantic roles
		preps = []
		wrongPrep = False
		original_POS_string, POS_frame = getPrimaryFrame(frame)
		#print("Trying frame...", POS_frame)
		foundVerb = False
		roles = defaultdict(str)
		# if the sentence matches this frame

		if len(POS_frame) == self.numRolesToFill or forced:
			x = 0
			for pos in frame['syntax']:  # for each part of speech in the frame
				if wrongPrep and not forced:
					break
				if pos['arg_type'] != "VERB" and pos['arg_type'] != "ADV":  # pos.attrib: # not a VERB or ADV, given
					# a value for each part of the event
					for i in range(x, len(self.event) - 1):  # minus 1 to exclude prep
						param = self.event[i]
						param_pos = ""
						if "Synset(" in param or "<" in param:
							param_pos = "NP"
						elif isVerbNet(param):
							if foundVerb:
								param_pos = "S"
							else:
								foundVerb = True
								x+=1
						if pos['arg_type'] == param_pos:  # matching POS
							roles[pos['value']] = param
							x += 1
							break
						elif pos['arg_type'] == "PREP" or pos['arg_type'] == "LEX":
							if pos['value'] == "":
								preps += [prep]
								break
							elif pos['value'][0].islower():
								# if it's an actual list of prepositions
								if prep in pos['value']:  # if the preposition is in this space-delimited list
									preps += [prep]
								else:
									wrongPrep = True
							else:
								preps += [prep]
							break
						elif param == "EmptyParameter":
							if pos['value'] and not pos['value'] in roles:
								roles[pos['value']] = "?" + pos['value']
							x += 1
						else:  # it's not a word that's found in WordNet
							roles[pos['value']] = param
							x += 1
							break

				elif pos['arg_type'] == "PREP":  # no attributes
					preps += [prep]
				elif pos['arg_type'] == "VERB":  # a verb
					x += 1

			self.roles = roles
			if forced:
				return True, preps, "Forced frame"
			return not wrongPrep, preps, "Correct preposition: "+str(not wrongPrep)
		return False, preps, "Cannot find matching syntax for verb"

	def findFrame(self, event, forced=True):
		#print("--------------2) FINDING FRAME")
		# find the verbnet class in the json, extract frames and determine the appropriate frame (syntax)
		subToVN4 = {
			"appear-48.1.1": "escape-51.1",
			"chew-39.2-1": "chew-39.2",
			"eat-39.1-3": "eat-39.1-1-1",
			"own-100": "own-100.1"
		}
		# finds a frame and fills the predicates
		[agent, verb, patient, theme, prep] = event
		

		if not isVerbNet(verb): # it's not a VerbNet category
			# print("NOT IN VERBNET")
			return False, verb+" is not a VerbNet category"

		if isVerbNet(patient) and shouldBeReplaced(verb):
			verb = event[1] = patient
			patient = event[2] = "EmptyParameter"
			print("<reason>Overrode \"continuing\" verb</reason>\n<event> "+rep(str(event))+"</event>")
			
		self.event = event

		numRolesToFill = len(event) - event.count("EmptyParameter")
		if prep != "EmptyParameter":
			numRolesToFill -= 1
		self.numRolesToFill = numRolesToFill

		if verb in subToVN4:
			classID = subToVN4[verb]
		else:
			classID = verb
		alpha, num = classID.split("-", 1)
		layers = num.split("-")
		baseID = alpha + "-" + layers.pop(0)
		# print("BASEID",baseID)
		# print(layers)
		if baseID in self.verbnet:
			vnclass = self.verbnet[baseID]
		else:
			# couldn't find verb in verbnet
			return False, "Cannot find "+classID+" in VerbNet"

		# pick out selectional restrictions
		self.getSelectors(vnclass)

		# find the appropriate frame
		ancestors = []
		if vnclass['frames']:
			ancestors = [vnclass['frames']]

		currVerb = baseID

		# the actual verb was a subclass, we should check this first
		# add all of the ancestors
		# depth of subclass corresponds to the number of "-"s in the verb name
		tempclass = vnclass
		while classID != currVerb:
			# print(currVerb)
			if layers:
				currVerb += "-" + layers.pop(0)
				if tempclass['subclasses']:
					for c in tempclass['subclasses']:
						if c['class_id'] == currVerb:
							ancestors.append(c['frames'])
							tempclass = c
							break

		# it's the main class but it has no frames
		tempclass = vnclass
		if not ancestors:
			for c in tempclass['subclasses']:
				ancestors.append(c['frames'])
				tempclass = c

		frame = None
		for relative in ancestors:
			for frame in relative:
				found, preps, reason = self.findFrameHelper(frame, prep)
				if found:
					# print("Found frame.")
					self.frameSelected = frame
					self.preps = preps
					return found, reason

		# give up and return the last one we saw
		#print("Forced frame.",event)
		if forced:
			found, preps, reason = self.findFrameHelper(frame, prep, forced=forced)
			self.frameSelected = frame
			self.preps = preps
			return True, "Forcing the syntax; last option was taken"
		else:
			return False, "No matching syntax was found"

	# # # # # # # For running other methods and doing higher-level management# # # # # #
	def checkValidity(self):
		#TODO: <PRP> should be last entity, "this"/"that" should be last Synset
		# if self.state.conditions == None and self.state.selRestrictions == None: return True

		# print("--------------3) FILLING PREDICATES")
		pred_objs = self.fillPredicates()

		# print("--------------4) CHECKING SELS")
		selRestrsAreGood, selReason = self.checkSelrestrs()

		# print("--------------5) PRED OBJECT")
		self.extractedPreds = PredicateManipulator(pred_objs, self.state.conditions, self.newSels)
		if selRestrsAreGood:
			#print("ROLES", self.roles)
			if not self.roles:
				return True, self.extractedPreds
			# print("--------------6) CHECKING PREDICATES")
			predicatesAreGood, predReason = self.extractedPreds.checkPredicates()
			if predicatesAreGood:
				self.newSels = self.extractedPreds.newSels
				return True, self.extractedPreds
			else:
				return False, predReason
		else:
			return False, selReason[0] + " cannot have any of the following tags: "+str(selReason[1])

	def updateState(self):#, state):
		# print("--------------7) UPDATING STATE")
		newPreds = self.extractedPreds.updatePredicates()
		# print("NEW PREDS",newPreds)
		self.extractedPreds.updateSelsAfterPreds()
		return self.state.update(newPreds, self.newSels)

