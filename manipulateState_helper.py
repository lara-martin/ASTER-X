from copy import deepcopy
from collections import defaultdict
from aster_utils import *

# vehicle and vehicle_part were combined into vehicle in the json
# edited to make "+comestible" only refer to dead things & added +broken
selMutualExclusion = {
	"+abstract": ["+animal", "+animate", "-animate", "+biotic", "+body_part", "+broken", "+comestible", "+concrete", "+elongated",
				  "+force", "+garment", "+human", "+int_control", "+location", "-location", "+machine", "+nonrigid",
				  "+pointy", "+refl", "-region", "+solid", "-solid", "+substance", "+vehicle"],
	"+animal": ["+abstract", "+broken", "+comestible", "+communication", "+currency", "+eventive", "-location", "+machine", "+nonrigid",
				"+organization", "+pointy", "-solid", "+sound", "+substance", "+time", "+vehicle"],
	"+animate": ["+abstract", "-animate", "+broken", "+comestible", "+communication", "+currency", "+eventive", "-location", "+organization",
				 "+sound", "+time"],
	"-animate": ["+abstract", "+animate", "+communication", "+currency", "+eventive", "+organization", "+sound",
				 "+time"],
	"+biotic": ["+abstract", "+communication", "+currency", "+eventive", "+garment", "+machine", "+organization",
				"+sound", "+time", "+vehicle"],  # I'm assuming clothes can't be eaten
	"+body_part": ["+abstract", "+communication", "+currency", "+eventive", "+garment", "+machine", "+organization",
				   "+refl", "-solid", "+sound", "+substance", "+time", "+vehicle"],
				# assuming there aren't robotic arms
	"+broken": ["+abstract", "+animal", "+animate", "+concrete", "+currency", "+eventive", "+force", "+human", "+int_control", "+location", 
					"-location", "+machine", "+nonrigid", "+organization", "+refl", "-region", "-solid", "+sound", "+substance", "+time", "+vehicle"],
				# added in concrete so that it works
	"+comestible": ["+abstract", "+communication", "+currency", "+eventive", "+garment", "+machine", "+organization", "+refl", "+sound", "+time", 
					"+vehicle", "+human", "+animate", "+animal", "+int_control"],
	"+communication": ["+animal", "+animate", "-animate", "+biotic", "+body_part", "+comestible", "+concrete",
					   "+currency", "+elongated", "+force", "+garment", "+human", "+int_control", "+location",
					   "-location", "+machine", "+nonrigid", "+organization", "+pointy", "+refl", "-region", "+solid",
					   "-solid", "+substance", "+time", "+vehicle"],
	"+concrete": ["+abstract", "+broken", "+communication", "+eventive", "+sound", "+time"],
	"+currency": ["+animal", "+animate", "+biotic", "+body_part", "+broken", "+comestible", "+communication", "+eventive",
				  "+force", "+garment", "+human", "+int_control", "+location", "-location", "+machine", "+organization",
				  "+refl", "-region", "-solid", "+substance", "+time", "+vehicle"],
	"+elongated": ["+abstract", "+communication", "+eventive", "+organization", "-solid", "+sound", "+substance",
				   "+time"],
	"+eventive": ["+animal", "+animate", "-animate", "+biotic", "+body_part", "+broken", "+comestible", "+concrete", "+currency",
				  "+elongated", "+force", "+garment", "+human", "+int_control", "+location", "+machine", "+nonrigid",
				  "+organization", "+pointy", "-region", "+solid", "-solid", "+substance", "+vehicle"],
	"+force": ["+abstract", "-animate", "+broken", "+communication", "+eventive", "+garment", "+location", "-location",
			   "+nonrigid", "+organization", "-region", "+time"],
	"+garment": ["+abstract", "+animate", "+communication", "+currency", "+eventive", "+force", "+human",
				 "+int_control", "+machine", "+organization", "+pointy", "+refl", "-solid", "+sound", "+substance",
				 "+time", "+vehicle"],
	"+human": ["+abstract", "+body_part", "+broken", "+comestible", "+communication", "+currency", "+eventive", "+garment", "+machine",
			   "+organization", "+pointy", "-solid", "+sound", "+substance", "+time", "+vehicle"],
	"+int_control": ["+abstract", "-animate", "+broken", "+comestible", "communication", "+currency", "+eventive", "+garment", "+organization",
					 "+sound", "+time"],
	"+location": ["+abstract", "+broken", "+communication", "+currency", "+elongated", "+eventive", "+force", "-location", "+refl",
				  "-solid", "+sound", "+substance", "+time"],
	"-location": ["+abstract", "+animal", "+animate", "+biotic", "+body_part", "+broken", "+comestible", "+communication",
				  "+currency", "+elongated", "+force", "+garment", "+human", "+int_control", "+location", "+machine",
				  "+nonrigid", "+pointy", "+refl", "+solid", "-solid", "+sound", "+substance", "+time", "+vehicle"],
	"+machine": ["+abstract", "+animal", "+biotic", "+body_part", "+broken", "+comestible", "+communication", "+currency",
				 "+eventive", "+garment", "+human", "+organization", "-solid", "+sound", "+substance", "+time"],
	"+nonrigid": ["+abstract", "+broken", "+communication", "+eventive", "+organization", "+pointy", "+sound", "+time",
				  "+vehicle"],
	"+organization": ["+animal", "+animate", "-animate", "+biotic", "+body_part", "+broken", "+comestible", "+communication",
					  "+concrete", "+currency", "+elongated", "+eventive", "+force", "+garment", "+int_control",
					  "+machine", "+nonrigid", "+pointy", "+solid", "-solid", "+sound", "+substance", "+time",
					  "+vehicle"],
	"+plural": ["+substance", "+time"],
	"+pointy": ["+abstract", "+animal", "+communication", "+currency", "+eventive", "+garment", "+human", "+nonrigid",
				"+organization", "-solid", "+sound", "+substance", "+time", "+vehicle"],
	"+refl": ["+abstract", "+body_part", "+broken", "+comestible", "+communication", "+currency", "+eventive", "+garment",
			  "+location", "-location", "-region"],
	"-region": ["+abstract", "+broken", "+communication", "+currency", "+eventive", "+force", "+refl", "+sound", "+substance",
				"+time"],
	"+solid": ["+abstract", "+communication", "+eventive", "-location", "+organization", "-solid", "+sound",
			   "+substance"],
	"-solid": ["+abstract", "+animal", "+body_part","+broken", "+communication", "+currency", "+elongated", "+eventive",
			   "+garment", "+human", "-location", "+machine", "+organization", "+pointy", "+solid", "+sound", "+time",
			   "+vehicle"],  # liquids
	"+sound": ["+animal", "+animate", "-animate", "+biotic", "+body_part","+broken", "+comestible", "+concrete", "+currency",
			   "+elongated", "+garment", "+human", "+int_control", "+location", "-location", "+machine", "+nonrigid",
			   "+organization", "+pointy", "-region", "+solid", "-solid", "+substance", "+time", "+vehicle"],
	"+substance": ["+abstract", "+animal", "+body_part", "+broken", "+communication", "+currency", "+elongated", "+eventive",
				   "+garment", "+human", "+location", "-location", "+machine", "+organization", "+plural", "+pointy",
				   "+solid", "+sound", "+time", "+vehicle"],  # also liquids
	"+time": ["+animal", "+animate", "-animate", "+biotic","+broken", "+body_part", "+comestible", "+communication", "+concrete",
			  "+currency", "+elongated", "+force", "+garment", "+human", "+int_control", "+location", "-location",
			  "+machine", "+nonrigid", "+organization", "+pointy", "-region", "+solid", "-solid", "+sound",
			  "+substance", "+vehicle"],
	"+vehicle": ["+abstract", "+animal", "-animate", "+biotic", "+body_part", "+broken", "+comestible", "+communication",
				 "+currency", "+eventive", "+garment", "+human", "-location", "+nonrigid", "+organization", "-solid",
				 "+sound", "+substance", "+time"],
	"+question": ["+animal", "+animate", "-animate", "+biotic", "+body_part", "+broken", "+comestible", "+concrete",
					"+currency", "+elongated", "+eventive", "+force", "+garment", "+human", "+int_control", "+location",
					"-location", "+machine", "+nonrigid", "+organization", "+pointy", "+refl", "-region", "+solid", "-solid",
					"+substance", "+time", "+vehicle"]
}


def changePredicate(predicate):
	current_name = predicate.predicate
	if current_name == "disappear":
		predicate.predicate = "visible"
		predicate.negated = not predicate.negated
	elif current_name == "appear":
		predicate.predicate = "visible"
	elif current_name == "location":
		predicate.predicate = "has_location"
	elif current_name == "be":
		predicate.predicate = "exist"
	elif current_name == "suffocate":
		predicate.predicate = "alive"
		predicate.negated = not predicate.negated
	elif current_name == "suffocated":
		predicate.predicate = "alive"
		predicate.negated = not predicate.negated
	elif current_name == "property":
		predicate.predicate = "has_property"
	elif current_name == "neglect":
		predicate.predicate = "take_care_of"
		predicate.negated = not predicate.negated
	elif current_name == "confined":
		predicate.predicate = "free"
		predicate.negated = not predicate.negated
	elif current_name == "destroyed":
		predicate.predicate = "function"
		predicate.negated = not predicate.negated
	elif current_name == "degradation_material_integrity":
		predicate.predicate = "function"
		predicate.negated = not predicate.negated
	elif current_name == "linger":
		predicate.predicate = "delay"
	elif current_name == "harmed":
		predicate.predicate = "healthy"
		predicate.negated = not predicate.negated
	return predicate




def actorOnlyAllowedOne(predicate_name):
	# the predicate.subject should only be allowed to have one of these
	return predicate_name in ["has_location", "capacity", "has_configuration", "has_orientation",
							  "has_position", "has_state", "has_val"]


def patientOnlyAllowedOne(predicate_name):
	# in all of the subjects, only one should have this for each patient
	return predicate_name in ["has_possession"]


def transitivePreds(predicate):
	if predicate.predicate not in ["together", "social_interaction", "correlated", "contact", "conflict",
								   "attached", "cooperate", "different", "group", "mingled", "relate"]:
		return None
	newPred = deepcopy(predicate)
	newPred.roles_to_fill = [predicate.subject]
	newPred.subject = predicate.roles_to_fill[0]
	return newPred

"""
def predDoesNotAgreeWithSelRes(predicate, sel):
	#TODO: finish this function
	exclusions = {
		"visible": ["+sound", "+eventive", "+time", "+abstract"],

	}

	if sel in exclusions[predicate]:
		return True
	return False
	#rush(agent,theme)
"""
"""
def defaultPredicates(selrestr):
	defaults = {
		"+human": ["alive", "visible", "healthy", "free", "exist"],
		"+concrete": ["exist", "visible"]		
	}
	if selrestr in defaults: return defaults[selrestr]
	else: return None
"""
	
def predLeadsToSel(pred, negated):
	#add new selrestrs after certain predicates are activated
	if pred == "alive" and negated:
		return "+comestible"
	if pred == "function" and not negated:
		return "+concrete"
	if pred == "function" and negated:
		return "+broken"
	return None
	
def replaceSels(newSel, oldSels):
	newSels = set()
	newSels.add(newSel)
	for old in oldSels:
		if old in selMutualExclusion[newSel]:
			continue
		else:
			newSels.add(old)
	return newSels
	
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	
def shouldRemovePred(predicate_name):
	# remove these predicates because they are redundant or are verb-specific
	return predicate_name in ["cause", "co-temporal", "do", "body_motion", 
							  "intrinsic_motion", "motion", "rotational_motion", "overlaps", "adv", "direction",
							  "in_reaction_to", "apply_heat", "meets", "repeated_sequence", "transfer", "transfer_info",
								"change_value", "fictive_motion", "continue"]
							  #convert, irrealis

def shouldBeReplaced(verb):
	x = ["continue-55.3", "begin-55.1", "begin-55.1-1", "complete-55.2-1"]
	if verb in x: return True
	return False
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class State:
	# holds current set of conditions & selectional restrictions
	def __init__(self):
		self.conditions = defaultdict(set)  # set of tuples that make up predicate
		# conditions[entity] = set((True/False, predicate, param1, param2, ...))
		# first element of tuple is whether or not it is negated
		self.selRestrictions = defaultdict(set)  # sels[entity] = set(sels)

	def update(self, conds, sels):
		self.conditions = conds
		self.selRestrictions = sels

	def returnDictionary(self):
		final_dict = defaultdict(set)
		for key in self.conditions.keys():
			if key not in self.selRestrictions or self.selRestrictions[key] == None:
				final_dict[key] = self.conditions[key]
			else:
				final_dict[key] = self.conditions[key] | self.selRestrictions[key]
		theRest = self.selRestrictions.keys() - self.conditions.keys()
		for key in theRest:
			final_dict[key] = self.selRestrictions[key]
		return final_dict


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Predicate:
	def __init__(self, predicate, subject, roles, eventType, negated=False):
		self.time = eventType
		self.negated = negated

		#y = "!" if self.negated else ""

		# print("ORIGINAL PRED", self.time, y, predicate_string)
		#predicate, roles_to_fill = predicate_string[:-1].split("(", 1)
		#roles_to_fill = [x.strip() for x in roles_to_fill.split(",")]
		self.subject = subject #roles_to_fill.pop(0)
		self.roles_to_fill = roles #roles_to_fill
		self.predicate = predicate  # +"("+",".join(roles_to_fill)+")"
		# print("SUBJECT",self.subject)
		# print("FULL PREDICATE",predicate+"("+",".join(roles_to_fill)+")")
		# print("TIME",self.time)
		# print("NEGATED",self.negated)
	
	def isSameAs(self, pred2):
		if self.time == pred2.time and self.negated == pred2.negated and self.subject == pred2.subject and self.roles_to_fill == pred2.roles_to_fill and self.predicate == pred2.predicate:
			return True
		return False


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def checkSel_SubCall(source_ls, role):
	#  checks to see that the the new restrictions in role work with current source_ls restrictions
	#  or if they don't match make sure there are no mutual exclusions
	finalRoles = set()

	if not source_ls:  # if the old is empty, it passes automatically
		return True, role
	if not role:
		return True, source_ls

	if "" in source_ls:
		source_ls.remove("")
	if "" in role:
		role.remove("")
	# print("OLD",source_ls)
	# print("NEW",role)

	if not source_ls:  # if the old is empty, it passes automatically
		return True, role
	if not role:
		return True, source_ls

	for new_role in role:
		if new_role in source_ls:
			finalRoles = deepcopy(source_ls)
			continue
		if "|" in new_role:
			sel1, sel2 = new_role.split("|")
			for old_role in source_ls:
				if sel1 in old_role:  # a match
					finalRoles.add(sel1)
				if sel2 in old_role:  # a match
					finalRoles.add(sel2)
				if "|" in old_role:
					o1, o2 = old_role.split("|")
					if sel1 in selMutualExclusion[o1] and sel1 in selMutualExclusion[o2] and sel2 in selMutualExclusion[
						o1] and sel2 in selMutualExclusion[o2]:  # not good
						return False, new_role
					if sel1 not in selMutualExclusion[o1]:  # sel1 passes
						finalRoles.add(sel1)
						finalRoles.add(o1)
					elif sel1 not in selMutualExclusion[o2]:  # sel1 passes
						finalRoles.add(sel1)
						finalRoles.add(o2)
					elif sel2 not in selMutualExclusion[o1]:  # sel2 passes
						finalRoles.add(sel2)
						finalRoles.add(o1)
					elif sel2 not in selMutualExclusion[o2]:  # sel2 passes
						finalRoles.add(sel2)
						finalRoles.add(o2)
				# else:
				else:
					# print(old_role)
					if sel1 in selMutualExclusion[old_role] and sel2 in selMutualExclusion[old_role]:  # not good
						return False, new_role
					if sel1 not in selMutualExclusion[old_role]:  # sel1 passes
						finalRoles.add(sel1)
						finalRoles.add(old_role)
					elif sel2 not in selMutualExclusion[old_role]:  # sel2 passes
						finalRoles.add(sel2)
						finalRoles.add(old_role)

		else:
			for old_role in source_ls:
				if "|" in old_role:
					o1, o2 = old_role.split("|")
					if new_role in selMutualExclusion[o1] and new_role in selMutualExclusion[o2]:  # not good
						return False, new_role
					if new_role not in selMutualExclusion[o1]:  # new_role passes
						finalRoles.add(new_role)
						finalRoles.add(o1)
					elif new_role not in selMutualExclusion[o2]:  # new_role passes
						finalRoles.add(new_role)
						finalRoles.add(o2)
				else:
					if new_role in old_role:  # a match
						finalRoles.add(new_role)
					elif new_role in selMutualExclusion[old_role]:  # not good
						return False, new_role
					elif new_role not in selMutualExclusion[old_role]:  # sel1 passes
						finalRoles.add(new_role)
						finalRoles.add(old_role)

	# print("RETURNED ROLES",finalRoles)
	return True, finalRoles


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PredicateManipulator:
	def __init__(self, predicates, unchangedState, newSels):
		self.prevConds = unchangedState  # state.conditions
		# conditions[entity][predicate] = set((True/False, param1, param2, ...))
		self.newSels = newSels
		self.predicates = predicates
		self.pre_conditions = set()
		self.post_conditions = set()

		if len(self.predicates) == 1:
			pred = self.predicates[0]
			if all([True if x.startswith("?") else False for x in pred.roles_to_fill]) and pred.roles_to_fill:
				x = 0
			# print("No predicates")
			else:
				pred.time = "post"
				pred = changePredicate(pred)
				self.post_conditions.add(pred)
		else:
			for pred in self.predicates:
				# if all of the roles are blank, don't add it
				if all([True if x.startswith("?") else False for x in pred.roles_to_fill]) and pred.roles_to_fill:
					continue

				pred = changePredicate(pred)
				# print("# # # # ",pred.predicate)

				if pred.time == "pre":
					self.pre_conditions.add(pred)
				elif pred.time == "post":
					self.post_conditions.add(pred)

				newPred = transitivePreds(pred)
				if newPred:
					if newPred.time == "pre":
						self.pre_conditions.add(newPred)
					elif newPred.time == "post":
						self.post_conditions.add(newPred)

		self.rememberedFilledRoles = defaultdict(str)
		
	def updateSelsAfterPreds(self):
		#for each post-condition
		for post_cond in self.post_conditions:
			#check to see if it needs to be updated
			newSel = predLeadsToSel(post_cond.predicate, post_cond.negated)
			if newSel:
				#get rid of exclusions
				self.newSels[post_cond.subject] = replaceSels(newSel, self.newSels[post_cond.subject])

	def tryFillingBlankRoles(self, existingEntity, placeholderEntity):
		# we want to check to see if this placeholder is already in our story
		# check sels
		placeholderSels = self.newSels[placeholderEntity]
		existingSels = self.newSels[existingEntity]
		if not placeholderSels or not existingSels:
			match = True
			newRoles = existingSels | placeholderSels
		else:
			match, newRoles = checkSel_SubCall(existingSels, placeholderSels)
		# print("MATCHING SELS",match,newRoles)

		if match:
			self.rememberedFilledRoles[placeholderEntity] = existingEntity
			self.newSels[existingEntity] = newRoles

		# send back to check predicates
		return match

	def checkPredicates(self):
		if self.prevConds is None:
			return True, "No previous state. Automatic pass."
		else:
			# oldpred = (negation, predicate, arg1, arg2, ...)
			for pre in self.pre_conditions:
				#print("PRE-CONDITION: ",pre.predicate)
				# check to see that the pre-conditions are satisfied
				# if it's a tag, replace it

				if patientOnlyAllowedOne(pre.predicate):
					for subject, preds in self.prevConds.items():
						for old_pred in preds:
							# print("OLD SUBJECT",subject)
							# print("OLD PREDICATES",old_pred)
							if not pre.negated and not old_pred[0]:  # they're both positive statements
								if old_pred[1] == pre.predicate and old_pred[2:] == tuple(pre.roles_to_fill) and subject != pre.subject:
									# it's the same object and predicate but not the same subject (someone else has
									# this!) unless it's a placeholder "?"
									if pre.subject.startswith("?"):
										# if pre.subject in self.rememberedFilledRoles:
										# check everything else to see if it matches the old
										match = self.tryFillingBlankRoles(subject, pre.subject)
										if match:
											# print(pre.subject,"is now",subject)
											# print(old_pred)
											pre.subject = subject
										else:
											return False, subject+" already "+pre.predicate+" of "+str(pre.roles_to_fill)+". "+pre.subject+" cannot."
									else:
										return False, subject+" already "+pre.predicate+" of "+str(pre.roles_to_fill)+". "+pre.subject+" cannot."

				elif pre.subject in self.prevConds:
					# we've seen this subject in the last state
					for old_pred in self.prevConds[pre.subject]:
						if old_pred[1] == pre.predicate:  # the predicate name matches
							if old_pred[2:] == tuple(pre.roles_to_fill) and pre.negated != old_pred[0]:
								# the arguments match and their negation is opposite
								n = ""
								if old_pred[0] == True:
									n = "not "
								return False, pre.subject+" is "+n+pre.predicate+" of "+str(old_pred[2:])+", but the opposite was expected."
							elif actorOnlyAllowedOne(pre.predicate):
								# the subject can only have one of these, so it better match! (using location as
								# example)
								if pre.negated == True:  # we just can't have it in this location
									if old_pred[2:] == tuple(pre.roles_to_fill) and old_pred[0] == False:
										return False, pre.subject+" is "+pre.predicate+" of "+str(pre.roles_to_fill)+". This new event expects it to not be."
								if pre.negated == False:  # it needs to be this location
									if (old_pred[0] == True and old_pred[2:] == tuple(pre.roles_to_fill)):
										# if it's the right location but negative
										return False, pre.subject+" is not "+pre.predicate+" of "+str(pre.roles_to_fill)+" but needs to be."
									if (not old_pred[0] and old_pred[2:] != tuple(pre.roles_to_fill)):
										# the wrong location and positive
										return False, pre.subject+" "+pre.predicate+" of "+str(old_pred[2:])+", not "+str(pre.roles_to_fill)
			return True, None
			
	def updatePost(self, prevConds):
		newConds = deepcopy(prevConds)
		for post in self.post_conditions:
			#print("POST-CONDITION: ",post.predicate)
			# replace blank roles if they were filled
			if post.subject in self.rememberedFilledRoles:
				post.subject = self.rememberedFilledRoles[post.subject]
			#elif post.subject.startswith("?"):
			#	continue
			roles = []
			for role in post.roles_to_fill:
				if role in self.rememberedFilledRoles:
					roles.append(self.rememberedFilledRoles[role])
				else:
					roles.append(role)

			pred_list = (post.negated, post.predicate) + tuple(roles)

			# only allowed one
			# remove all of the old predicates of this type for this subject
			if actorOnlyAllowedOne(post.predicate):
				for old_pred in self.prevConds[post.subject]:
					if old_pred[1] == post.predicate and old_pred in newConds[post.subject]:
						newConds[post.subject].remove(old_pred)

			# check all of the subjects to make sure that nobody else has this
			if patientOnlyAllowedOne(post.predicate):
				for subject, old_preds in self.prevConds.items():
					for old_pred in old_preds:
						if post.predicate == old_pred[1] and tuple(post.roles_to_fill) == old_pred[2:] and old_pred in \
								newConds[subject]:
							# same predicate and same arguments
							newConds[subject].remove(old_pred)

			# it's now negated, get rid of it
			opposite_pred_list = (not post.negated, post.predicate) + tuple(roles)
			"""
			if post.negated == True and opposite_pred_list in newConds[post.subject]:
				newConds[post.subject].remove(opposite_pred_list)
				newConds[post.subject].add(pred_list)
			"""
			if opposite_pred_list in newConds[post.subject]:
				newConds[post.subject].remove(opposite_pred_list)
				newConds[post.subject].add(pred_list)
			else:
				newConds[post.subject].add(pred_list)
		#print("NEW CONDS",newConds)
		return newConds
				
	def updatePre(self):
		prevConds = self.prevConds
		# pre-conditions should already all be valid; add them to the state		
		for pre in self.pre_conditions:
			#print("PRE-CONDITION: ",pre.predicate)
			
			# replace blank roles if they were filled
			if pre.subject in self.rememberedFilledRoles:
				pre.subject = self.rememberedFilledRoles[pre.subject]
			roles = []
			for role in pre.roles_to_fill:
				if role in self.rememberedFilledRoles:
					roles.append(self.rememberedFilledRoles[role])
				else:
					roles.append(role)

			pred_list = (pre.negated, pre.predicate) + tuple(roles)
			prevConds[pre.subject].add(pred_list)
		#print("PREV CONDS",prevConds)
		return prevConds


	def updatePredicates(self):
		# updating the facts of previous state		
		prevConds = self.updatePre()
		return self.updatePost(prevConds)

