import sys
sys.path.insert(1, "../")
import pickle
from story_environment_neuro import *
from decode_redo_pipeline_top_p_multi import Decoder
import numpy as np
from data_utils import *
import argparse
from memoryGraph_scifi2 import MemoryGraph
import datetime
from semantic_fillIn_class_offloaded_vn34 import FillIn
from aster_utils import *
from BERT_fill_class import *

models = DataHolder(model_name="scifi")

parser = argparse.ArgumentParser()
parser.add_argument(
  "--config",
  help="path to json config",
  required=True
)
args = parser.parse_args()
config_filepath = args.config
config = read_config(config_filepath)

env = storyEnv(config['data']['verbose'])

h = None
c = None
word2id, id2word = read_vocab(config['data']['vocab_file'])
seq2seq_model = Decoder(config_path=config_filepath, top_n=config['data']['top_n'])
src_data = read_bucket_data(word2id, id2word,
  src = config['data']['test_src'],
  config = None
)

seeds = [x.split("|||")[0] for x in open("../data/bucketed_events_test_nameFix.txt", 'r').readlines()]
fillObject = FillIn(models, verbose=config['data']['verbose'], remove_transitive=True)

#test verbs
verbs = ["fill-9.8", "suspect-81", "keep-15.2", "throw-17.1"]

######################################

def cleanBERT(string):
    while "[UNK]" in string:
        string = string.replace("[UNK]","")
    while "# " in string:
        string = string.replace("# ","#")
    while " #" in string:
        string = string.replace(" #","#")
    while "#" in string:
        string = string.replace("#","")
    while " ," in string:
        string = string.replace(" ,",",")
    while " ." in string:
        string = string.replace(" .",".")
    while "  " in string:
        string = string.replace("  "," ")
    string = string.strip()
    return string

def read_vocab(file_path):
    vocab = [word for word in pickle.load(open(file_path, 'rb'))]
    word2id = {}
    id2word = {}
    for ind, word in enumerate(vocab):
        word2id[word] = ind
        id2word[ind] = word
    return word2id, id2word
 
def printPred(pred):
	x = ""
	if pred[0] == True:
		x = "not "
	print("<fact>"+x+rep(pred[1])+"("+rep(",".join(pred[2:]))+") </fact>")

def printState(state):
	print("<state>")
	print("<!--Current story world state-->")
	state_keys = list(state.keys())
	state_keys.sort()
	for entity in state_keys:
		if state[entity]:
			print("<entity>")
			print("<name>"+rep(entity)+"</name>")
			print("<facts>")
			for fact in state[entity]:
				if not type(fact) == str:
					printPred(fact)
				else:
					print("<fact>"+rep(fact)+"</fact>")
			print("</facts>")
			print("</entity>")
	print("</state>")

def prepToPrint(event, memory):
	event = swapLastParams(event)
	unfilled = rep(str(event))
	filled, memory = fillObject.fillEvent(event, memory)
	return unfilled, filled, memory
	
	
def getAction(config, env, results, memory, history):
	if config['model']['causal'] == True:
		pruned_candidates = env.validate(results, memory, history, models, config['model']['forced_frame'])
		#print(f"NUM AFTER PRUNED (MAX {config['data']['top_n']}): {len(pruned_candidates)}")
		print(f"<numValidCandidates> {len(pruned_candidates)} out of {len(results)} possible </numValidCandidates>")
		if len(pruned_candidates) == 0:
			print("</step>\n</story>")
			print("No more candidate events!")
			env.reset()
			return None, env
		action = pruned_candidates[0] #this is a (event, manipulateState object) tuple
		#TODO: should env come from pruned_candidates?
		next_state = env.step(action)
		printState(next_state)
		return action, env
	else:
		if config['model']["original_mode"] == True:
			pruned_candidates = env.onlyFillPronouns(results, memory, history)
		else:
			pruned_candidates = env.nonCausal_validate(results, memory, history)
		if len(pruned_candidates) == 0:
			print("</step>\n</story>")
			print("No more candidate events!")
			env.reset()
			return None, env
		print(f"<numValidCandidates> {len(pruned_candidates)} out of {len(results)} possible </numValidCandidates>")
		action = pruned_candidates[0]
		return action, env
		
		
def getSentence(filled_event, event, sentence_memory):
	#clean before BERT
	final_event = []
	for i, param in enumerate(filled_event):
		if i != 1 and isVerbNet(event[i]):
			final_event += ["to",param]
		elif "EmptyParameter" in param:
			continue
		else:
			final_event += [param]

	#E2S
	max_masks = 3
	sentence = BERT_fill(sentence_memory, final_event, max_masks)
	if sentence:
		sentence = cleanBERT(" ".join(sentence))
		sentence = sentence.strip()
		while "  " in sentence:
			sentence = sentence.replace("  "," ")
		print("SENTENCE",sentence)	
		return sentence
	else:
		return ""
	
######################################

print('<?xml version="1.0" encoding="UTF-8" ?>')
print(f"<!--{datetime.date.today()}-->")
print("<!--**Version Information**-->")
print(f"<!--CAUSAL: {config['model']['causal']}-->")
if config['model']['causal']:
	print(f"<!--FORCE FIND VERBNET FRAME: {config['model']['forced_frame']}-->")
	print(f"<!--VERB RESAMPLING (for DRL): {config['model']['forced_frame']}-->")
else:
	print(f"<!--PROPERLY FORMATED EVENTS ONLY: {not config['model']['original_mode']}-->")
print("<!--#########################-->")
print("<!--**About**-->")
print("<!--Log file for ASTER story generator system. Each story has a number of steps. In each step, the system goes through a set of candidate events, determining if each is valid and giving reasons why or why not it is. Out of the valid events, the system selects one.-->")


for j, event in enumerate(seeds):
	env.reset()
	event = event.split(" ")
	print("<story>")
	print("<!--A new story-->")
	memory = MemoryGraph(models)
	print("<step id=\""+str(0)+"\">")
	action, env = getAction(config, env, [event], memory, [])
	if not action:
		print("<error> Start event cannot be properly added to state </error>\n</story>")
		continue
	if type(action) == tuple:
		event = action[0]
	else:
		event = action
	print_event, filled_event, memory = prepToPrint(copy.deepcopy(event), memory)
	print("<startingEvent>\n<!--The user-given event to start the story-->\n"+print_event+ "</startingEvent>")
	print("<filledEvent>\n<!--An example of the event, randomly filled with real words-->\n"+str(filled_event)+"</filledEvent>")
	memory.add_event(event)
	history = [event]
	print("</step>")

	print_history = [filled_event]
	sentence = getSentence(filled_event, event, [])
	if not sentence:
		print("<error> Can't turn event "+str(filled_event)+" into a sentence. </error>\n</story>")
		continue
	sentence_memory = [sentence]



	#####Generate Events#####
	for i in range(0,5): #length of story
	#run through seq2seq/seq2seq2seq to get next distribution of events
		print("<step id=\""+str(i+1)+"\">")
		print("<!--Going through candidate events to find the next event in the story-->")
		results, h, c = seq2seq_model.pipeline_predict([event], h, c, start=True)
		#find a consistent one
		action, env = getAction(config, env, results, memory, history)
		if not action:
			print("</step>\n<final_story>"+str(print_history)+"</final_story>\n</story>")
			break
		if type(action) == tuple:
			event = action[0]
		else:
			event = action
		memory.add_event(event)
		history.append(event)
		print_event, filled_event, memory = prepToPrint(copy.deepcopy(event), memory)
		print("<selectedEvent>"+print_event+ "</selectedEvent>")
		print("<filledEvent>\n<!--An example of the event, randomly filled with real words-->\n"+str(filled_event)+"</filledEvent>")
		print_history.append(filled_event)
		print("<story_so_far>"+str(print_history)+"</story_so_far>")
		sentence = getSentence(filled_event, event, sentence_memory)
		if not sentence:
			print("<error> Can't turn event "+str(filled_event)+" into a sentence. </error>")
			break
		sentence_memory.append(sentence)

		print("</step>")
	print("<final_story>"+str(sentence_memory)+"</final_story>")
	print("</story>")


