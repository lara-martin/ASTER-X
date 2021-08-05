"""Data utilities."""
import torch
from torch.autograd import Variable
import operator
import json
import pickle
import numpy as np
import datetime



def read_config(file_path):
	"""Read JSON config."""
	json_object = json.load(open(file_path, 'r'))
	return json_object


def read_vocab(file_path):
	vocab = [word for word in pickle.load(open(file_path, 'rb'))]
	word2id = {}
	id2word = {}
	for ind, word in enumerate(vocab):
		word2id[word] = ind
		id2word[ind] = word

	return word2id, id2word


def read_unigram(file_path):
	return pickle.load(open(file_path, 'rb'))


def read_bucket_data(word2id, id2word, src, config):
	"""Read data from files."""
	src_lines = []
	with open(src, 'r') as f:
		for ind, line in enumerate(f):
			src_lines.append(line.strip())


	src = {'data': src_lines, 'word2id': word2id, 'id2word': id2word}
	del src_lines

	return src

#from https://www.geeksforgeeks.org/python-find-maximum-length-sub-list-in-a-nested-list/
def findMaxLength(lst):
    #maxList = max(lst, key = len)
    #print("MAX LENGTH",lst)
    maxLength = max(map(len, lst))
    return maxLength

def get_story_batch_helper(lines, line_index, t_index, end_token):

	batch = []
	for i in range(line_index,len(lines)+1):
		if end_token in lines[i]: break
		batch.append(lines[i].strip().split("|||"))


	max_story_length = findMaxLength(batch)
	if t_index == max_story_length-1:
		return None, None, i+1, 0
		#if t_index is the last one, line_index = i+1

	src_lines = []
	trg_lines = []
	for story in batch:
		if len(story)>t_index:
			src_lines.append(story[t_index].split())
		else:
			src_lines.append(["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"])
		if len(story)> t_index+1:
			trg_lines.append(story[t_index+1].split())
		else:
			trg_lines.append(["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"])

	return src_lines, trg_lines, line_index, t_index+1


def get_story_batch_fit_helper(lines, line_index, t_index, end_token, max_size, min_size):

	batch = []
	curr_size = 0
	for i in range(line_index,len(lines)):
		if end_token in lines[i]:
			if curr_size <= max_size and curr_size >= min_size:
				break
			continue
		if curr_size >= max_size:
			i-=1
			break
		batch.append(lines[i].strip().split("|||"))
		curr_size+=1

	max_story_length = findMaxLength(batch)
	if t_index == max_story_length-1:
		return None, None, i+1, 0
		#if t_index is the last one, line_index = i+1

	src_lines = []
	trg_lines = []
	for story in batch:
		if len(story)>t_index:
			src_lines.append(story[t_index].split())
		else:
			src_lines.append(["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"])
		if len(story)> t_index+1:
			trg_lines.append(story[t_index+1].split())
		else:
			trg_lines.append(["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"])

	#print(i, t_index)
	#print("SRC",src_lines)
	#print("TRG", trg_lines)
	return src_lines, trg_lines, line_index, t_index+1



def get_story_batch_fit_helper_bigram(lines, line_index, t_index, end_token, max_size, min_size):

	batch = []
	curr_size = 0
	for i in range(line_index,len(lines)):
		if end_token in lines[i]:
			if curr_size <= max_size and curr_size >= min_size:
				break
			continue
		if curr_size > max_size:
			i-=1
			break
		batch.append(lines[i].strip().split("|||"))
		curr_size+=1

	max_story_length = findMaxLength(batch)
	if t_index == max_story_length-2:
		return None, None, i+1, 0
		#if t_index is the last one, line_index = i+1

	src_lines = []
	trg_lines = []
	for story in batch:
		src = []
		if len(story)>t_index:
			src = story[t_index].split()
		else:
			src = ["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"]
		if len(story)> t_index+1:
			src += story[t_index+1].split()
		else:
			src+= ["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"]
		src_lines.append(src)
		if len(story)> t_index+2:
			trg_lines.append(story[t_index+2].split())
		else:
			trg_lines.append(["<pad>", "<pad>", "<pad>", "<pad>", "<pad>"])

	#print(i, t_index)
	#print("SRC",src_lines)
	#print("TRG", trg_lines)
	return src_lines, trg_lines, line_index, t_index+1


def getAllBatches(lines, end_token, max_size, min_size, hidden = True):
	j = 0
	t = 0
	count = 0
	batches = {}
	restart = True
	while j < len(lines):
		if t == 0 or hidden == False: restart = True
		else: restart = False
		src_lines, trg_lines, j, t = get_story_batch_fit_helper(lines, j, t, end_token, max_size, min_size)
		#print(src_lines, trg_lines)
		if src_lines !=None:
			batches[count] = {'restart': restart, 'src':src_lines, 'trg':trg_lines}
			count+=1
	return batches


def getAllBigramBatches(lines, end_token, max_size, min_size, hidden = True):
	j = 0
	t = 0
	count = 0
	batches = {}
	restart = True
	while j < len(lines):
		if t == 0 or hidden == False: restart = True
		else: restart = False
		src_lines, trg_lines, j, t = get_story_batch_fit_helper_bigram(lines, j, t, end_token, max_size, min_size)
		#print(src_lines, trg_lines)
		if src_lines !=None:
			batches[count] = {'restart': restart, 'src':src_lines, 'trg':trg_lines}
			count+=1
	return batches


def fillStartEnd(lines, add_start, add_end):
	if add_start and add_end:
		lines = [
			['<s>'] + line + ['</s>']
			for line in lines
		]
	elif add_start and not add_end:
		lines = [
			['<s>'] + line
			for line in lines
		]
	elif not add_start and add_end:
		lines = [
			line + ['</s>']
			for line in lines
		]
	elif not add_start and not add_end:
		lines = [
			line
			for line in lines
		]
	#print("PADDED",lines)
	return lines

def process_batch(
	src_lines, trg_lines, word2ind, add_start=True, add_end=True
):
	"""Find story at start_index until end_token"""
	#print("FIRST STORY",lines[line_index])

	#src_lines, trg_lines, line_index, t_index = get_story_batch_helper(lines, line_index, t_index, end_token)

	#print("SOURCE",src_lines)
	#print("TARGET",trg_lines)

	#if src_lines == None: return None, None, None, None, line_index, t_index

	src_lines = fillStartEnd(src_lines, add_start, add_end)
	trg_lines = fillStartEnd(trg_lines, add_start, add_end)

	#pad events, convert to indices, and create output lines for trg

	max_src_event_length = findMaxLength(src_lines)
	max_trg_event_length = findMaxLength(trg_lines)
	src_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]

	src_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]


	trg_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_trg_event_length - len(line))
		for line in trg_lines
	]

	trg_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
		[word2ind['<pad>']] * (max_trg_event_length - len(line))
		for line in trg_lines
	]

	src_input_lines = Variable(torch.LongTensor(src_input_lines)).cuda()
	trg_input_lines = Variable(torch.LongTensor(trg_input_lines)).cuda()
	src_output_lines = Variable(torch.LongTensor(src_output_lines)).cuda()
	trg_output_lines = Variable(torch.LongTensor(trg_output_lines)).cuda()

	return src_input_lines, src_output_lines, trg_input_lines, trg_output_lines


def get_story_batch(
	lines, word2ind, line_index, t_index, end_token, add_start=True, add_end=True
):
	"""Find story at start_index until end_token"""
	#print("FIRST STORY",lines[line_index])

	src_lines, trg_lines, line_index, t_index = get_story_batch_helper(lines, line_index, t_index, end_token)

	#print("SOURCE",src_lines)
	#print("TARGET",trg_lines)

	if src_lines == None: return None, None, None, None, line_index, t_index

	src_lines = fillStartEnd(src_lines, add_start, add_end)
	trg_lines = fillStartEnd(trg_lines, add_start, add_end)

	#pad events, convert to indices, and create output lines for trg
	max_src_event_length = findMaxLength(src_lines)
	max_trg_event_length = findMaxLength(trg_lines)
	src_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]

	src_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]


	trg_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_trg_event_length - len(line))
		for line in trg_lines
	]

	trg_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
		[word2ind['<pad>']] * (max_trg_event_length - len(line))
		for line in trg_lines
	]

	src_input_lines = Variable(torch.LongTensor(src_input_lines)).cuda()
	trg_input_lines = Variable(torch.LongTensor(trg_input_lines)).cuda()
	src_output_lines = Variable(torch.LongTensor(src_output_lines)).cuda()
	trg_output_lines = Variable(torch.LongTensor(trg_output_lines)).cuda()

	return src_input_lines, src_output_lines, trg_input_lines, trg_output_lines, line_index, t_index


def read_data_pipeline(data, t, word2id, add_start=True, add_end=True):
	#data is a single story
	split = data.split("|||")
	if len(split) > t+1:
		#print(split[t], "|||", split[t+1])
		src_line = fillStartEnd([split[t].split()], add_start, add_end)
		trg_line = fillStartEnd([split[t+1].split()], add_start, add_end)

		src_line = [[word2id[w] if w in word2id else word2id['<unk>'] for w in src_line[0]]]
		trg_line = [[word2id[w] if w in word2id else word2id['<unk>'] for w in trg_line[0]]]
		src_line = torch.LongTensor(src_line).cuda()
		t = t+1
		"""
		elif len(data) == t+1:
			split = data.split("|||")
			src_line = [word2id[w] if w in word2id else word2id['<unk>'] for w in split[t].split()]
			trg_line = ["<pad>" for w in split[t].split()]
		"""
	elif len(split) == t+1:
		#print(split[t])
		src_line = fillStartEnd([split[t].split()], add_start, add_end)
		src_line = [[word2id[w] if w in word2id else word2id['<unk>'] for w in src_line[0]]]
		src_line = torch.LongTensor(src_line).cuda()
		trg_line = None
		t = 0
	else:
		#print("NONE")
		src_line = None
		trg_line = None
		t = 0
	return src_line, trg_line, t


def pipeline_srcOnly(src_lines, word2ind, add_start=True, add_end=True):
	#data is a single story
	src_lines = fillStartEnd(src_lines, add_start, add_end)
	max_src_event_length = findMaxLength(src_lines)
	src_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]

	src_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]

	src_input_lines = Variable(torch.LongTensor(src_input_lines)).cuda()
	src_output_lines = Variable(torch.LongTensor(src_output_lines)).cuda()

	return src_input_lines, src_output_lines


def pipeline_srcOnly_duplicate(src_lines, word2ind, num_dups ,add_start=True, add_end=True):
	#data is a single EVENT
	if len(src_lines) > 1: raise ValueError("Data should only be a single input event")
	src_lines = fillStartEnd(src_lines, add_start, add_end)
	max_src_event_length = findMaxLength(src_lines)
	src_lines = src_lines[0]
	src_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in src_lines[:-1]] +
		[word2ind['<pad>']] * (max_src_event_length - len(src_lines))
		for n in range(0, num_dups)
	]

	src_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in src_lines[1:]] +
		[word2ind['<pad>']] * (max_src_event_length - len(src_lines))
		for n in range(0, num_dups)
	]

	src_input_lines = Variable(torch.LongTensor(src_input_lines)).cuda()
	src_output_lines = Variable(torch.LongTensor(src_output_lines)).cuda()

	return src_input_lines, src_output_lines


def process_batch_pipeline(
	src_lines, trg_lines, word2ind, add_start=True, add_end=True
):

	src_lines = fillStartEnd(src_lines, add_start, add_end)
	trg_lines = fillStartEnd(trg_lines, add_start, add_end)

	#pad events, convert to indices, and create output lines for trg
	max_src_event_length = findMaxLength(src_lines)
	max_trg_event_length = findMaxLength(trg_lines)
	src_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]


	trg_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_trg_event_length - len(line))
		for line in trg_lines
	]

	src_input_lines = torch.LongTensor(src_input_lines).cuda()
	trg_input_lines = torch.LongTensor(trg_input_lines).cuda()

	return src_input_lines, trg_input_lines


def init_state(input, num_layers, bidirectional, src_hidden_dim):
    """Get cell states and hidden states."""
    #print(input.size(0))
    #print(input.size(1))
    batch_size = input.size(0) #\
        #if self.encoder.batch_first else input.size(1)

    num_directions = 2 if bidirectional else 1
    src_hidden_dim = src_hidden_dim // 2 if bidirectional else src_hidden_dim

    h0_encoder = Variable(torch.zeros(
        num_layers * num_directions,
        batch_size,
        src_hidden_dim
    ), requires_grad=False)

    c0_encoder = Variable(torch.zeros(
        num_layers * num_directions,
        batch_size,
        src_hidden_dim
    ), requires_grad=False)

    return h0_encoder.cuda(), c0_encoder.cuda()










def read_policy_data_set(encoder_inputs, outputs, buckets, batch_size):
    data_set = [[] for _ in buckets]
    for i in range(0,batch_size):
        cur_vector = [encoder_inputs[i][3-j] for j in xrange(5)]
        target_vector = [outputs[j][i] for j in xrange(5)] #this was already 5, should it be 6?
        cur_vector = np.array(cur_vector)
        target_vector = np.array(target_vector)
        data_set[0].append([cur_vector,target_vector])
    return data_set



def process_batch_srcOnly(
	src_lines, word2ind, add_start=True, add_end=True
):
	print("SRC_LINES",src_lines)
	src_lines = fillStartEnd(src_lines, add_start, add_end)

	#pad events, convert to indices, and create output lines for trg

	max_src_event_length = findMaxLength(src_lines)
	print("SRC_LINES, filled",src_lines)
	src_input_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[:-1]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]

	src_output_lines = [
		[word2ind[w] if w in word2ind else word2ind['<unk>'] for w in line[1:]] +
		[word2ind['<pad>']] * (max_src_event_length - len(line))
		for line in src_lines
	]


	src_input_lines = Variable(torch.LongTensor(src_input_lines)).cuda()
	src_output_lines = Variable(torch.LongTensor(src_output_lines)).cuda()

	return src_input_lines, src_output_lines
