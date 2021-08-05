#!/usr/bin/env python
# coding: utf-8


'''
Basic Setup
'''
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForNextSentencePrediction

# Load pre-trained model (weights)
NSPmodel = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
NSPmodel.eval()

LMmodel = BertForMaskedLM.from_pretrained('bert-base-uncased')
LMmodel.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch
import numpy as np
from itertools import zip_longest
from itertools import product
import time

# enable GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LMmodel.to(device)
NSPmodel.to(device)


def maskEvent(inpEvent):
    '''
    Args:
        inpEvent(string): event tuple
    Returns:
        EVENT(string): masked event
    '''
    EVENT = []
    for i in range(len(inpEvent)):
        inpEvent[i] = inpEvent[i].lower()
    while(inpEvent[-1]=='emptyparameter'):
        inpEvent = inpEvent[:-1]
    EVENT = " ".join(inpEvent).replace('emptyparameter', '[MASK]')
    return EVENT



def score_sentence(first, second):
    ''' 
    Args:
        first(list): A list of tokens, including [CLS] in the front
        second(list): A list of tokens, including [SEP] in the end
    Returns:
        score: measures how likely that "second" is the next sentence of "first"
    '''
    len1 = len(first)
    len2 = len(second)
    INPUT = first[:]
    INPUT.extend(second)
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    segments_ids = []
    segments_ids.extend(np.zeros(len1+len2, np.int32).tolist())
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    predictions = NSPmodel(tokens_tensor, segments_tensors )
    score = predictions[0][0][0].cpu().detach().numpy()
    # print(score)
    return score


def score_single_sentence(second):
    '''
    Args:
        second(list): A list of tokens, including [SEP] in the end
    Returns:
        score: measures how likely that "second" is a sentence in natural language
    '''
    INPUT = second[:]
    INPUT.insert(0, '[CLS]')
    len2 = len(INPUT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    segments_ids = []
    segments_ids.extend(np.zeros(len2, np.int32).tolist())
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)
    predictions = NSPmodel(tokens_tensor, segments_tensors )
    score = predictions[0][0].detach().numpy()
    # print(score)
    return score


def E2S(event, max_masks):
    '''
    Args:
        event(string): A string representing an event, without [CLS], [SEP], or period.
        max_masks(int): The maximum number of masks that is allowed between every two consecutive tokens.
        i.e. at each blank the possible number of masks is in the range [1, max_masks]
    Returns:
        sentences(list): A list of strings, each string is a sentence created by inserting [MASK]s between some tokens of the event in a certain way
    ''' 
    sentences = []
    event = event.rstrip()
    event = event.split(" ")
    # print(event)
    l = len(event)
    for m in product([i for i in range(0,max_masks+1)],repeat = l+1):
        loc=l
        sen = event[:]
        while loc>=0:
            for i in range(m[loc]):
                sen.insert(loc,'[MASK]')
            loc -= 1
        sen.append('.')
        sentences.append(" ".join(sen))
    return sentences


def fill_in_blanks(first, second):
    '''
    Args:
        first(list): A list of tokens, with period and without [CLS], which is a complete sentence.
        second(list): A list of tokens, with period and without [SEP], which may contain several [MASK]s.
    Returns:
        first(list): A list of tokens which form a complete sentence, including [CLS] in the front.
        second(list): A list of tokens which represent the complete sentence after BERT replacing those [MASK]s in the input with real words, including [SEP] in the end.

        The resturned arguments are ready to be passed into "score_sentence" function.
    '''
    first.insert(0, '[CLS]')
    second.append('[SEP]')
    len1 = len(first)
    len2 = len(second)
    INPUT = first[:]
    INPUT.extend(second)
    index = []
    for i in range(len1-1, len(INPUT)):
        if INPUT[i]=='[MASK]':
            index.append(i)
    segments_ids = []
    segments_ids.extend(np.zeros(len(INPUT), np.int32).tolist())
        
    indexed_tokens = tokenizer.convert_tokens_to_ids(INPUT)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])
    tokens_tensor = tokens_tensor.to(device)
    segments_tensor = segments_tensor.to(device)

    with torch.no_grad():
        predictions = LMmodel(tokens_tensor, segments_tensor)[0]
    for i in index:
        predicted_index = torch.argmax(predictions[0, i]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        second[i-len1] = predicted_token[0]
 
    # generated sentences with period in the middle or consecutive repitition of words will be discarded before scoring
    for i in range(len(second)-1):
        if second[i] == second[i+1]:
            second = []
            return (first, second)
    #print("len(second) = ", len(second))
    for i in range(len(second)-2):
        if second[i] == '.':
            second = []
            #print(second)
            return (first, second)    
    return (first, second)


def avg_bleu(cand, ref, w):
    '''
    Args:
        cand(string): Generated sentence
        ref(string): Ground truth sentence
        w(list): A list of weights for BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    Returns:
        score: Weighted average of BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    '''
    from nltk.translate.bleu_score import sentence_bleu
    
    reference = []
    reference.append(ref.split(" "))
    candidate = cand.split(" ")
    #print(candidate, reference)
    score = sentence_bleu(reference, candidate, weights=w)
    #print(score)
    return score


def avg_bleu_remove_overlap(event, cand, ref, w, no_overlap=[]):
    ''' Since the words in the event must appear in the generated sentence, so they are already known.
        This function is going to compute BLEU scores between candidates and references
        by looking at the sentences after removing the already-known words in the event.

    Args:
        event(string): The event from which "cand" is generated
        cand(string): Generated sentence
        ref(string): Ground truth sentence
        w(list): A list of weights for BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        no_overlap(list): Store the results in the order of "event, candidate, reference" for every example. It is ready to be written into an output file.
    Returns:
        score: Weighted average of BLEU-1, BLEU-2, BLEU-3, BLEU-4
    '''
    from nltk.translate.bleu_score import sentence_bleu
    e = event.split(" ")
    reference = []
    reference.append(ref.split(" "))
    candidate = cand.split(" ")
    for word in e:
        if word in candidate:
            candidate.remove(word)
        if word in reference[0]:
            reference[0].remove(word)
    #print("\nafter removing overlap: ")
    print(e, candidate, reference)
    score = sentence_bleu(reference, candidate, weights=w)
    r = ""
    r += " ".join(e) + "\n"
    r += " ".join(candidate) + "\n"
    r += " ".join(reference[0]) + "\n"
    no_overlap.append(r)
    return score


def BERT_fill(prev, e, max_masks):
    '''
    Args:
        prev(list): historic context, a list of N-1 previous complete sentences, each previous sentence is a list of words. 
        e(string): A string representing a event, without [CLS], [SEP], or period.
    Returns:
        best_sentence(list):
        BERT will explore all kinds of sentence structures by generating a sentence with certain number of [MASK]s between each two consecutive tokens.
        Then BERT will score all the generated sentences and choose the one with the highest score.
    '''
    start = time.time()

    masked = maskEvent(e)
    sentences = E2S(masked, max_masks)
    best_sentence = None
    highest_score = -float('inf')
    #print(len(sentences))
    count = 0
    for s in sentences:
        if time.time() > start+20: return e
        second = tokenizer.tokenize(s)
        first = []
        for i in range(len(prev)):
            first.extend(prev[i])
        complete_first, complete_second = fill_in_blanks(first, second)
        if len(complete_second) == 0:
            continue
        count +=1
        score = score_sentence(complete_first, complete_second)
        #print("SCORE",score)
        if score>highest_score:
            highest_score = score
            best_sentence = complete_second

    #print(count)
    try:
        best_sentence.remove('[SEP]')
    except AttributeError:
        return ""
    return best_sentence

"""
'''
Specify Inputs
'''
answerfile = "whole_sentence_seg.txt"
outputfile = "noOverlap_output_maxmask2_new.txt"
scorefile = "noOverlap_maxmask2_new.txt"

prev = []
prev.append(['the', 'talks', 'stalled', 'again','.'])
prev.append(['at', 'gate', 'daniel','plans', 'to', 'go', 'to', 'antarctica', 'but', 'then', 'weir', 'enters', 'and', 'tells', 'him','.'])
prev.append(['he', 'gets', 'angry', 'about', 'the', 'situation','.'])
prev.append(['she', 'cannot', 'do', 'anything','.'])
prev.append(['she', 'leaves','.'])
prev.append(['they', 'are', 'not', 'allowed', 'to', 'use', 'the', 'place', 'one','.'])
prev.append(['she', 'later', 'talks', 'with', 'someone', 'and', 'its', 'members','protest','.'])
prev.append(['they', 'cannot', 'use', 'it', 'because', 'they', 'want', 'to', 'show', 'their', 'goodwill', 'to', 'the', 'world','.'])
prev.append(['she', 'tells', 'them','.'])

prev.append(['carter', 'then', 'proposes', 'using', 'the', 'modified', 'it', 'to', 'get', 'to', 'there', 'to', 'contact', 'him', 'who', 'could', 'help', 'jack','.'])
prev.append(['they', 'perhaps', 'need', 'that', 'ship', 'in', 'the', 'future', 'to', 'defend', 'off', 'the','.'])
prev.append(['the', 'engines', 'could', 'burn', 'out', 'on', 'the', 'flight','.'])
prev.append(['however', 'weir', 'denies', 'request','.'])
prev.append(['later', 'carter', 'talks', 'with', 'weir', 'privately', 'and', 'asks', 'her', 'to', 'consider', 'her', 'request','.'])
prev.append(['she', 'denies','.'])
prev.append(['carter', 'threatens', 'to', 'refuse', 'to', 'work', 'on', 'the', 'modified', 'cargo', 'ship','.'])
prev.append(['she', 'gets', 'the', 'allowance','.'])
prev.append(['carter', 'then', 'talks', 'with', 'him', 'about', 'their', 'flight','.'])
prev.append(['daniel', 'enters','.'])

prev.append(['he', 'has', 'to', 'stay', 'because', 'if', 'the', 'two', 'fail', 'he', 'would', 'be', 'the', 'only', 'one', 'left', 'to', 'help','.'])
prev.append(['he', 'is', 'informed','.'])
prev.append(['some', 'time', 'later', 'carter', 'and', 'tim', 'fly', 'through', 'hyper','space', 'to', 'mars','.'])
prev.append(['neill', 'modified', 'the', 'engines','.'])
prev.append(['he', 'did', 'it','.'])
prev.append(['during', 'the', 'flight', 'carter', 'tries', 'to', 'find', 'out', 'but', 'she', 'cannot', 'find', 'out','.'])
prev.append(['she', 'then', 'tries', 'to', 'start', 'a', 'conversation', 'with', 'tim','.'])
prev.append(['suddenly', 'the', 'gate', 'is', 'activated','.'])
prev.append(['they', 'receive', 'a', 'text', 'message', 'from', 'carol',',', 'a', 'system', 'lord', 'who', 'wants', 'to', 'arrange', 'a', 'meeting', 'between', 'earth', 'and', 'the', 'system', 'lords','.'])
prev.append(['weir', 'is', 'then', 'authorized', 'by', 'president', 'henry', 'hayes', 'to', 'start', 'negotiations','.'])

lengths = []
gram = 30
for i in range(gram-1):
    lengths.append(len(prev[i]))



outputs= []
max_masks = 2


'''
Extract events
'''
EVENT, event_per_line = extract_raw_event()

best_sentence = BERT_fill(prev, e, max_masks)


'''
Begin BERT Fillling!
'''
line_index = 0
e_index = 0
for e in EVENT:
    best_sentence = BERT_fill(prev, e, max_masks)
    prev.pop(0)
    lengths.pop(0)
    lengths.append(len(best_sentence))
    prev.append(best_sentence)

    if e_index == 0:
        e_index = event_per_line[line_index]
        line_index += 1
        if len(outputs)>0:
            print(outputs[-1])
        outputs.append(" ".join(best_sentence))
        e_index -= 1
    else:
        outputs[-1] = outputs[-1]+" "+(" ".join(best_sentence))
        e_index -= 1



print(outputs)

'''
Begin writing generated sentences!
'''
with open(outputfile, "w") as text_file:
    text_file.write("\n".join(outputs))


'''
Compute BLEU scores
'''
cand_index = 0
event_index = 0
indv_unit = []
indv_bi = []
cumu_bi = []
cumu_tri = []

indv_unit_noOverlap = []
indv_bi_noOverlap = []
cumu_bi_noOverlap = []
cumu_tri_noOverlap = []
no_overlap = []

with open(answerfile, "r") as text_file:
    for line in text_file:
        line = line.strip("\n")
        indv_unit.append(avg_bleu(outputs[cand_index], line, (1,0,0,0)))
        indv_bi.append(avg_bleu(outputs[cand_index], line, (0,1,0,0)))
        cumu_bi.append(avg_bleu(outputs[cand_index], line, (0.5,0.5,0,0)))
        cumu_tri.append(avg_bleu(outputs[cand_index], line, (1.0/3.0, 1.0/3.0, 1.0/3.0,0)))

        num_events = event_per_line[cand_index] - 1
        recovered_e = EVENT[event_index]
        event_index += 1
        while num_events>0:
            recovered_e += " " + EVENT[event_index]
            event_index += 1
            num_events -= 1
        print(recovered_e)
        indv_unit_noOverlap.append(avg_bleu_remove_overlap(recovered_e, outputs[cand_index], line, (1,0,0,0), no_overlap))
        indv_bi_noOverlap.append(avg_bleu_remove_overlap(recovered_e, outputs[cand_index], line, (0,1,0,0)))
        cumu_bi_noOverlap.append(avg_bleu_remove_overlap(recovered_e, outputs[cand_index], line, (0.5,0.5,0,0)))
        cumu_tri_noOverlap.append(avg_bleu_remove_overlap(recovered_e, outputs[cand_index], line, (1.0/3.0, 1.0/3.0, 1.0/3.0,0)))
        cand_index += 1

import statistics
print("Average of BLEU, unit gram = ", statistics.mean(indv_unit))
print("Average of BLEU, individual bi gram = ", statistics.mean(indv_bi))
print("Average of BLEU, 2-grams cumulative = ", statistics.mean(cumu_bi))
print("Average of BLEU, 3-grams cumulative = ", statistics.mean(cumu_tri))
print("\nAverage of BLEU, get rid of overlap: \n")
print("Average of BLEU, unit gram = ", statistics.mean(indv_unit_noOverlap))
print("Average of BLEU, individual bi gram = ", statistics.mean(indv_bi_noOverlap))
print("Average of BLEU, 2-grams cumulative = ", statistics.mean(cumu_bi_noOverlap))
print("Average of BLEU, 3-grams cumulative = ", statistics.mean(cumu_tri_noOverlap))


'''
Begin Writing scores!
'''
with open(scorefile, "w") as text_file:
    text_file.write("Unit gram: \n")
    text_file.write(str(indv_unit))
    text_file.write(str(statistics.mean(indv_unit)))

    text_file.write("\nIndividual bi gram: \n")
    text_file.write(str(indv_bi))
    text_file.write(str(statistics.mean(indv_bi)))

    text_file.write("\nCumulative 2-grams: \n")
    text_file.write(str(cumu_bi))
    text_file.write(str(statistics.mean(cumu_bi)))

    text_file.write("\nCumulative 3-grams: \n")
    text_file.write(str(cumu_tri))
    text_file.write(str(statistics.mean(cumu_tri)))

    # BLEU scores, remove overlap between generated sentence and event tuple
    text_file.write("\nBLEU scores after removing overlap between generated sentence and event tuple: \nUnit gram: \n")
    text_file.write(str(indv_unit))
    text_file.write(str(statistics.mean(indv_unit_noOverlap)))

    text_file.write("\nIndividual bi gram: \n")
    text_file.write(str(indv_bi))
    text_file.write(str(statistics.mean(indv_bi_noOverlap)))

    text_file.write("\nCumulative 2-grams: \n")
    text_file.write(str(cumu_bi))
    text_file.write(str(statistics.mean(cumu_bi_noOverlap)))

    text_file.write("\nCumulative 3-grams: \n")
    text_file.write(str(cumu_tri))
    text_file.write(str(statistics.mean(cumu_tri_noOverlap)))

    text_file.write("\n".join(no_overlap))
"""
