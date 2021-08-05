import sys
from data_utils import *
from model_hiddenState_buckets import Seq2SeqAttentionSharedEmbeddingHidden
import numpy as np
import argparse
import os
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
	#from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class Decoder:
	def __init__(self, config_path, top_n=1):
		self.config = read_config(config_path)
		load_dir = self.config['data']['load_dir']
		model_name = self.config['data']['preload_weights']

		self.w2i, self.i2w = read_vocab(self.config['data']['vocab_file'])
		self.max_length = self.config['data']['max_length']
		src_vocab_size = len(self.w2i)
		self.top_n = top_n


		self.model = Seq2SeqAttentionSharedEmbeddingHidden(
			emb_dim=self.config['model']['dim_word_src'],
			vocab_size=src_vocab_size,
			hidden_dim=self.config['model']['hidden_dim'],
			pad_token=self.w2i['<pad>'],
			bidirectional=self.config['model']['bidirectional'],
			nlayers=self.config['model']['n_layers_src'],
			nlayers_trg=self.config['model']['n_layers_trg'],
			dropout=self.config['model']['dropout']
		  ).cuda()


		if load_dir:
			self.model.load_state_dict(torch.load(
				open(os.path.join(load_dir,model_name), 'rb')
			))


	def pipeline_predict(self,data, h, c, start=False):
		input_lines_src, _ = pipeline_srcOnly_duplicate(data, self.w2i, self.top_n)
		#print(input_lines_src.data.cpu())
		#input_lines_src, _ = process_batch_pipeline(batch['src'], batch['trg'], self.w2i, add_start=True, add_end=True)
		if start == True or self.config['model']['hidden'] == False:
			h, c = init_state(input_lines_src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['hidden_dim'])
				#restart hidden state; input_lines_src only used for size here

		input_lines_trg = torch.LongTensor(
			[
				[self.w2i['<s>']]
				for i in range(0, input_lines_src.data.size(0))
			]
		).cuda()
		#print(input_lines_trg.data.cpu())

		options = []

		size = 0
		temperature = 1.0
		decoder_init_state, ctx, h, c, c_t = self.model.decode_encoder(input_lines_src, h, c)
		
		while size < self.max_length:
			#decoder_logit, _, _ = self.model(input_lines_src, input_lines_trg, h, c)
			decoder_logit = self.model.decode_decoder(input_lines_trg, decoder_init_state, c_t, ctx)
			h.detach_()
			c.detach_()

			decoder_logit = decoder_logit[0, -1, :] / temperature
			filtered_logit = decoder_logit #top_k_top_p_filtering(decoder_logit, top_k=0, top_p=0.9)

			probabilities = F.softmax(filtered_logit, dim=-1)
			#next_token = torch.multinomial(probabilities, 1)
			
			"""
			word_probs = self.model.decode(
				decoder_logit
			)#.data.cpu().numpy()#.argmax(axis=-1) # batch size x event x vocab size
			"""
			n = self.top_n
			word_max = torch.multinomial(probabilities, n) #no replacement by default
			#print([self.i2w[x] for x in word_max.data.cpu().numpy()])
			
			#word_max = probabilities.max(-1)			

			#input_lines_trg = word_max.indices
			size = input_lines_trg.size(-1)
			unsqueezed = word_max.unsqueeze(0)
			transposed = torch.transpose(unsqueezed, 0, 1)
			#print(transposed.data.cpu())

			input_lines_trg = torch.cat(
				 (input_lines_trg, transposed),
				 1
			)
			#print(input_lines_trg.data.cpu())

		gen_events = input_lines_trg.data.cpu().numpy().tolist()#[1:]
		#options.append(gen_event)
		#print(gen_events)
		returning = []
		#quit()
		for sentence_pred in gen_events: #for each of the n
			#for sentence_pred in option_n: #for each item in batch
				sentence_pred = sentence_pred[1:]

				sentence_pred = [self.i2w[x] for x in sentence_pred]#[:5]]

				if '</s>' in sentence_pred:
					index = sentence_pred.index('</s>')
					sentence_pred = sentence_pred[:index]

				returning.append(sentence_pred)
				#print('Predicted : %s ' % (' '.join(sentence_pred)))
				#print('-----------------------------------------------')
		return returning, h, c

	def pipeline_predict_alreadyID(self,input_lines_src, h, c, start=False):
		if start == True or self.config['model']['hidden'] == False:
			h, c = init_state(input_lines_src, self.config['model']['n_layers_src'], self.config['model']['bidirectional'], self.config['model']['hidden_dim'])
				#restart hidden state; input_lines_src only used for size here

		input_lines_trg = torch.LongTensor(
			[
				[self.w2i['<s>']]
				for i in range(0, input_lines_src.data.size(0))
			]
		).cuda()
		#print(input_lines_trg.data.cpu())

		options = []

		size = 0
		temperature = 1.0
		decoder_init_state, ctx, h, c, c_t = self.model.decode_encoder(input_lines_src, h, c)
		
		while size < self.max_length:
			#decoder_logit, _, _ = self.model(input_lines_src, input_lines_trg, h, c)
			decoder_logit = self.model.decode_decoder(input_lines_trg, decoder_init_state, c_t, ctx)
			h.detach_()
			c.detach_()

			decoder_logit = decoder_logit[0, -1, :] / temperature
			filtered_logit = decoder_logit #top_k_top_p_filtering(decoder_logit, top_k=0, top_p=0.9)

			probabilities = F.softmax(filtered_logit, dim=-1)
			#next_token = torch.multinomial(probabilities, 1)
			
			"""
			word_probs = self.model.decode(
				decoder_logit
			)#.data.cpu().numpy()#.argmax(axis=-1) # batch size x event x vocab size
			"""
			n = self.top_n
			word_max = torch.multinomial(probabilities, n) #no replacement by default
			#print([self.i2w[x] for x in word_max.data.cpu().numpy()])
			
			#word_max = probabilities.max(-1)			

			#input_lines_trg = word_max.indices
			size = input_lines_trg.size(-1)
			unsqueezed = word_max.unsqueeze(0)
			transposed = torch.transpose(unsqueezed, 0, 1)
			#print(transposed.data.cpu())

			input_lines_trg = torch.cat(
				 (input_lines_trg, transposed),
				 1
			)
			#print(input_lines_trg.data.cpu())

		gen_events = input_lines_trg.data.cpu().numpy().tolist()#[1:]
		#options.append(gen_event)
		#print(gen_events)
		returning = []
		#quit()
		for sentence_pred in gen_events: #for each of the n
			#for sentence_pred in option_n: #for each item in batch
				sentence_pred = sentence_pred[1:]

				sentence_pred = [self.i2w[x] for x in sentence_pred]#[:5]]

				if '</s>' in sentence_pred:
					index = sentence_pred.index('</s>')
					sentence_pred = sentence_pred[:index]

				returning.append(sentence_pred)
				#print('Predicted : %s ' % (' '.join(sentence_pred)))
				#print('-----------------------------------------------')
		return returning, h, c



