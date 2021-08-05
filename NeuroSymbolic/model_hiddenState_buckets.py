"""Sequence to Sequence models."""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

#Adapted from https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/model.py
#Check out https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L1165 for Torch's dimension defaults.

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1) #####ADDED DIMENSION
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

class LSTMAttentionDot(nn.Module):
    """A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            #print "LSTMAttentionDot recurrance()"
            #print input.size()
            #print self.input_weights(input), self.hidden_weights(hx)
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            #TODO check for hidden[0] or hidden
            output.append(isinstance(hidden, tuple) and hidden[0])

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class Seq2SeqAttentionSharedEmbeddingHidden(nn.Module):
    """Container module with an encoder, decoder, embeddings."""

    def __init__(
        self,
        emb_dim,
        vocab_size,
        hidden_dim,
        pad_token,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.
    ):
        """Initialize model."""
        super(Seq2SeqAttentionSharedEmbeddingHidden, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = hidden_dim
        self.trg_hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token = pad_token

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            self.pad_token
        )

        self.src_hidden_dim = self.src_hidden_dim // 2 \
            if self.bidirectional else self.src_hidden_dim

        self.encoder = nn.LSTM(
            emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = LSTMAttentionDot(
            emb_dim,
            self.trg_hidden_dim,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            self.trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(self.trg_hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)


    def forward(self, input_src, input_trg, h_encoder, c_encoder):
        """Propogate input through the network."""
        src_emb = self.embedding(input_src)
        trg_emb = self.embedding(input_trg)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (h_encoder, c_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1) #context

        trg_h, (_, _) = self.decoder( #forward(input, hidden, ctx, ctx_mask=None); returns output, hidden
            trg_emb,
            (decoder_init_state, c_t),
            ctx#,
            #ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit, src_h_t, src_c_t


    def decode_encoder(self, input_src, h_encoder, c_encoder):
        src_emb = self.embedding(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (h_encoder, c_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1) #context
        return decoder_init_state, ctx, src_h_t, src_c_t, c_t

    def decode_decoder(self, input_trg, decoder_init_state, c_t, ctx): #one word at a time
        trg_emb = self.embedding(input_trg)

        trg_h, (_, _) = self.decoder( #forward(input, hidden, ctx, ctx_mask=None); returns output, hidden
            trg_emb,
            (decoder_init_state, c_t),
            ctx
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
            )
        return decoder_logit


    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.softmax(logits_reshape, dim = -1) #####ADDED DIMENSION
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs
