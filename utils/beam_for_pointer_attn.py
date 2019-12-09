#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division
from utils.config import *
import sys
from zhon.hanzi import punctuation
import  string
# reload(sys)
# sys.setdefaultencoding('utf8')
from utils.rouge import rouge
import os
import time
import torch
from torch.autograd import Variable


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)

def dup_batch(batch, idx, dup_times):
    new_batch = {}
    input_len = batch["input_lengths"][idx]
    for key in ["input_batch", "target_batch"]:
        new_batch[key] = batch[key][:input_len, idx:idx+1].repeat(1, dup_times)

    if "input_ext_vocab_batch" in batch:
        for key in ["input_ext_vocab_batch", "target_ext_vocab_batch"]:
            new_batch[key] = batch[key][:input_len, idx:idx+1].repeat(1, dup_times)
        new_batch["article_oovs"] = [batch["article_oovs"][idx] for _ in range(dup_times)]
        new_batch["max_art_oovs"] =  batch["max_art_oovs"]

    for key in ["input_txt", "target_txt"]:
        new_batch[key] = [batch[key][idx] for _ in range(dup_times)]
    for key in ["input_lengths", "target_lengths"]:
        new_batch[key] = batch[key][idx:idx+1].repeat(dup_times)

    return new_batch

class BeamSearch(object):
    def __init__(self, model, args, lang):
        
        self.model = model
        self.args = args
        self.lang = lang

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):

        batch_size = batch["input_lengths"].size(0)
        decoded_sents = []
    
        for i in range(batch_size):
            new_batch = dup_batch(batch, i, self.args["beam_size"])
            enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = self.model.get_input_from_batch(new_batch)
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search_sample(enc_batch, enc_padding_mask, enc_lens, 
                                            enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            if self.args["pointer_gen"]:
                art_oovs = batch["article_oovs"][i]
                len_oovs = len(art_oovs)
                decoded_words = []
                for idx in output_ids:
                    if idx < self.args["vocab_size"]:
                        decoded_words.append(self.lang.idx2word[idx])    
                    elif idx - self.args["vocab_size"] < len_oovs:
                        decoded_words.append(art_oovs[idx - self.args["vocab_size"]])
                    else:
                        raise ValueError("invalid output id")
            else:
                decoded_words = [self.lang.idx2word[idx] for idx in output_ids]

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index('EOS')
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            decoded_sents.append(decoded_words)
        return decoded_sents

    def beam_search_sample(self, enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0):
        #batch should have only one example by duplicate
        
        encoder_outputs, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze(0)
        dec_c = dec_c.squeeze(0)
        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[SOS_idx],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if self.args["is_coverage"] else None))
                 for _ in range(self.args["beam_size"])]
        results = []
        steps = 0
        while steps < min(self.args["max_dec_step"], self.args["max_r"]) and len(results) < self.args["beam_size"]:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.args["vocab_size"] else UNK_idx \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if USE_CUDA:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if self.args["is_coverage"]:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)
            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t, _ = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps, training=False)
            if self.args["use_rl"]:
                final_dist[:, self.lang.word2idx["举动"]] = 0.0
                final_dist[:, self.lang.word2idx["受益者"]] = 0.0

            topk_log_probs, topk_ids = torch.topk(final_dist, self.args["beam_size"] * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if self.args["is_coverage"] else None)

                for j in range(self.args["beam_size"] * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].data[0],
                                   log_prob=topk_log_probs[i, j].data[0],
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == EOS_idx:
                    non_punc_steps = len(h.tokens) -  len([token for token in h.tokens if token < self.args["vocab_size"] and (self.lang.idx2word[float(token)] in punctuation or self.lang.idx2word[float(token)] in string.punctuation)])
                    if non_punc_steps >= self.args["min_dec_steps"]:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.args["beam_size"] or len(results) == self.args["beam_size"]:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]
