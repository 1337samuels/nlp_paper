from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
from itertools import product
import logging
import os
import pickle
import random


import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


from func import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, BertConfig
from func import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForLatentConnector
from func import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from func import XLNetLMHeadModel, XLNetTokenizer
from func import TransfoXLLMHeadModel, TransfoXLTokenizer
from func import BertForLatentConnector, BertTokenizer

from collections import defaultdict
from modules import VAE
from utils import (TextDataset_Split,
                   TextDataset_2Tokenizers, BucketingDataLoader)


import pdb


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(
            tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    else:
        dataset = TextDataset_Split(
            tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    assert isinstance(tokenizer, list)
    if not evaluate:
        args.batch_size = args.per_gpu_train_batch_size * \
            max(1, args.n_gpu)
        file_path = args.train_data_file
    else:
        args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        file_path = args.eval_data_file
    dataloader = BucketingDataLoader(
        file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=False)
    return dataloader


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence_conditional(model, length, context, past=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu', decoder_tokenizer=None):

    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        while True:
            # for _ in trange(length):
            inputs = {'input_ids': generated, 'past': past}
            # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            # pdb.set_trace()
            if next_token.unsqueeze(0)[0, 0].item() == decoder_tokenizer.encode('<EOS>')[0]:
                break

    return generated

# a wrapper function to choose between different play modes


def evaluate_latent_space(args, model_encoder, model_decoder, encoder_tokenizer, decoder_tokenizer, generator, prefix=""):

    eval_dataloader = build_dataload_and_cache_examples(
        args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)

    # Eval!
    logger.info("***** Running recontruction evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)

    model_encoder.eval()
    model_decoder.eval()

    result = calc_rec(model_encoder, model_decoder, eval_dataloader,
                      encoder_tokenizer, decoder_tokenizer, generator, args, ns=100)
    result_file_name = "eval_recontruction_results.txt"

    eval_output_dir = args.output_dir
    output_eval_file = os.path.join(eval_output_dir, result_file_name)

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval {} results *****".format(args.play_mode))
        for key in sorted(result.keys()):
            logger.info("  %s \n %s", key, str(result[key]))
            writer.write("%s \n" % (key))
            for past, text in result[key]:
                writer.write("%s \n %s \n" % (past, text))

    return result


def calc_rec(model_encoder, model_decoder, eval_dataloader, encoder_tokenizer, decoder_tokenizer, generator, args, ns=1):

    count = 0
    result = dict()
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction"):

        # pdb.set_trace()
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        context_tokens = decoder_tokenizer.encode('<BOS>')
        with torch.no_grad():

            text_x0 = encoder_tokenizer.decode(
                x0[0, :x_lengths[0, 0]].tolist(), clean_up_tokenization_spaces=True)[0]
            # result["INPUT TEXT " + str(count)].append(text_x0)

            pooled_hidden_fea = model_encoder(
                x0, attention_mask=(x0 > 0).float())[1]

            # Connect hidden feature to the latent space
            # latent_z, loss_kl = model_vae.connect(pooled_hidden_fea)
            mean, _ = model_encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = mean.squeeze(1)

            outs = []
            for past in generator(latent_z):
                out = sample_sequence_conditional(
                    model=model_decoder,
                    context=context_tokens,
                    past=past,
                    # Chunyuan: Fix length; or use <EOS> to complete a sentence
                    length=x_lengths[0, 1],
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                    decoder_tokenizer=decoder_tokenizer
                )
                text_x1 = decoder_tokenizer.decode(
                    out[0, :].tolist(), clean_up_tokenization_spaces=True)
                text_x1 = text_x1.split()[1:-1]
                text_x1 = ' '.join(text_x1) + '\n'
                outs.append((past, text_x1))
            result[text_x0] = outs
        # Once is enough:)
        break

    return result


def test_each_dimension(latent_z: torch.Tensor):
    past = latent_z.detach().clone()
    for index, change in product(range(32), [1, -1]):
        past[:, index] += change
        yield past


def test_single_dimension(latent_z: torch.Tensor):
    past = latent_z.detach().clone()
    for change in range(32):
        past[:, 0] += change * 0.01
        yield past


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default='Snli',
                        type=str, help="The dataset.")

    # Variational auto-encoder
    parser.add_argument("--latent_size", default=32,
                        type=int, help="Latent space dimension.")
    parser.add_argument("--total_sents", default=10, type=int,
                        help="Total sentences to test recontruction.")
    parser.add_argument("--num_interpolation_steps", default=10,
                        type=int, help="Total sentences to test recontruction.")
    parser.add_argument("--play_mode", default="interpolation", type=str,
                        help="interpolation or reconstruction.")

    # Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    # Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")

    # Variational auto-encoder
    parser.add_argument("--nz", default=32, type=int,
                        help="Latent space dimension.")

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--single-dimension", action='store_true',
                        help="Test only a single dimension")

    args = parser.parse_args()
    args.dataset = 'snli'
    args.use_philly = False
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.encoder_model_type = args.encoder_model_type.lower()
    args.decoder_model_type = args.decoder_model_type.lower()
    global_step = args.gloabl_step_eval

    output_encoder_dir = os.path.join(
        args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
    output_decoder_dir = os.path.join(
        args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step))
    checkpoints = [[output_encoder_dir, output_decoder_dir]]
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    # Load a trained Encoder model and vocabulary that you have fine-tuned
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[
        args.encoder_model_type]
    model_encoder = encoder_model_class.from_pretrained(
        output_encoder_dir, latent_size=args.latent_size)
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(
        args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path, do_lower_case=args.do_lower_case)

    model_encoder.to(args.device)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer_encoder.max_len_single_sentence
    args.block_size = min(
        args.block_size, tokenizer_encoder.max_len_single_sentence)

    # Load a trained Decoder model and vocabulary that you have fine-tuned
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[
        args.decoder_model_type]
    model_decoder = decoder_model_class.from_pretrained(
        output_decoder_dir, latent_size=args.latent_size)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(
        args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
    model_decoder.to(args.device)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer_decoder.max_len_single_sentence
    args.block_size = min(
        args.block_size, tokenizer_decoder.max_len_single_sentence)

    special_tokens_dict = {'pad_token': '<PAD>',
                           'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    assert tokenizer_decoder.pad_token == '<PAD>'
    generator = test_single_dimension if args.single_dimension else test_each_dimension

    evaluate_latent_space(
        args, model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, generator=generator, prefix=global_step)


if __name__ == '__main__':
    main()
