
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from tqdm import tqdm


import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.basename(__file__), 'code'))


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_model_classes():
    from func import (BertConfig, BertForLatentConnector, BertTokenizer,
                      GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer)
    return {
        'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
        'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
    }


def build_dataload_and_cache_examples(args, tokenizer):
    from utils import (BucketingDataLoader)

    assert isinstance(tokenizer, list)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    file_path = args.eval_data_file
    dataloader = BucketingDataLoader(
        file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=False)
    return dataloader


# a wrapper function to choose between different play modes
def evaluate_latent_space(args, model_encoder, encoder_tokenizer, decoder_tokenizer, prefix=""):

    eval_dataloader = build_dataload_and_cache_examples(
        args, [encoder_tokenizer, decoder_tokenizer])

    # Eval!
    logger.info("***** Running recontruction evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)

    model_encoder.eval()

    result = calc_rec(model_encoder, eval_dataloader,
                      encoder_tokenizer, args)
    return result


def calc_rec(model_encoder, eval_dataloader, encoder_tokenizer, args):
    result = defaultdict(str)
    for batch in tqdm(eval_dataloader, desc="Evaluating recontruction"):
        x0, x1, x_lengths = batch

        max_len_values, _ = x_lengths.max(0)
        x0 = x0[:, :max_len_values[0]]
        x1 = x1[:, :max_len_values[1]]

        x0 = x0.to(args.device)
        x1 = x1.to(args.device)
        x_lengths = x_lengths.to(args.device)

        with torch.no_grad():

            text_x0 = encoder_tokenizer.decode(
                x0[0, :x_lengths[0, 0]].tolist(), clean_up_tokenization_spaces=True)[0]

            pooled_hidden_fea = model_encoder(
                x0, attention_mask=(x0 > 0).float())[1]

            mean, _ = model_encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = mean.squeeze(1)
            result[text_x0] = latent_z

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", default=None, type=str,
                        help="An input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The directory where checkpoints are saved.")
    parser.add_argument("--output-file", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Variational auto-encoder
    parser.add_argument("--latent_size", default=32,
                        type=int, help="Latent space dimension.")

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

    parser.add_argument('--global_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    args = parser.parse_args()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.encoder_model_type = args.encoder_model_type.lower()
    args.decoder_model_type = args.decoder_model_type.lower()
    global_step = args.global_step_eval

    output_encoder_dir = os.path.join(
        args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
    output_decoder_dir = os.path.join(
        args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step))
    checkpoints = [[output_encoder_dir, output_decoder_dir]]
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    model_classes = get_model_classes()
    # Load a trained Encoder model and vocabulary that you have fine-tuned
    _, encoder_model_class, encoder_tokenizer_class = model_classes[
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
    _, decoder_model_class, decoder_tokenizer_class = model_classes[
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

    results = evaluate_latent_space(
        args, model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder)

    with open(args.output_file, 'w') as f:
        for result in results.values():
            f.write(str(result))


if __name__ == '__main__':
    main()
