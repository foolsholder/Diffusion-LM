"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    set_seed(101)
    args = create_argparser().parse_args()

    #dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True

    # args.diffusion_steps = 200 #500  # DEBUG

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    import torch
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    # diffusion.rescale_timesteps = False  # DEBUG --> REMOVE
    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    #model.to(dist_util.dev())
    model.to('cuda:0')
    model.eval() # DEBUG

    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    logger.log("sampling...")

    model3 = get_weights(model2, args)

    def sample_sample(model_x):

        all_images = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample_shape = (args.batch_size, 64, args.in_channel)
            print(sample_shape)
            sample = sample_fn(
                model_x,
                sample_shape,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
                model_kwargs=model_kwargs,
                top_p =args.top_p,
            )
            all_images += [sample.cpu().numpy()]

        arr = np.concatenate(all_images, axis=0)
        print(arr.shape, 'full shape')
        arr = arr[: args.num_samples * args.mbr_sample]

        word_lst_e2e = []
        x_t = th.tensor(arr).cuda()
        reshaped_x_t = x_t
        logits = model_x.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
        for seq in cands.indices:
            if isinstance(tokenizer, dict):
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
            else:
                tokens = tokenizer.decode(seq.squeeze(-1))
            word_lst_e2e.append(tokens)

        for idx, xx in enumerate(word_lst_e2e):
            word_lst_e2e[idx] = xx.replace('PAD', '[PAD]').replace('UNK', '[UNK]').replace('START', '').replace('END', '')
        return word_lst_e2e

    from improved_diffusion.metrics import BloomMetric, GPTMetric
    model.eval()

    from tqdm.auto import trange
    def generate_text(num_texts):
        generated_texts = []

        num_iters = num_texts // 64
        for _ in trange(num_iters):
            text = sample_sample(model)
            generated_texts += text
        return generated_texts

    def compute_metric(metric_fn, batch_size, texts):
        ind = 0
        metric = 0.0

        num_iters = len(texts) // batch_size
        for _ in trange(num_iters):
            text = texts[ind:ind + batch_size]
            metric += metric_fn(text) * len(text)
            ind += batch_size
            print(ind)
        return metric / len(texts)

    def estimate_model(num_texts):
        texts = generate_text(num_texts)

        metric_bloom_fn = BloomMetric(device="cuda:0")
        metric_gpt_fn = GPTMetric(device="cuda:0")
        metrics_json = dict()
        metrics_file = f"metrics.json"

        metric_bloom = compute_metric(metric_bloom_fn, 8, texts)
        print(f"Bloom metric: {metric_bloom:0.5f}")
        metric_gpt = compute_metric(metric_gpt_fn, 8, texts)
        print(f"GPT2 metric: {metric_gpt:0.5f}")
        metrics_json = {"Bloom metric": metric_bloom, "GPT2 metric": metric_gpt}

        import json
        with open(metrics_file, "w") as file:
            json.dump(metrics_json, file)

    estimate_model(
        2 ** 15
    )


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
