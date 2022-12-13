import argparse
import json
import os

import torch
import clip
import clip.simple_tokenizer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from corpus_meteor_score import corpus_meteor_score
from model.ZeroCLIP import CLIPTextGenerator
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=15)
    parser.add_argument("--cond_text", type=str, default="Image of a")
    parser.add_argument("--reset_context_delta", action="store_true",
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--multi_gpu", action="store_true")

    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])

    parser.add_argument("--caption_img_path", type=str, default='example_images/captions/COCO_val2014_000000008775.jpg',
                        help="Path to image for captioning")

    parser.add_argument("--arithmetics_imgs", nargs="+",
                        default=['example_images/arithmetics/woman2.jpg',
                                 'example_images/arithmetics/king2.jpg',
                                 'example_images/arithmetics/man2.jpg'])
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])

    parser.add_argument("--gt_annot_path", type=str, default=None,
                        help="Path to COCO-format JSON file with ground-truth annotations")

    args = parser.parse_args()

    return args

def run(args, img_path):
    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))

    if os.path.isdir(img_path):
        imgs_paths = [os.path.join(img_path, file_name) for file_name in os.listdir(img_path)]
    else:
        imgs_paths = [img_path]

    if args.gt_annot_path is not None:
        with open(args.gt_annot_path) as annot_file:
            gt_data = json.load(annot_file)

        filenames_to_paths = {os.path.basename(path): path for path in imgs_paths}
        file_ids_to_paths = {info['id']: filenames_to_paths[info['file_name']] for info in gt_data['images'] if info['file_name'] in filenames_to_paths}
        gt_captions = {file_ids_to_paths[info['image_id']]: info['caption'] for info in gt_data['annotations'] if info['image_id'] in file_ids_to_paths}
    else:
        gt_captions = dict()

    tokenizer = clip.simple_tokenizer.SimpleTokenizer()

    all_token_strs = []
    for this_path in imgs_paths:
        print(f'Captioning "{this_path}":')
        image_features = text_generator.get_img_feature([this_path], None)
        captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

        encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
        encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

        print(captions)
        print('Best clip:', args.cond_text + captions[best_clip_idx])

        if this_path in gt_captions:
            print(f'Ground truth: {gt_captions[this_path]}')
            gt_caption_tokens = tokenizer.encode(gt_captions[this_path])
            gt_caption_token_strs = [tokenizer.decode([token]) for token in gt_caption_tokens]
            hyp_caption_tokens = tokenizer.encode(captions[best_clip_idx])
            hyp_caption_token_strs = [tokenizer.decode([token]) for token in hyp_caption_tokens]
            all_token_strs.append((hyp_caption_token_strs, gt_caption_token_strs))
            print(f'BLEU-4 for this caption: {sentence_bleu([gt_caption_token_strs], hyp_caption_token_strs)}')
            print(f'METEOR for this caption: {single_meteor_score(gt_caption_token_strs, hyp_caption_token_strs)}')

    if len(all_token_strs) > 0:
        print('Overall metrics:')
        print(f'BLEU-4: {corpus_bleu([[gt_strs] for _, gt_strs in all_token_strs], [hyp_strs for hyp_strs, _ in all_token_strs])}')
        print(f'METEOR: {corpus_meteor_score([gt_strs for _, gt_strs in all_token_strs], [hyp_strs for hyp_strs, _ in all_token_strs])}')
    else:
        print('No overall metrics to report.')


def run_arithmetic(args, imgs_path, img_weights):
    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))

    image_features = text_generator.get_combined_feature(imgs_path, [], img_weights, None)
    captions = text_generator.run(image_features, args.cond_text, beam_size=args.beam_size)

    encoded_captions = [text_generator.clip.encode_text(clip.tokenize(c).to(text_generator.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', args.cond_text + captions[best_clip_idx])

if __name__ == "__main__":
    args = get_args()

    if args.run_type == 'caption':
        run(args, img_path=args.caption_img_path)
    elif args.run_type == 'arithmetics':
        args.arithmetics_weights = [float(x) for x in args.arithmetics_weights]
        run_arithmetic(args, imgs_path=args.arithmetics_imgs, img_weights=args.arithmetics_weights)
    else:
        raise Exception('run_type must be caption or arithmetics!')