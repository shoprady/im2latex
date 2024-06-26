# load checkpoint and evaluating
from os.path import join, dirname, exists
from os import makedirs
from functools import partial
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab
from utils import collate_fn
from model import LatexProducer, Im2LatexModel
from model.score import score_files


def main():

    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path', type=str, default="./ckpts/best_ckpt.pt", ###
                        help='path of the evaluated model')

    # model args
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=1) ###
    parser.add_argument("--result_path", type=str,
                        default="./results/result.txt", help="The file to store result")
    parser.add_argument("--ref_path", type=str,
                        default="./sample_results/ref.txt", help="The file to store reference")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")
    parser.add_argument("--split", type=str,
                        default="test", help="The data split to decode")

    args = parser.parse_args()

    # Ensure the results directory exists ###
    result_dir = dirname(args.result_path)
    if not exists(result_dir):
        makedirs(result_dir)

    # loading models
    checkpoint = torch.load(join(args.model_path), map_location=torch.device('cpu')) ###
    model_args = checkpoint['args']

    # read the dictionary and set other relevant parameters
    vocab = load_vocab(args.data_path)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    # loading Test Sets
    data_loader = DataLoader(
        Im2LatexDataset(args.data_path, args.split, args.max_len),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=4
    )

    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    result_file = open(args.result_path, 'w')
    ref_file = open(args.ref_path, 'w')

    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size)

    for imgs, tgt4training, tgt4cal_loss in tqdm(data_loader):
        try:
            reference = latex_producer._idx2formulas(tgt4cal_loss)
            results = latex_producer(imgs)
        except RuntimeError:
            break

        result_file.write('\n'.join(results))
        ref_file.write('\n'.join(reference))

    result_file.close()
    ref_file.close()
    score = score_files(args.result_path, args.ref_path)
    print("beam search result:", score)


if __name__ == "__main__":
    main()
