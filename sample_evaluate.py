# load checkpoint and evaluating
from os.path import join, dirname, exists
from os import makedirs, listdir
import argparse

from tqdm import tqdm

from PIL import Image

import torch
from torchvision import transforms

from build_vocab import Vocab, load_vocab
from model import LatexProducer, Im2LatexModel


def main():

    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path', type=str, default="./ckpts/best_ckpt.pt",
                        help='path of the evaluated model') ### pt 파일 위치

    # model args
    parser.add_argument("--data_path", type=str, ### 이미지 데이터 경로 지정
                        default="./sample_data/", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=1) ### 5 -> 1 수정
    parser.add_argument("--result_path", type=str, ### 라텍스 txt 저장 경로 지정
                        default="./sample_results/result.txt", help="The file to store result")
    parser.add_argument("--ref_path", type=str, ### 본 코드에서 ref.txt는 제외함
                        default="./sample_results/ref.txt", help="The file to store reference")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")

    args = parser.parse_args()

    # Ensure the results directory exists ###
    result_dir = dirname(args.result_path)
    if not exists(result_dir):
        makedirs(result_dir)

    # loading models
    checkpoint = torch.load(join(args.model_path), map_location=torch.device('cpu'))
    model_args = checkpoint['args']

    # read the dictionary and set other relevant parameters
    vocab = load_vocab(args.data_path)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # produce latex (results.txt)
    result_file = open(args.result_path, 'w')

    # decoding에 있는 class
    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size)
    
    transform = transforms.ToTensor()
    imgs = listdir(args.data_path + "input")
    for img_name in tqdm(imgs):
        img_path = args.data_path + "input/" + img_name
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        results = latex_producer(img_tensor)
        result_file.write('\n'.join(results))

    result_file.close()

if __name__ == "__main__":
    main()
