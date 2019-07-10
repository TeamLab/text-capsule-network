import os
import argparse

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices='20news mr trec sst mpqa reuters mrshort imdb mnist'.split()
                        , help='Name of dataset')
    parser.add_argument('--data_path', default='{}/../dataset/'.format(THIS_DIR), help='pickle data dir')
    parser.add_argument('--gpu_id', default="5", type=str)
    parser.add_argument('--save_ckpt', default='./model_ckpt')
    
    # common train hyer-params
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--dropout_ratio', default=0.1, type=float)
    

    # specific dataset parser
    if parser.parse_args().dataset == 'trec':
        parser = trec_config(parser)
    elif parser.parse_args().dataset == '20news':
        parser = news_config(parser)
    elif parser.parse_args().dataset == 'reuters':
        parser = news_config(parser)
    elif parser.parse_args().dataset == 'mr':
        parser = mr_config(parser)
    elif parser.parse_args().dataset == 'mrshort':
        parser = mrshort_config(parser)
    elif parser.parse_args().dataset == 'mpqa':
        parser = mpqa_config(parser)
    else:
        parser = imdb_config(parser)

    return parser.parse_args()


def news_config(parser):
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=3, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=10, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing zero is static routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser


def reuters_config(parser):
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=3, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=10, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser


def mr_config(parser):
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=3, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=16, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser


def mrshort_config(parser):
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=3, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=10, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser


def trec_config(parser):
    parser.add_argument('--batch_size', default=50, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=5, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=10, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser


def mpqa_config(parser):
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=1, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=10, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser


def imdb_config(parser):
    parser.add_argument('--batch_size', default=40, type=int, help='Batch size')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--embedding_size', default=300, type=int, help='Embedding size')
    parser.add_argument('--filter_size', default=3, type=int, help='convolution filter size of capsule network')
    parser.add_argument('--num_cap', default=6, type=int, help='set number of capsule')
    parser.add_argument('--len_ui', default=10, type=int, help='set length of primary capsule property')
    parser.add_argument('--len_vj', default=16, type=int, help='set length of output capsule property')
    parser.add_argument('--routing', default=0, type=int, help='method of routing')
    parser.add_argument('--num_filter', default=256, type=int, help='number of filters')
    return parser
