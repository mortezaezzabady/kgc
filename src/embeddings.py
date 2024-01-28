import csv
import os
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import predict
import pandas as pd
import numpy as np
import torch
from data_handler import write_entity_to_id, write_relation_to_id
from tqdm import tqdm
from datetime import datetime
import argparse
import yaml

with open('config.yml', 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=config['data']['type'], help='dataset name')
parser.add_argument('--model', type=str, default=config['embeddings']['graph']['model'], help='model name')
parser.add_argument('--epochs', type=int, default=config['embeddings']['graph']['epochs'], help='number of epochs')
parser.add_argument('--dim', type=int, default=config['embeddings']['graph']['dim'], help='embedding dimension')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()

data_path = config['data'][args.dataset]['path']
result_file = os.path.join(config['results']['path'], args.dataset, 'results.csv')
embeddings_path = os.path.join(config['embeddings']['path'], args.dataset)
model_path = os.path.join(embeddings_path, args.model + '_' + str(args.epochs))

def create_tsv(filename):
    if os.path.exists(filename.replace('.txt', '.tsv')):
        return TriplesFactory.from_path(filename.replace('.txt', '.tsv'))
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.strip().split('\t') for line in f]
    if args.verbose:
        print(filename, len(data))
    with open(filename.replace('.txt', '.tsv'), 'w', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for row in data:
            writer.writerow(row)
    return TriplesFactory.from_path(filename.replace('.txt', '.tsv'))

def train(args, train_rels, valid_rels, test_rels):
    result = pipeline(
        training=train_rels,
        validation=valid_rels,
        testing=test_rels,
        model=args.model,
        epochs=args.epochs,
        model_kwargs={
            'embedding_dim': args.dim,
        },
    )
    result.save_to_directory(model_path)

def pred():
    reciprocal_ranks_head, reciprocal_ranks_tail = [], []
    hits_at_100_head, hits_at_100_tail = 0, 0
    hits_at_ten_head, hits_at_ten_tail = 0, 0
    hits_at_three_head, hits_at_three_tail = 0, 0
    hits_at_one_head, hits_at_one_tail = 0, 0
    ranks_head, ranks_tail = [], []

    model = torch.load(os.path.join(model_path, 'trained_model.pkl'))
    for triple in tqdm(test_rels.triples):
        head, rel, tail = triple
        predicted_heads_df = predict.get_head_prediction_df(model, rel, tail, triples_factory=train_rels, remove_known=False)
        predicted_tails_df = predict.get_tail_prediction_df(model, head, rel, triples_factory=train_rels, remove_known=False)

        head_rank = predicted_heads_df.head_label.tolist().index(head) + 1
        tail_rank = predicted_tails_df.tail_label.tolist().index(tail) + 1

        ranks_head.append(head_rank)
        ranks_tail.append(tail_rank)
        reciprocal_ranks_head.append(1 / head_rank)
        reciprocal_ranks_tail.append(1 / tail_rank)

        if head_rank <= 100:
            hits_at_100_head += 1
        if head_rank <= 10:
            hits_at_ten_head += 1
        if head_rank <= 3:
            hits_at_three_head += 1
        if head_rank <= 1:
            hits_at_one_head += 1
        if tail_rank <= 100:
            hits_at_100_tail += 1
        if tail_rank <= 10:
            hits_at_ten_tail += 1
        if tail_rank <= 3:
            hits_at_three_tail += 1
        if tail_rank <= 1:
            hits_at_one_tail += 1
    cumulative_hits_100 = ((hits_at_100_head / (len(ranks_head))) + (hits_at_100_tail / (len(ranks_tail)))) / 2
    cumulative_hits_ten = ((hits_at_ten_head / (len(ranks_head))) + (hits_at_ten_tail / (len(ranks_tail)))) / 2
    cumulative_hits_three = ((hits_at_three_head / (len(ranks_head))) + (hits_at_three_tail / (len(ranks_tail)))) / 2
    cumulative_hits_one = ((hits_at_one_head / (len(ranks_head))) + (hits_at_one_tail / (len(ranks_tail)))) / 2
    cumulative_mean_rank = ((np.mean(ranks_head) + np.mean(ranks_tail)) / 2)
    cumulative_mean_recip_rank = ((np.mean(reciprocal_ranks_head) + np.mean(reciprocal_ranks_tail)) / 2)
    if args.verbose:
        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))
    row = [args.dataset, args.model, args.epochs, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 
        cumulative_hits_100, cumulative_hits_ten, cumulative_hits_three, cumulative_hits_one, cumulative_mean_rank, cumulative_mean_recip_rank]
    columns = ['dataset', 'model_name', 'epochs', 'time', 'hits@100', 'hits@10', 'hits@3', 'hits@1', 'MR', 'MRR']
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(result_file, mode='a', header=not os.path.exists(result_file))

train_rels = create_tsv(data_path + 'train.txt')
valid_rels = create_tsv(data_path + 'valid.txt')
test_rels = create_tsv(data_path + 'test.txt')

# write_entity_to_id(os.path.join(data_path, 'entity2id.txt'), train_rels.entity_to_id)
# write_relation_to_id(os.path.join(data_path, 'relation2id.txt'), train_rels.relation_to_id)

# train(args, train_rels, valid_rels, test_rels)
# pred()
