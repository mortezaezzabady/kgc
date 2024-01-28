import numpy as np
from corpus import Corpus

import torch
import os


def write_triples_to_file(filename, triples_data, id2entity):
    with open(filename, 'w', encoding='utf8') as f:
        for triple in triples_data:
            f.write(id2entity[triple['head']] + '\t' + str(triple['type']) + '\t' + id2entity[triple['tail']] + '\n')

def write_entity_to_id(filename, entity2id):
    with open(filename, 'w', encoding='utf8') as f:
        for entity, entity_id in entity2id.items():
            f.write(entity + '\t' + str(entity_id) + '\n')

def write_relation_to_id(filename, relation2id):
    with open(filename, 'w', encoding='utf8') as f:
        for relation, relation_id in relation2id.items():
            f.write(relation + '\t' + str(relation_id) + '\n')

def read_entity_from_id(filename):
    entity2id = {}
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id

def read_relation_from_id(filename):
    relation2id = {}
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

def init_embeddings(model_path):
    model = torch.load(os.path.join(model_path, 'trained_model.pkl'))
    return model.entity_representations[0](indices=None).detach().cpu().numpy(), model.relation_representations[0](indices=None).detach().cpu().numpy()

def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2

def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)

def build_corpus(args, is_unweigted=False, directed=True):
    path = args.data
    entity2id = read_entity_from_id(os.path.join(path, 'entity2id.txt'))
    relation2id = read_relation_from_id(os.path.join(path, 'relation2id.txt'))

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    word_embedding_model = None if not args.get_ctx and not args.use_ctx else args.word_embedding_model
    word_embedding_dim = args.word_embedding_size
    word_embedding_path = args.data
    corpus = Corpus((train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, word_embedding_model, word_embedding_dim, word_embedding_path, args.get_2hop)

    if args.embedding_model != 'random':
        entity_embeddings, relation_embeddings = init_embeddings(args.emb_path)
        print("Initialised relations and entities from", args.embedding_model)

    else:
        entity_embeddings = np.random.randn(len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)