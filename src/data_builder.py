from html import entities
from math import degrees
from tqdm import tqdm
from random import shuffle
import os
import json
import ast
import argparse
from data_handler import write_entity_to_id, write_relation_to_id, write_triples_to_file
import yaml
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gt', help='dataset name')
parser.add_argument('--split_ratio', default=[0.1, 0.1], help='ratio of data spliting', nargs=2)
parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

args = parser.parse_args()
with open('config.yml', 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

data_path = config['data'][args.dataset]['raw']
out_path = config['data'][args.dataset]['path']

def process_em_graph(graph):
    nodes, edges = [], []
    for predicate in graph:
        src = predicate['source'].lower()
        nodes.append(src)
        for arg in predicate['arguments']:
            nodes.append(arg['value'].lower())
            edges.append({'head': src, 'tail': arg['value'].lower(), 'type': arg['role']})
    return list(set(nodes)), edges

def convert_em_graph(text, graph):
    nodes = {}
    edges = []
    emvista_attrs = ['Time', 'Cotime', 'TimeExact', 'Cotimeexact', 'TimeMin', 'Cotimemin', 'TimeMax', 'Cotimemax', 'TimeDuration', 'Cotimeduration', 
                        'TimeFuzzy', 'Cotimefuzzy', 'Measure', 'Comeasure', 'Measuremax', 'Measuremin', 'Measureexact', 'Comeasureexact', 'Measurefuzzy']
    for predicate in graph:
        src = predicate['source'].lower()
        args = {}
        for arg in predicate['arguments']:
            args[arg['role']] = arg['value'].lower()
        keys = list(args.keys())
        if 'Experiencer' in keys and 'Attribute' in keys:
            new_src = args['Experiencer']
            if new_src not in nodes:
                nodes[new_src] = {'attributes': []}
            if 'attributes' not in nodes[new_src]:
                nodes[new_src]['attributes'] = []
            nodes[new_src]['attributes'].append(args['Attribute'])
        elif len(keys) == 2 and 'Theme' in keys: 
            src = args['Theme']
            keys.remove('Theme')
            if keys[0] in emvista_attrs:  
                if src not in nodes:
                    nodes[src] = {keys[0]: args[keys[0]], 'attributes': []}
                else:
                    nodes[src][keys[0]] = args[keys[0]]
        else:
            if src not in nodes:
                nodes[src] = {'attributes': []}
            for k, v in args.items():
                if k in emvista_attrs:
                    nodes[src][k] = v
                else:
                    if v not in nodes:
                        nodes[v] = {'attributes': []}
                    edges.append({'head': src, 'tail': v, 'type': k})

    for predicate in graph:
        src = predicate['source'].lower()
        args = {}
        for arg in predicate['arguments']:
            args[arg['role']] = arg['value'].lower()
        keys = list(args.keys())
        if 'Experiencer' in keys and 'Attribute' in keys:
            new_src = args['Experiencer']
            if src != new_src and src in nodes and new_src in nodes:
                for k, v in nodes[src].items():
                    if k != 'attributes':
                        nodes[new_src][k] = v
                    else:
                        nodes[new_src]['attributes'].extend(v)
                del nodes[src]
                for edge in edges:
                    if edge['tail'] == src:
                        edge['tail'] = new_src
                    if edge['head'] == src:
                        edge['head'] = new_src

        elif len(keys) == 2 and 'Theme' in keys: 
            continue
        else:
            for k, v in args.items():
                if k not in emvista_attrs and v not in nodes:
                    for kk, vv in nodes.items():
                        if v in vv['attributes']:
                            for edge in edges:
                                if edge['tail'] == v:
                                    edge['tail'] = kk
                                    break

    return nodes, edges

def build_em(merge=False):
    triples = []
    entities, entity2id, id2entity, relations = {}, {}, {}, []
    with open(os.path.join(data_path, '10k_graphs_hydrogenfc.json'), 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        for k in tqdm(range(len(data))):
            nodes, edges = process_em_graph(data[k]['graph']['predicates'])
            # nodes, edges = convert_em_graph(data[k]['text'], data[k]['graph']['predicates'])
            if not merge:
                cnt = len(entities)
                e2i = {}
                for i, j in enumerate(nodes): #.keys()
                    txt = '_'.join(j.split()).lower()
                    entities[cnt + i] = j
                    id2entity[cnt + i] = txt + '_' + str(k)
                    entity2id[txt + '_' + str(k)] = cnt + i
                    e2i[j] = i
                for r in edges:
                    relations.append(r['type'].lower())
                    triples.append({'type': r['type'].lower(), 'head': e2i[r['head']] + cnt, 'tail': e2i[r['tail']] + cnt})

    if args.verbose:
        print('#entities', len(entities), '#relations', len(relations), '#triples', len(triples))

    write_entity_to_id(os.path.join(out_path, 'entity2id.txt'), entity2id)
    write_relation_to_id(os.path.join(out_path, 'relation2id.txt'), dict((j, i) for i, j in enumerate(set(relations))))
    write_triples_to_file(os.path.join(out_path, 'data.txt'), triples, id2entity)

def process_gt_file(file_name):
    actors, _entities, phrases, _relations = [], {}, {}, []
    has_entity, has_phrase, arcs = {}, {}, {}

    with open(os.path.join(data_path, file_name), 'r', encoding='utf8') as f:
        for line in f:
            data = ast.literal_eval(line)
            if data['type'] == 'node':
                if 'AnnotatedEntity' in data['labels']:
                    _entities[data['id']] = {'type': data['properties']['type_in_context']}
                elif 'DatastoreActor' in data['labels']:
                    actors.append({'id': data['id'], 'type': data['labels'][1]})
                elif 'Phrase' in data['labels']:
                    phrases[data['id']] = data['properties']['text_content']
                elif 'AnnotatedRelation' in data['labels']:
                    _relations.append({'id': data['id'], 'type': data['properties']['predicate']})
            elif data['type'] == 'relationship':
                if data['label'] in ['has_object_phrase', 'has_subject_phrase']:
                    c = data['label'].split('_')[1]
                    if data['start']['id'] not in arcs:
                        arcs[data['start']['id']] = {'id': data['id'], c + '_id': data['end']['id']}
                    else:
                        arcs[data['start']['id']][c + '_id'] = data['end']['id']
                elif data['label'] == 'has_entity':
                    has_entity[data['start']['id']] = {'id': data['id'], 'entity_id': data['end']['id']}
                elif data['label'] == 'has_phrase':
                    has_phrase[data['start']['id']] = {'id': data['id'], 'phrase_id': data['end']['id']}
                
    entities, relations = {}, []
    for e in tqdm(actors):
        entity_id = [i for i, v in has_entity.items() if v['entity_id'] == e['id']][0]
        entities[e['id']] = {'id': e['id'], 'type': e['type'], 'text': phrases[has_phrase[entity_id]['phrase_id']]}

    for r in tqdm(_relations):
        head_phrase_id = arcs[r['id']]['subject_id']
        tail_phrase_id = arcs[r['id']]['object_id']
        _head_entity_id = [e for e, v in has_phrase.items() if v['phrase_id'] == head_phrase_id][0]
        _tail_entity_id = [e for e, v in has_phrase.items() if v['phrase_id'] == tail_phrase_id][0]
        relations.append({'id': r['id'], 'type': r['type'], 'head': has_entity[_head_entity_id]['entity_id'], 'tail': has_entity[_tail_entity_id]['entity_id']})
        

    relations = [r for r in relations if r['type'] != 'unclear']
    if args.verbose:
        print('number of known relations ->', len(relations))

    in_degrees = {}
    out_degrees = {}

    for e in entities:
        in_degrees[e] = 0
        out_degrees[e] = 0

    for r in relations:
        in_degrees[r['head']] += 1
        out_degrees[r['tail']] += 1

    non_zeros = []
    for id, e in entities.items():
        if in_degrees[id] + out_degrees[id] > 0:
            non_zeros.append(e)
    if args.verbose:
        print('number of non-zero entities ->', len(non_zeros))
    return non_zeros, relations

def build_gt_old():
    geotrend_files = list(filter(lambda f: f.endswith('.json'), os.listdir(data_path)))
    data = []
    entities, entity2id, id2entity, relations = {}, {}, {}, []
    for i, file_name in enumerate(geotrend_files):
        _entities, _relations = process_gt_file(file_name)
        id2id = {}
        for e in _entities:
            id = len(entities)
            txt = '_'.join(e['text'].split()).lower()
            if txt in entity2id:
                id = entity2id[txt]
            else:
                entity2id[txt] = id
                id2entity[id] = txt
            id2id[e['id']] = id
            entities[id] = {'type': e['type'].lower(), 'text': txt}
        for r in _relations:
            relations.append(r['type'].lower())
            data.append({'type': r['type'].lower(), 'head': id2id[r['head']], 'tail': id2id[r['tail']]})
    if args.verbose:
        print('#entities', len(entities), '#relations', len(relations), '#triples', len(data))
    write_entity_to_id(os.path.join(out_path, 'entity2id.txt'), entity2id)
    write_relation_to_id(os.path.join(out_path, 'relation2id.txt'), dict((j, i) for i, j in enumerate(set(relations))))
    write_triples_to_file(os.path.join(out_path, 'data.txt'), data, id2entity)

def build_gt():
    searches = list(filter(lambda f: os.path.isdir(os.path.join(data_path, f)), os.listdir(data_path)))
    triples = []
    for search in searches:
        # print('\n------------------------\n', search)
        entities_file = os.path.join(data_path, search, 'entities', 'Annotation-ActorAnnotation.csv')
        relations_type_file = os.path.join(data_path, search, 'entities', 'Annotation-RelationAnnotation.csv')
        relations_file = os.path.join(data_path, search, 'relations', 'Annotation-ACTOR-Annotation.csv')
        entity_id2name = {}
        relation_id2type = {}
        with open(entities_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            is_first = True
            for row in reader:
                if is_first:
                    is_first = False
                    continue
                entity_id2name[row[0]] = row[1]

        with open(relations_type_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            is_first = True
            for row in reader:
                if is_first:
                    is_first = False
                    continue
                relation_id2type[row[0]] = row[1]

        
        with open(relations_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            is_head = True
            head, relation, tail = None, None, None
            is_first = True
            for row in reader:
                if is_first:
                    is_first = False
                    continue
                relation = relation_id2type[row[0]]
                if relation == 'unclear':
                    continue
                if is_head:
                    head = entity_id2name[row[1]]
                else:
                    tail = entity_id2name[row[1]]
                    triples.append((head, relation, tail))
                is_head = not is_head

    heads = [triple[0] for triple in triples]
    tails = [triple[2] for triple in triples]
    heads.extend(tails)
    entities = set(heads)
    relations = set([triple[1] for triple in triples])

    if args.verbose:
        print('# of relations', len(relations))
        print('# of entities', len(entities))
        print('# of triples', len(triples))
    with open(os.path.join(out_path, 'data.txt'), 'w', encoding='utf8') as f:
        for triple in triples:
            f.write(triple[0]+ '\t' + triple[1] + '\t' + triple[2] + '\n')

def remove_duplicates():
    with open(os.path.join(out_path, 'data.txt'), 'r', encoding='utf-8') as f:
        data = list(map(lambda x: x.lower(), f.readlines()))

    if args.verbose:
        print(len(data), len(set(data)))
    data = list(set(data))

    with open(os.path.join(out_path, 'data.txt'), 'w', encoding='utf-8') as f:
        for d in data:
            f.write(d)

def split_data():
    data, train, valid, test = [], [], [], []
    e_train, e_valid, e_test = set(), set(), set()

    with open(os.path.join(out_path, 'data.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            x = line.strip().split('\t')
            if x[0] not in e_train or x[2] not in e_train:
                train.append(line)
                e_train.add(x[0])
                e_train.add(x[2])
            else:
                data.append(line)

        n_test = int(len(lines) * args.split_ratio[0])
        n_valid = int((len(lines) - n_test) * args.split_ratio[1])
        n_train = len(lines) - n_test - n_valid
        if args.verbose:
            print('#triples', len(lines), 'test size', n_test, 'valid size', n_valid, 'train size', n_train, 'initial train size', len(train))

    shuffle(data)
    n_train -= len(train)
    train.extend(data[:n_train])
    valid.extend(data[n_train:n_train + n_valid])
    test.extend(data[n_train + n_valid:])

    e_train = set()
    for r in train:
        x = r.strip().split('\t')
        e_train.add(x[0])
        e_train.add(x[2])

    for r in valid:
        x = r.strip().split('\t')
        e_valid.add(x[0])
        e_valid.add(x[2])

    for r in test:
        x = r.strip().split('\t')
        e_test.add(x[0])
        e_test.add(x[2])

    if args.verbose:
        print(len(e_train), len(e_valid), len(e_test))
        print(len(train), len(valid), len(test))
        print(set(e_test) <= set(e_train) and set(e_valid) <= set(e_train))

    for split in ['train', 'valid', 'test']:
        with open(os.path.join(out_path, split + '.txt'), 'w', encoding='utf-8') as f:
            for r in eval(split):
                f.write(r)

    in_degrees = {}

    with open(os.path.join(out_path, 'train.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            x = line.strip().split('\t')
            if x[2] not in in_degrees:
                in_degrees[x[2]] = 0
            in_degrees[x[2]] += 1

    # mean degrees
    mean_in_degree = sum(in_degrees.values()) / len(in_degrees)
    if args.verbose:
        print('mean in degree', mean_in_degree)

    # median degrees
    in_degrees = sorted(in_degrees.values())
    median_in_degree = in_degrees[int(len(in_degrees) / 2)]
    if args.verbose:
        print('median in degree', median_in_degree)

if args.dataset == 'gt':
    build_gt()
remove_duplicates()
split_data()
