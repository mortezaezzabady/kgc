from numpy import dtype
import torch
import os
import pandas as pd
from transformers import AdamW, BertModel, BertTokenizer, BertForMaskedLM, AutoModel, AutoTokenizer
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
import re
import json
import random

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

sentence_templates = {
    'capitalistic': '[h] and [t] has a capitalistic relation.',
    'client':       '[h] is a client of [t].',
    'competitor':   '[h] and [t] are competitors.',
    'mention':      '[h] mentions [t].',
    'partner':      '[h] and [t] are partners.',
    'product':      '[h] is a product of [t].',
    'trial':        '[h] and [t] are in a trial.',
}
emvista_relations = ['material', 'codestination', 'comaterial', 'attribute', 'condition', 'restriction', 'theme', 'il', 'destination', 'illustration', 'patient', 'location', 'value',
'co-patient', 'coexperiencer', 'measure', 'cotheme', 'co-agent', 'opposition', 'consequence', 'stimulus', 'colocationexact', 'agent', 'beneficiary', 'copatient', 'locationsource',
'coevent', 'manner', 'addition', 'alternative', 'coattribute', 'time', 'cotopic', 'whatever', 'copurpose', 'cosource', 'locationexact', 'experiencer', 'timemax', 'source', 'enumeration',
'coagent', 'comparison', 'event', 'coproduct', 'purpose', 'instrument', 'product', 'asset', 'coasset', 'corecipient', 'cotime', 'costimulus', 'recipient', 'topic',
'co-theme', 'type', 'result', 'cobeneficiary', 'coinstrument', 'timemin', 'cocause', 'copivot', 'pivot', 'colocation', 'cause']

fb_bad_rels = {
    '/music/performance_role/regular_performances./music/group_membership/role': 'present',
    '/film/film/release_date_s./film/film_regional_release_date/film_release_region': 'released in',
    '/people/deceased_person/place_of_death': 'in',
    '/people/cause_of_death/people': 'cause of death',
    '/award/award_category/disciplines_or_subjects': 'discipline or subject',
    '/language/human_language/countries_spoken_in': 'spoken in',
    '/people/ethnicity/people': 'ethnicity',
    '/base/popstra/celebrity/dated./base/popstra/dated/participant': 'participated',
    '/organization/organization/headquarters./location/mailing_address/citytown': 'city or town',
    '/film/film/costume_design_by': 'costume designed by',
    '/organization/organization/headquarters./location/mailing_address/state_province_region': 'state or province region',
    '/film/film/story_by': 'story is by',
    '/film/film_set_designer/film_sets_designed': 'film sets were designed by',
    '/film/film/film_production_design_by': 'film production design is by',
    '/organization/organization/place_founded': 'founded',
    '/film/film/release_date_s./film/film_regional_release_date/film_regional_debut_venue': 'film\'s regional debut venue',
    '/film/actor/film./film/performance/special_performance_type': 'special performance of type',
    '/film/film/film_art_direction_by': 'film art direction was by',
    '/education/university/international_tuition./measurement_unit/dated_money_value/currency': 'measured',
    '/film/film/personal_appearances./film/personal_film_appearance/person': 'personal film appearances',
}

wnr = {
    "also_see":"seen together",
    "derivationally_related_form":"derived from",
    "has_part":"part",
    "hypernym":"broader category encompassing",
    "instance_hypernym":"particular instance",
    "member_meronym":"member",
    "member_of_domain_region":"associated with a specific region",
    "member_of_domain_usage":"pertains to a particular field or domain of usage",
    "similar_to":"similar to",
    "synset_domain_topic_of":"topic area to",
    "verb_group":"semantic relationship in verb form"
}

filename = 'train.csv'
BASE_HID_DIM = 768 #1024 #768


class BERT(torch.nn.Module):
    def __init__(self, model, hid_dim, cls_dim):
        super().__init__()
        self.bert = model
        self.linear1 = nn.Linear(BASE_HID_DIM, hid_dim)
        self.linear2 = nn.Linear(hid_dim, cls_dim)
        self.hid_dim = hid_dim

    def forward(self, input_ids, mask):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=mask, return_dict=False)
        batch_size, seq_length, hidden_size = sequence_output.shape
        linear1_output = self.linear1(sequence_output[:, :, :].view(-1, BASE_HID_DIM)) 
        new_embeddings = linear1_output.reshape(batch_size, seq_length, self.hid_dim)
        linear2_output = self.linear2(new_embeddings)
        return torch.mean(linear2_output, dim=1), new_embeddings

def read_relation_from_id(filename):
    relation2id = {}
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split()[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

def generate_sentence(args, head, relation, tail):
    if args.dataset == 'fb' or args.dataset == 'wn':
        templates = json.load(open(os.path.join(args.data, 'samples.json')))
        template = templates[relation]['templates'][0] #random.choice(templates[relation]['templates'])
        return template.replace('[h]', head).replace('[t]', tail)
    return 'relation of ' + head + ' and ' + tail + ' is ' + relation
    return sentence_templates[relation].replace('[h]', head).replace('[t]', tail)

def clean_sentence(sentence):
    sentence = sentence.replace('_', ' ')
    sentence = sentence.replace('nerd:', '')
#    sentence = re.sub(r'[^\w\s]|\d+', '', sentence)
    return sentence

def generate_csv(args, preprocess=False):
    data = []
    with open(os.path.join(args.data, 'train.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            head, relation, tail = line.strip().split()
            sentence = generate_sentence(args, head, relation, tail)
            if preprocess:
                sentence = clean_sentence(sentence)
            data.append([sentence, relation2id[relation]])
#            data.append([sentence, emvista_relations.index(relation)])
#            data.append([generate_sentence(head, relation, tail), list(sentence_templates.keys()).index(relation)])
    df = pd.DataFrame(data, columns=['sentence', 'relation'])
    df.to_csv(os.path.join(args.data, filename), index=False)

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BERT(AutoModel.from_pretrained(args.model), hid_dim=args.dim, cls_dim=args.num_cls)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    for epoch in range(args.epochs):
        epoch_losses = []
        for j, batch in tqdm(enumerate(pd.read_csv(os.path.join(args.data, filename), chunksize=args.batch_size)), desc='Epoch {}'.format(epoch)):
            sentences = batch['sentence'].tolist()
            labels = torch.LongTensor(np.array(batch['relation'].tolist())).to(device)
            encoded_dict = tokenizer.batch_encode_plus(
                                sentences, 
                                add_special_tokens = True, 
                                max_length = args.max_len, 
                                padding = True, 
                                truncation=True, 
                                return_attention_mask = True, 
                                return_tensors = 'pt',
                        )
            input_ids = encoded_dict['input_ids'].to(device)
            attention_mask = encoded_dict['attention_mask'].to(device)
            outputs, embeddings = model(input_ids, attention_mask)
            outputs = F.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            epoch_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        if args.verbose:
            print('Epoch {} loss: {}'.format(epoch, np.mean(epoch_losses)))
    torch.save(model.state_dict(), os.path.join(args.data, 'Fine_Tuned_BertModel'))

def get_word_vector(sentence, tokenizer, model):
    encoded = tokenizer.encode_plus(sentence, return_tensors='pt', add_special_tokens=False).to(device)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))

    with torch.no_grad():
        output, embeddings = model(encoded['input_ids'], encoded['attention_mask'])
    embeddings = embeddings.squeeze(0)
    return embeddings, tokens

def get_triple_vectors(args, head, relation, tail, tokenizer, model):
    sentence = generate_sentence(args, head, relation, tail)
    sentence = clean_sentence(sentence)
    embeddings, tokens = get_word_vector(sentence, tokenizer, model)
    if args.dataset == 'fb':
        if relation in fb_bad_rels:
            relation = fb_bad_rels[relation]
        else:
            relation = ' '.join(relation.split('/')[-1].split('_'))
            if sentence.find(relation + 's') != -1:
                relation = relation + 's'
            elif relation.endswith('s'):
                if sentence.find(relation[:-1] + ' ') != -1:
                    relation = relation[:-1]
                elif sentence[len(sentence) - len(relation) + 1:] == relation[:-1]:
                    relation = relation[:-1]
    elif args.dataset == 'wn':
        relation = wnr[relation]
    head_tokens = tokenizer.tokenize(clean_sentence(head))
    relation_tokens = tokenizer.tokenize(clean_sentence(relation))
    tail_tokens = tokenizer.tokenize(clean_sentence(tail))

    # print(tokens)
    # print(head_tokens)
    # print(relation_tokens)
    # print(tail_tokens)

    h_start = [i for i in range(len(embeddings) - len(head_tokens) + 1) if tokens[i: i + len(head_tokens)] == head_tokens][0]
    H = torch.mean(embeddings[h_start: h_start + len(head_tokens)], 0)  
    t_start = [i for i in range(len(embeddings) - len(tail_tokens) + 1) if tokens[i: i + len(tail_tokens)] == tail_tokens][0]
    T = torch.mean(embeddings[t_start: t_start + len(tail_tokens)], 0) 
    print('XXX', len(embeddings), len(relation_tokens), len(tokens))
    r_start = [i for i in range(len(embeddings) - len(relation_tokens) + 1) if tokens[i: i + len(relation_tokens)] == relation_tokens]
    print(len(r_start), relation, ' XXX ', sentence)
    r_start = r_start[0]
    R = torch.mean(embeddings[r_start: r_start + len(relation_tokens)], 0) 
    C = torch.cat((R, H, T))
    return H, R, T, C

def test(args, h, r, t):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BERT(AutoModel.from_pretrained(args.model), args.dim, args.num_cls)
    model.load_state_dict(torch.load(os.path.join(args.data, 'Fine_Tuned_BertModel')))
    model.to(device)
    model.eval()

    H, R, T, C = get_triple_vectors(args, h, r, t, tokenizer, model)
    # print(C.shape, R.shape, H.shape, T.shape)
    
if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config['data']['type'], help='dataset name')
    parser.add_argument('--model', type=str, default=config['embeddings']['word']['model'], help='model name')
    parser.add_argument('--epochs', type=int, default=config['embeddings']['word']['epochs'], help='number of epochs')
    parser.add_argument('--lr', type=float, default=config['embeddings']['word']['lr'], help='learning rate')
    parser.add_argument('--max_len', type=int, default=config['embeddings']['word']['max_length'], help='max length of sentence')
    parser.add_argument('--batch_size', type=int, default=config['embeddings']['word']['batch_size'], help='chunk size')
    parser.add_argument('--max_grad_norm', type=float, default=config['embeddings']['word']['max_grad_norm'], help='max gradient norm')
    parser.add_argument('--dim', type=int, default=config['embeddings']['word']['dim'], help='embedding dimension')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')

    args = parser.parse_args()
    args.data = config['data'][args.dataset]['path']
    relation2id = read_relation_from_id(os.path.join(args.data, 'relation2id.txt'))
    args.num_cls = len(relation2id)
    generate_csv(args, True)
    train(args)
    		
    # test(args, 'World_War_II', '/film/film_subject/films', 'The_Guns_of_Navarone')
