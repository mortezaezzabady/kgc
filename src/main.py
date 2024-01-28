from corpus import Corpus
from data_handler import build_corpus, read_entity_from_id, read_relation_from_id, init_embeddings
from copy import deepcopy

import pickle
import argparse

import torch
import os
from models import SpKBGATModified, SpKBGATConvOnly
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy
from utils import save_model
import random
import time
import yaml

CUDA = torch.cuda.is_available()

def parse_args():
    with open('config.yml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    args = argparse.ArgumentParser()
    # network arguments

    args.add_argument('-mode', '--mode', type=str, default=config['mode'], help='system mode')
    args.add_argument('-data', '--data',
                      default=config['data']['type'], help='dataset name')
    args.add_argument('-e_g', '--epochs_gat', type=int,
                      default=config['gat']['epochs'], help='Number of epochs')
    args.add_argument('-e_c', '--epochs_conv', type=int,
                      default=config['conv']['epochs'], help='Number of epochs')
    args.add_argument('-w_gat', '--weight_decay_gat', type=float,
                      default=config['gat']['weight_decay'], help='L2 reglarization for gat')
    args.add_argument('-w_conv', '--weight_decay_conv', type=float,
                      default=config['conv']['weight_decay'], help='L2 reglarization for conv')
    args.add_argument('-emb', '--embedding_model', type=str, default=config['embeddings']['graph']['model'], help='embedding model')
    args.add_argument('-emb_size', '--embedding_size', type=int,
                      default=config['embeddings']['graph']['dim'], help='Size of embeddings (if pretrained not used)')
    args.add_argument('-ctx_emb', '--word_embedding_model', type=str, default=config['embeddings']['word']['model'], help='word embedding model')
    args.add_argument('-ctx_emb_size', '--word_embedding_size', type=int, default=config['embeddings']['word']['dim'], help='Size of word embeddings')

    args.add_argument('-l', '--lr', type=float, default=config['gat']['lr'])
    args.add_argument('-g2hop', '--get_2hop', type=bool, default=config['nhop']['get'])
    args.add_argument('-u2hop', '--use_2hop', type=bool, default=config['nhop']['use'])
    args.add_argument('-p2hop', '--partial_2hop', type=bool, default=config['nhop']['partial'])
    args.add_argument('-gctx', '--get_ctx', type=bool, default=config['embeddings']['word']['get'])
    args.add_argument('-uctx', '--use_ctx', type=bool, default=config['embeddings']['word']['use'])

    # arguments for GAT
    args.add_argument('-b_gat', '--batch_size_gat', type=int,
                      default=config['gat']['batch_size'], help='Batch size for GAT')
    args.add_argument('-neg_s_gat', '--valid_invalid_ratio_gat', type=int,
                      default=config['gat']['valid_invalid_ratio_gat'], help='Ratio of valid to invalid triples for GAT training')
    args.add_argument('-drop_GAT', '--drop_GAT', type=float,
                      default=config['gat']['dropout'], help='Dropout probability for SpGAT layer')
    args.add_argument('-alpha', '--alpha', type=float,
                      default=config['gat']['alpha'], help='LeakyRelu alphs for SpGAT layer')
    args.add_argument('-out_dim', '--entity_out_dim', type=int, nargs='+',
                      default=config['gat']['dim'], help='Entity output embedding dimensions')
    args.add_argument('-h_gat', '--nheads_GAT', type=int, nargs='+',
                      default=config['gat']['heads'], help='Multihead attention SpGAT')
    args.add_argument('-margin', '--margin', type=float,
                      default=config['gat']['margin'], help='Margin used in hinge loss')

    # arguments for convolution network
    args.add_argument('-b_conv', '--batch_size_conv', type=int,
                      default=config['conv']['batch_size'], help='Batch size for conv')
    args.add_argument('-alpha_conv', '--alpha_conv', type=float,
                      default=config['conv']['alpha'], help='LeakyRelu alphas for conv layer')
    args.add_argument('-neg_s_conv', '--valid_invalid_ratio_conv', type=int, default=config['conv']['valid_invalid_ratio_gat'],
                      help='Ratio of valid to invalid triples for convolution training')
    args.add_argument('-o', '--out_channels', type=int, default=config['conv']['channels'],
                      help='Number of output channels in conv layer')
    args.add_argument('-drop_conv', '--drop_conv', type=float,
                      default=config['conv']['dropout'], help='Dropout probability for convolution layer')

    args = args.parse_args()
    
    args.emb_path = os.path.join(config['embeddings']['path'], args.data, args.embedding_model + '_' + str(config['embeddings']['graph']['epochs']))
    args.output_folder = os.path.join(config['checkpoints']['path'], args.data, 'out')
    args.results_path = os.path.join(config['results']['path'], args.data)
    args.dataset = args.data
    args.data = config['data'][args.data]['path']
    print(args.emb_path, args.output_folder, args.results_path, args.data)
    return args

def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss

def train_gat(args, Corpus_, entity_embeddings, relation_embeddings, node_neighbors_2hop):

    # Creating the gat model here.
    ####################################

    print('Defining model')

    print(
        '\nModel type -> GAT layer with {} heads used , Initital Embeddings training'.format(args.nheads_GAT[0]))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, 0 if not args.use_ctx else args.word_embedding_size)

    if CUDA:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    current_batch_2hop_indices = torch.tensor([], dtype=torch.int64)
    if(args.use_2hop):
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args,
                                                                          Corpus_.unique_entities_train, node_neighbors_2hop)

    if CUDA:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()
    else:
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices))

    epoch_losses = []   # losses of all epochs
    print('Number of epochs {}'.format(args.epochs_gat))

    edge_list_nhop = torch.cat(
            (current_batch_2hop_indices[:, 3].unsqueeze(-1), current_batch_2hop_indices[:, 0].unsqueeze(-1)), dim=1).t()
    print(edge_list_nhop.shape)

    if args.use_ctx:
        ctx_nhop_emb = []
        for i in range(edge_list_nhop.shape[1]):
            ctx_nhop_emb.append(Corpus_.ctx_nhop_emb[(int(edge_list_nhop[0, i].item()), int(edge_list_nhop[1, i].item()))])
        ctx_nhop_emb = torch.stack(ctx_nhop_emb)
        ctx = torch.cat((Corpus_.ctx_emb, ctx_nhop_emb.cuda()), dim=0)
        Corpus_.set_ctx_emb(ctx)
        print('C', Corpus_.ctx_emb.size()) 
    else:
        Corpus_.set_ctx_emb(torch.tensor([], dtype=torch.int64))   

    for epoch in range(args.epochs_gat):
        print('\nepoch-> ', epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices, Corpus_.ctx_emb)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print('Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}'.format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        if epoch % 100 == 99:
            print('Epoch {} , average loss {} , epoch_time {}'.format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if epoch % 100 == 99:
            save_model(model_gat, args.data, epoch, args.output_folder)

def train_conv(args, Corpus_, entity_embeddings, relation_embeddings):

    # Creating convolution model here.
    ####################################

    print('Defining model')
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, 0 if not args.use_ctx else args.word_embedding_size)
    print('Only Conv model trained')
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.cuda()
        model_gat.cuda()

    model_gat.load_state_dict(torch.load(
        '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)), strict=False) 
    model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    model_conv.final_relation_embeddings = model_gat.final_relation_embeddings

    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)

    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)

    margin_loss = torch.nn.SoftMarginLoss()

    epoch_losses = []   # losses of all epochs
    print('Number of epochs {}'.format(args.epochs_conv))

    for epoch in range(args.epochs_conv):
        print('\nepoch-> ', epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print('Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}'.format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        if epoch % 10 == 9:
            print('Epoch {} , average loss {} , epoch_time {}'.format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if epoch % 10 == 9:
            save_model(model_conv, args.data, epoch, os.path.join(args.output_folder, 'conv'))

def evaluate_conv(args, unique_entities, Corpus_, entity_embeddings, relation_embeddings):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}/trained_{1}.pth'.format(os.path.join(args.output_folder, 'conv'), args.epochs_conv - 1)), strict=False)

    model_conv.cuda()
    model_conv.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_conv, unique_entities)

if __name__ == '__main__':
    args = parse_args()
    
    Corpus_, entity_embeddings, relation_embeddings = build_corpus(args, is_unweigted=False, directed=True)

    if(args.get_2hop):
        file = args.data + '/2hop.pickle'
        with open(file, 'wb') as handle:
            pickle.dump(Corpus_.node_neighbors_2hop, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if(args.use_2hop):
        print('Opening node_neighbors pickle object')
        file = args.data + '/2hop.pickle'
        with open(file, 'rb') as handle:
            node_neighbors_2hop = pickle.load(handle)
        
    entity_embeddings_copied = deepcopy(entity_embeddings)
    relation_embeddings_copied = deepcopy(relation_embeddings)

    print('Initial entity dimensions {} , relation dimensions {}'.format(
        entity_embeddings.size(), relation_embeddings.size()))

    if args.get_ctx:
        ctx_emb = Corpus_.get_contextualized_embeddings(args, Corpus_.train_adj_matrix)
        file = args.data + '/ctx.pickle'

        with open(file, 'wb') as handle:
            pickle.dump(ctx_emb, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, Corpus_.unique_entities_train, node_neighbors_2hop)
        current_batch_2hop_indices = Variable(
            torch.LongTensor(current_batch_2hop_indices)).cuda()

        edge_list_nhop = torch.cat(
            (current_batch_2hop_indices[:, 3].unsqueeze(-1), current_batch_2hop_indices[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [current_batch_2hop_indices[:, 1].unsqueeze(-1), current_batch_2hop_indices[:, 2].unsqueeze(-1)], dim=1)

        ctx_nhop_emb = Corpus_.get_nhop_contextualized_embedding(args, edge_list_nhop, edge_type_nhop)
        file = args.data + '/ctx_2hop.pickle'

        with open(file, 'wb') as handle:
            pickle.dump(ctx_nhop_emb, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    if args.use_ctx:
        file = args.data + '/ctx.pickle'
        with open(file, 'rb') as handle:
            ctx_emb = pickle.load(handle)
        print('A', ctx_emb.size())

        file = args.data + '/ctx_2hop.pickle'
        with open(file, 'rb') as handle:
            ctx_nhop_emb = pickle.load(handle)
        print('B', len(ctx_nhop_emb))

        Corpus_.set_ctx_emb(ctx_emb.cuda())
        Corpus_.set_ctx_nhop_emb(ctx_nhop_emb)

    if args.mode == 'train_gat':
        train_gat(args, Corpus_, entity_embeddings, relation_embeddings, node_neighbors_2hop)
    elif args.mode == 'train_conv':
        train_conv(args, Corpus_, entity_embeddings, relation_embeddings)
    elif args.mode == 'eval':
        evaluate_conv(args, Corpus_.unique_entities_train, Corpus_, entity_embeddings, relation_embeddings)
