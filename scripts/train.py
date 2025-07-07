import os
import sys
import json
import argparse
import collections
import torch
import torch.optim as optim
import numpy as np
# import wandb
import random
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.dataset import ScannetQADataset, ScannetQADatasetConfig
from lib.solver import Solver
from lib.config import CONF
from lib.loss_helper import get_loss
from models.qa_module import ScanQA

project_name = "ScanQA_v1.0"
SCANQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_train.json"))) 
SCANQA_VAL = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_val.json")))

# constants
DC = ScannetQADatasetConfig()

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. XYZ_COLOR", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    # Training
    parser.add_argument("--cur_criterion", type=str, default="answer_acc_at1", help="data augmentation type")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000) # 5000
    parser.add_argument("--train_num_scenes", type=int, default=-1, help="Number of train scenes [default: -1]")
    parser.add_argument("--val_num_scenes", type=int, default=-1, help="Number of val scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Optimizer   
    parser.add_argument("--optim_name", type=str, help="optimizer name", default="adam")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lr", type=float, help="initial learning rate", default=5e-4)
    parser.add_argument("--adam_beta1", type=float, help="beta1 hyperparameter for the Adam optimizer", default=0.9)
    parser.add_argument("--adam_beta2", type=float, help="beta2 hyperparameter for the Adam optimizer", default=0.999) # 0.98
    parser.add_argument("--adam_epsilon", type=float, help="epsilon hyperparameter for the Adam optimizer", default=1e-8) # 1e-9
    parser.add_argument("--amsgrad", action="store_true", help="Use amsgrad for Adam")
    parser.add_argument('--lr_decay_step', nargs='+', type=int, default=[100, 200]) # 15
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of learning rate", default=0.2) # 01, 0.2
    parser.add_argument('--bn_decay_step', type=int, default=20)
    parser.add_argument("--bn_decay_rate", type=float, help="bn rate", default=0.5)
    parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm ", default=1.0)
    # Data
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use data augmentations.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    # Model
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size[default: 256]")
    ## pointnet & votenet & proposal
    parser.add_argument("--vote_radius", type=float, help="", default=0.3) # 5
    parser.add_argument("--vote_nsample", type=int, help="", default=16) # 512
    parser.add_argument("--pointnet_width", type=int, help="", default=1)
    parser.add_argument("--pointnet_depth", type=int, help="", default=2)
    parser.add_argument("--seed_feat_dim", type=int, help="", default=256) # or 288
    parser.add_argument("--proposal_size", type=int, help="", default=128)    
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--use_seed_lang", action="store_true", help="Fuse seed feature and language feature.")    
    ## module option
    parser.add_argument("--no_object_mask", action="store_true", help="objectness_mask for qa")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_answer", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    # Pretrain
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    # Loss
    parser.add_argument("--vote_loss_weight", type=float, help="vote_net loss weight", default=1.0)
    parser.add_argument("--objectness_loss_weight", type=float, help="objectness loss weight", default=0.5)
    parser.add_argument("--box_loss_weight", type=float, help="box loss weight", default=1.0)
    parser.add_argument("--sem_cls_loss_weight", type=float, help="sem_cls loss weight", default=0.1)
    parser.add_argument("--ref_loss_weight", type=float, help="reference loss weight", default=0.1)
    parser.add_argument("--lang_loss_weight", type=float, help="language loss weight", default=0.1)
    parser.add_argument("--answer_loss_weight", type=float, help="answer loss weight", default=0.1)  
    # Answer
    parser.add_argument("--answer_cls_loss", type=str, help="answer classifier loss", default="bce") # ce, bce
    parser.add_argument("--answer_max_size", type=int, help="maximum size of answer candicates", default=-1) # default use all
    parser.add_argument("--answer_min_freq", type=int, help="minimum frequence of answers", default=1)
    parser.add_argument("--answer_pdrop", type=float, help="dropout_rate of answer_cls", default=0.3)
    # Question
    parser.add_argument("--tokenizer_name", type=str, help="Pretrained tokenizer name", default="spacy_blank") # or bert-base-uncased, bert-large-uncased-whole-word-masking, distilbert-base-uncased
    parser.add_argument("--lang_num_layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--lang_use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze BERT ebmedding model")
    parser.add_argument("--finetune_bert_last_layer", action="store_true", help="Finetue BERT last layer")
    parser.add_argument("--lang_pdrop", type=float, help="dropout_rate of lang_cls", default=0.3)
    ## MCAN
    parser.add_argument("--mcan_pdrop", type=float, help="", default=0.1)
    parser.add_argument("--mcan_flat_mlp_size", type=int, help="", default=256) # mcan: 512
    parser.add_argument("--mcan_flat_glimpses", type=int, help="", default=1)
    parser.add_argument("--mcan_flat_out_size", type=int, help="", default=512) # mcan: 1024
    parser.add_argument("--mcan_num_heads", type=int, help="", default=8)
    parser.add_argument("--mcan_num_layers", type=int, help="", default=2) # mcan: 6

    # AL modified
    parser.add_argument("--AL_mode", type=str, help="Active selection", default="random")
    parser.add_argument("--AL_oracle", action="store_true", help="Active exclusion")

    args = parser.parse_args()
    return args
    

def get_answer_cands(args, scanqa):
    answer_counter = sum([data["answers"] for data in scanqa["train"]], [])
    answer_counter = collections.Counter(sorted(answer_counter))
    num_all_answers = len(answer_counter)
    answer_max_size = args.answer_max_size
    if answer_max_size < 0:
        answer_max_size = len(answer_counter)
    answer_counter = dict([x for x in answer_counter.most_common()[:answer_max_size] if x[1] >= args.answer_min_freq])
    print("using {} answers out of {} ones".format(len(answer_counter), num_all_answers))    
    answer_cands = sorted(answer_counter.keys())
    return answer_cands, answer_counter


def get_dataloader(args, scanqa, all_scene_list, split, config, augment, bs_override=None):
    answer_cands, answer_counter = get_answer_cands(args, scanqa)
    config.num_answers = len(answer_cands)

    if 'bert-' in args.tokenizer_name: 
        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None

    dataset = ScannetQADataset(
        scanqa=scanqa[split], 
        scanqa_all_scene=all_scene_list, 
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        tokenizer=tokenizer,
        augment=augment,
        debug=args.debug,
    )
    bs = args.batch_size if bs_override is None else bs_override
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    return dataset, dataloader


def get_model(args, config):
    if "bert-" in args.tokenizer_name:
        from transformers import AutoConfig
        bert_model_name = args.tokenizer_name
        bert_config = AutoConfig.from_pretrained(bert_model_name)
        if hasattr(bert_config, "hidden_size"):
            lang_emb_size = bert_config.hidden_size
        else:
            # for distllbert
            lang_emb_size = bert_config.dim
    else:
        bert_model_name = None
        lang_emb_size = 300 # glove emb_size

    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)

    model = ScanQA(
        num_answers=config.num_answers,
        # proposal
        input_feature_dim=input_channels,            
        num_object_class=config.num_class, 
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals, 
        seed_feat_dim=args.seed_feat_dim,
        proposal_size=args.proposal_size,
        pointnet_width=args.pointnet_width,
        pointnet_depth=args.pointnet_depth,        
        vote_radius=args.vote_radius, 
        vote_nsample=args.vote_nsample,            
        # qa
        #answer_cls_loss="ce",
        answer_pdrop=args.answer_pdrop,
        mcan_num_layers=args.mcan_num_layers,
        mcan_num_heads=args.mcan_num_heads,
        mcan_pdrop=args.mcan_pdrop,
        mcan_flat_mlp_size=args.mcan_flat_mlp_size, 
        mcan_flat_glimpses=args.mcan_flat_glimpses,
        mcan_flat_out_size=args.mcan_flat_out_size,
        # lang
        lang_use_bidir=args.lang_use_bidir,
        lang_num_layers=args.lang_num_layers,
        lang_emb_size=lang_emb_size,
        lang_pdrop=args.lang_pdrop,
        bert_model_name=bert_model_name,
        freeze_bert=args.freeze_bert,
        finetune_bert_last_layer=args.finetune_bert_last_layer,
        # common
        hidden_size=args.hidden_size,
        # option
        use_object_mask=(not args.no_object_mask),
        use_lang_cls=(not args.no_lang_cls),
        use_reference=(not args.no_reference),
        use_answer=(not args.no_answer),            
    )

    # to CUDA
    model = model.cuda()
    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args, DC)
    #wandb.watch(model, log_freq=100)

    if args.optim_name == 'adam':
        model_params = [{"params": model.parameters()}]
        optimizer = optim.Adam(
            model_params,
            lr=args.lr, 
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
            weight_decay=args.wd, 
            amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                weight_decay=args.wd, 
                                amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw_cb':
        from transformers import AdamW
        optimizer = AdamW(model.parameters(), lr=args.lr, 
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                weight_decay=args.wd)
    else:
        raise NotImplementedError

    print('set optimizer...')
    print(optimizer)
    print()

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    loss_weights = {}
    loss_weights['vote_loss']       = args.vote_loss_weight
    loss_weights['objectness_loss'] = args.objectness_loss_weight 
    loss_weights['box_loss']        = args.box_loss_weight
    loss_weights['sem_cls_loss']    = args.sem_cls_loss_weight
    loss_weights['ref_loss']        = args.ref_loss_weight
    loss_weights['lang_loss']       = args.lang_loss_weight
    loss_weights['answer_loss']     = args.answer_loss_weight

    solver = Solver(
        model=model, 
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        cur_criterion=args.cur_criterion,
        detection=not args.no_detection,
        use_reference=not args.no_reference, 
        use_answer=not args.no_answer,
        use_lang_classifier=not args.no_lang_cls,
        max_grad_norm=args.max_grad_norm,
        lr_decay_step=args.lr_decay_step,
        lr_decay_rate=args.lr_decay_rate,
        bn_decay_step=args.bn_decay_step,
        bn_decay_rate=args.bn_decay_rate,
        loss_weights=loss_weights,
    )
    num_params = get_num_params(model)

    return solver, num_params, root, stamp

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    answer_vocab = train_dataset.answer_counter
    with open(os.path.join(root, "answer_vocab.json"), "w") as f:
        json.dump(answer_vocab, f, indent=4)        



def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list

def get_scanqa(scanqa_train, scanqa_val, train_num_scenes, val_num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanqa_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanqa_val])))

    # set train_num_scenes
    if train_num_scenes <= -1: 
        train_num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= train_num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:train_num_scenes]

    # filter data in chosen scenes
    new_scanqa_train = []
    for data in scanqa_train:
        if data["scene_id"] in train_scene_list:
            new_scanqa_train.append(data)
    
    # experiment
    # num_to_keep = len(new_scanqa_train) // 2
    # selected_indices = random.sample(range(len(new_scanqa_train)), num_to_keep)
    # new_scanqa_train = [new_scanqa_train[i] for i in selected_indices]

    # set val_num_scenes
    if val_num_scenes <= -1: 
        val_num_scenes = len(val_scene_list)
    else:
        assert len(val_scene_list) >= val_num_scenes

    # slice val_scene_list
    val_scene_list = val_scene_list[:val_num_scenes]        

    new_scanqa_val = []
    for data in scanqa_val:
        if data["scene_id"] in val_scene_list:
            new_scanqa_val.append(data)

    # new_scanqa_val = scanqa_val[0:4]  # debugging

    # all scanqa scene
    all_scene_list = train_scene_list + val_scene_list
    print("train on {} samples and val on {} samples".format(len(new_scanqa_train), len(new_scanqa_val)))
    return new_scanqa_train, new_scanqa_val, all_scene_list


def calc_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropy

def calc_entropy_2D(logit_matrix):
    res = []
    for logits in logit_matrix:
        res.append(calc_entropy(logits))
    return res

def random_choosing(unlabeled_data):
    if len(unlabeled_data) > 1000:
        new_labeled_data = random.sample(unlabeled_data, 1000)
    else:
        new_labeled_data = unlabeled_data
    return new_labeled_data

def entropy_feed(solver, dataloader):
    solver._set_phase("val")
    question_ids = []
    lang_entropy = []
    ans_entropy = []
    print("[AL] Inference on unlabeled dataset ...")
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if type(data_dict[key]) is dict:
                data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
            elif type(data_dict[key]) is not list:
                data_dict[key] = data_dict[key].cuda()

        # with torch.autograd.set_detect_anomaly(True):
        with torch.no_grad():  # forward
            data_dict = solver._forward(data_dict)
            question_ids += data_dict["question_id_raw"]
            lang_entropy += calc_entropy_2D(data_dict["lang_scores"])
            ans_entropy += calc_entropy_2D(data_dict["answer_scores"])
    
    solver._set_phase("train")
    pairs = list(zip(question_ids, lang_entropy, ans_entropy))
    # pairs = sorted(pairs, key=lambda x: x[1] + x[2], reverse=True)
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
    return pairs

def entropy(solver, unlabeled_dataloader, unlabeled_data):  # entropy
    new_labeled_qid = entropy_feed(solver, unlabeled_dataloader)[:1000]
    new_labeled_qid = [_[0] for _ in new_labeled_qid]
    new_labeled_data = [_ for _ in unlabeled_data if _['question_id'] in new_labeled_qid]
    print(f"[AL] Label {len(new_labeled_data)} data using entropy.")
    return new_labeled_data


def logmeanexp(logits, dim=0, keepdim=False):
    return (logits - F.log_softmax(logits, dim=dim)).mean(dim=dim, keepdim=keepdim) \
        - torch.log(torch.tensor(logits.size(dim), device=logits.device))

def calc_infogain_2D(logit_matrix):
    res = []
    for logits in logit_matrix:
        logpy = logmeanexp(logits, dim=0, keepdim=True)
        infogain = (torch.exp(logits) * (logits - logpy)).mean(dim=0).sum(dim=0)
        res.append(infogain)
    return res

def infogain_feed(solver, dataloader):
    solver._set_phase("val")
    question_ids = []
    lang_entropy = []
    ans_entropy = []
    print("[AL] Inference on unlabeled dataset ...")
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if type(data_dict[key]) is dict:
                data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
            elif type(data_dict[key]) is not list:
                data_dict[key] = data_dict[key].cuda()

        # with torch.autograd.set_detect_anomaly(True):
        with torch.no_grad():  # forward
            data_dict = solver._forward(data_dict)
            question_ids += data_dict["question_id_raw"]
            lang_entropy += calc_infogain_2D(data_dict["lang_scores"])
            ans_entropy += calc_infogain_2D(data_dict["answer_scores"])
    
    solver._set_phase("train")
    pairs = list(zip(question_ids, lang_entropy, ans_entropy))
    pairs = sorted(pairs, key=lambda x: x[1] + x[2], reverse=True)
    return pairs

def infogain(solver, unlabeled_dataloader, unlabeled_data):  # information gain
    new_labeled_qid = infogain_feed(solver, unlabeled_dataloader)[:1000]
    new_labeled_qid = [_[0] for _ in new_labeled_qid]
    new_labeled_data = [_ for _ in unlabeled_data if _['question_id'] in new_labeled_qid]
    print(f"[AL] Label {len(new_labeled_data)} data using infogain.")
    return new_labeled_data


def calc_variance(AL_Phi, logits):
    with torch.no_grad():
        logits = logits.cuda()
        probs = F.softmax(logits, dim=-1).cuda()  # probs: N, AL_Phi: N x D
        _, res = torch.linalg.slogdet(torch.cov(AL_Phi.T, aweights=probs))  # cov: D x D = 768 x 768
    return res

def calc_variance_2D(AL_Phi, logit_matrix):
    res = []
    for logits in logit_matrix:
        res.append(calc_variance(AL_Phi, logits))
    return res

def variance_feed(solver, dataloader, AL_Phi):
    solver._set_phase("val")
    question_ids = []
    ans_variance = []
    print("[AL] Inference on unlabeled dataset ...")
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if type(data_dict[key]) is dict:
                data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
            elif type(data_dict[key]) is not list:
                data_dict[key] = data_dict[key].cuda()

        # with torch.autograd.set_detect_anomaly(True):
        with torch.no_grad():  # forward
            data_dict = solver._forward(data_dict)
            question_ids += data_dict["question_id_raw"]
            ans_variance += calc_variance_2D(AL_Phi, data_dict["answer_scores"])
    
    solver._set_phase("train")
    pairs = list(zip(question_ids, ans_variance))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs

def variance(solver, unlabeled_dataloader, unlabeled_data, AL_Phi, lim):  # variance
    new_labeled_qid = variance_feed(solver, unlabeled_dataloader, AL_Phi)[:lim]
    new_labeled_qid = [_[0] for _ in new_labeled_qid]
    new_labeled_data = [_ for _ in unlabeled_data if _['question_id'] in new_labeled_qid]
    print(f"[AL] Label {len(new_labeled_data)} data using variance (lim = {lim}).")
    return new_labeled_data

def excl_feed(args, solver, dataloader, AL_Phi):
    solver._set_phase("val")
    question_ids = []
    ans_variance = []
    ans_loss = []
    print("[AL] Inference on labeled dataset ...")
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if type(data_dict[key]) is dict:
                data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
            elif type(data_dict[key]) is not list:
                data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():  # forward
            data_dict = solver._forward(data_dict)
            question_ids += data_dict["question_id_raw"]
            ans_variance += calc_variance_2D(AL_Phi, data_dict["answer_scores"])
            _, data_dict = get_loss(
                data_dict=data_dict, 
                config=DC, 
                detection=True,
                use_reference=(not args.no_reference), 
                use_lang_classifier=(not args.no_lang_cls),
                use_answer=(not args.no_answer),
            )
            ans_loss.append(data_dict['answer_loss'].item())
            # ans_loss.append(data_dict['loss'].item())
    
    solver._set_phase("train")
    return question_ids, ans_variance, ans_loss

def excl(args, solver, labeled_dataloader, labeled_data, AL_Phi):
    question_ids, variance, loss = excl_feed(args, solver, labeled_dataloader, AL_Phi)
    with torch.no_grad():
        variance = torch.tensor(variance, dtype=torch.float32)
        loss = torch.tensor(loss, dtype=torch.float32)
        variance_mean = torch.mean(variance)
        variacne_dev = torch.std(variance)
        loss_mean = torch.mean(loss)
        loss_dev = torch.std(loss)
        cond = (variance < variance_mean - variacne_dev) & (loss > loss_mean + loss_dev)
        # cond = (variance < variance_mean + variacne_dev) & (loss > loss_mean - loss_dev * 3)
        indices = torch.where(cond)[0]
    # if len(indices) > 500:
    #     indices = torch.tensor(random.sample(indices.tolist(), 500))
    indices = [int(idx.item()) for idx in indices]
    excluded_qid = [question_ids[idx] for idx in indices]
    excluded_data = [_ for _ in labeled_data if _['question_id'] in excluded_qid]
    print(f"[AL] Excluded {len(excluded_data)} data from labeled dataset.")
    return excluded_data


def train(args):
    # WandB init
    # wandb.init(project=project_name, config=args)

    # init training dataset
    print("preparing data...")
    scanqa_train, scanqa_val, all_scene_list = get_scanqa(SCANQA_TRAIN, SCANQA_VAL, args.train_num_scenes, args.val_num_scenes)
    scanqa_labeled = random.sample(scanqa_train, 1000)
    scanqa_unlabeled = [_ for _ in scanqa_train if _ not in scanqa_labeled]
    scanqa = {
        "train": scanqa_train,
        "val": scanqa_val,
        "labeled": scanqa_labeled,
        "unlabeled": scanqa_unlabeled
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanqa, all_scene_list, "labeled", DC, not args.no_augment)
    val_dataset, val_dataloader = get_dataloader(args, scanqa, all_scene_list, "val", DC, False)
    print("train on {} samples and val on {} samples".format(len(train_dataset), len(val_dataset)))

    # AL modified: calculate \Phi
    if args.AL_mode == 'variance' or args.AL_oracle:
        if os.path.exists('AL_Phi_cache.pt'):
            print('[AL] Using cached AL_Phi.')
            with open('AL_Phi_cache.pt', 'rb') as f:
                AL_Phi = torch.load(f)
        else:
            answer_cands, _ = get_answer_cands(args, scanqa=scanqa)

            from transformers import BertTokenizer, BertModel
            AL_BERT_MODEL = 'bert-base-uncased'
            print(f"[AL] Loading {AL_BERT_MODEL} ...")
            tokenizer = BertTokenizer.from_pretrained(AL_BERT_MODEL)
            model = BertModel.from_pretrained(AL_BERT_MODEL).to(f'cuda:{args.gpu}')

            print("[AL] Tokenizing ...")
            tokenized_texts = [tokenizer.encode(text, return_tensors='pt') for text in answer_cands]

            print(f"[AL] Encoding using {AL_BERT_MODEL} ...")
            AL_Phi = []
            for text in tqdm(tokenized_texts):
                with torch.no_grad():
                    outputs = model(text.to(model.device)).last_hidden_state
                    AL_Phi.append(outputs[:, 0, :].squeeze().detach())
            AL_phi_norm = [torch.norm(AL_phi, p=2) for AL_phi in AL_Phi]
            AL_phi_norm = torch.tensor(AL_phi_norm).cuda()
            AL_Phi = torch.stack(AL_Phi).cuda()
            print(f"[AL] Semantic matrix computed, size = {AL_Phi.shape}")

            with open('AL_Phi_cache.pt', 'wb') as f:
                torch.save(AL_Phi, f)

    if args.AL_oracle:
        SCANQAR_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANQAR, "ScanQA_Refined_v3.0_train.json")))
        qar_dict = {}
        for qa in SCANQAR_TRAIN:
            qar_dict[qa['question_id']] = qa

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root, stamp = get_solver(args, dataloader)
    # if stamp:
    #     wandb.run.name = stamp
    #     wandb.run.save()

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    
    # AL, moved from solver.py
    solver._start()
    # setting
    solver.epoch = args.epoch
    solver.verbose = args.verbose

    solver._total_iter["train"] = len(solver.dataloader["train"]) * args.epoch
    solver._total_iter["val"] = len(solver.dataloader["val"]) * solver.val_step

    for epoch_id in range(args.epoch):
        try:
            # AL modified
            if len(scanqa["unlabeled"]) > 0:
                _, unlabeled_dataloader = get_dataloader(args, scanqa, all_scene_list, "unlabeled", DC, not args.no_augment)
                if args.AL_mode == 'random':
                    new_labeled_data = random_choosing(scanqa["unlabeled"])
                elif args.AL_mode == 'entropy':
                    new_labeled_data = entropy(solver, unlabeled_dataloader, scanqa["unlabeled"])
                elif args.AL_mode == 'infogain':
                    new_labeled_data = infogain(solver, unlabeled_dataloader, scanqa["unlabeled"])
                elif args.AL_mode == 'variance':
                    if len(scanqa["unlabeled"]) > 100:
                        new_labeled_data = variance(solver, unlabeled_dataloader, scanqa["unlabeled"], AL_Phi, 1000)
                    else:
                        new_labeled_data = []
                else:  # not implemented
                    print(f"[AL] Strategy {args.AL_mode} is not implemented, using random instead.")
                    new_labeled_data = random_choosing(scanqa["unlabeled"])
                scanqa_labeled += new_labeled_data
                scanqa_unlabeled = [_ for _ in scanqa_unlabeled if _ not in new_labeled_data]
                scanqa["labeled"] = scanqa_labeled
                scanqa["unlabeled"] = scanqa_unlabeled
                labeled_dataset, labeled_dataloader = get_dataloader(args, scanqa, all_scene_list, "labeled", DC, not args.no_augment, bs_override=1)

                # if args.AL_exclusion and len(scanqa_labeled) > 20000 and epoch_id < 27:  # try 15000
                if args.AL_oracle and epoch_id in [0, 5, 10, 20, 25]:
                    excluded_data = excl(args, solver, labeled_dataloader, scanqa['labeled'], AL_Phi)
                    print(f'Excluded data ({len(excluded_data)}): ')
                    print(excluded_data)
                    scanqa_labeled = [_ for _ in scanqa_labeled if _ not in excluded_data]
                    cnt_relabel = 0
                    for _ in excluded_data:
                        if _['question_id'] in qar_dict.keys():
                            scanqa_labeled.append(qar_dict[_['question_id']])
                            cnt_relabel += 1
                    print(f'[AL] Relabeled {cnt_relabel} data.')
                    scanqa["labeled"] = scanqa_labeled
                    scanqa["unlabeled"] = scanqa_unlabeled

                train_dataset, train_dataloader = get_dataloader(args, scanqa, all_scene_list, "labeled", DC, not args.no_augment)
                dataloader["train"] = train_dataloader
                solver._total_iter["train"] = len(solver.dataloader["train"]) * args.epoch
                solver._total_iter["val"] = len(solver.dataloader["val"]) * solver.val_step
                print("[AL @ epoch {}] train on {} samples and val on {} samples".format(epoch_id, len(train_dataset), len(val_dataset)))


            solver._log("epoch {} starting...".format(epoch_id + 1))
            # feed
            print(f"Length of dataloader: {len(solver.dataloader['train'])}")
            solver._feed(solver.dataloader["train"], "train", epoch_id)

            solver._log("saving last models...\n")
            model_root = os.path.join(CONF.PATH.OUTPUT, solver.stamp)
            torch.save(solver.model.state_dict(), os.path.join(model_root, "model_last.pth"))
            
            # update lr scheduler
            if solver.lr_scheduler:
                print("update learning rate --> {}\n".format(solver.lr_scheduler.get_lr()))
                solver.lr_scheduler.step()

            # update bn scheduler
            if solver.bn_scheduler:
                print("update batch normalization momentum --> {}\n".format(solver.bn_scheduler.lmbd(solver.bn_scheduler.last_epoch)))
                solver.bn_scheduler.step()
            
        except KeyboardInterrupt:
            # finish training
            solver._finish(epoch_id)
            exit()

    # finish training
    solver._finish(epoch_id)

if __name__ == "__main__":
    args = parse_option()
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
    
