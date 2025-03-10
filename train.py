# coding: utf-8

import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import os
from glob import glob
import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
from data import ComparisonsDataset, CustomTransform
import logging
from datetime import date
from scripts.train_script import train

pd.options.mode.chained_assignment = None  # default='warn'


def arg_parse():
    """Parser for program arguments."""

    parser = argparse.ArgumentParser(description='Training subjective safety')
    parser.add_argument('--cuda', help="run with cuda", action='store_true')
    parser.add_argument('--cuda_id', help="gpu id", default=0, type=int)
    parser.add_argument('--comparisons', help="path to pickle comparisons DataFrame", default="comparisons_df.pickle", type=str)
    parser.add_argument('--dataset', help="dataset images directory path", default="images/", type=str)
    parser.add_argument('--max_epochs', help="maximum training epochs", default=1, type=int)
    parser.add_argument('--batch_size', help="batch size", default=32, type=int)
    parser.add_argument('--resume', help="resume training", action='store_true')
    parser.add_argument('--resume_checkpoint', help="resume training on checkpoint", type=str)
    parser.add_argument('--resume_realimages', '--r', help="resume training using real images", action='store_true')
    parser.add_argument('--epoch', help="epoch to load training", default=1, type=int)
    parser.add_argument('--model_dir', help="directory to load and save models", default='models/', type=str)
    parser.add_argument('--model', help="model to use, rcnn, sscnn or rsscnn", default='rcnn', type=str,
                        choices=['rsscnn',  # Classification + Ranking Loss
                                 'sscnn',   # Classification Loss Only
                                 'rcnn',    # Ranking Loss Only
                                 ])
    parser.add_argument('--backbone', help="backbone to use, alex or vgg or dense", default='alex', type=str,
                        choices=['alex', 'vgg', 'dense', 'resnet', 'deit_base', 'deit_small', 'deit_base_distilled'])
    parser.add_argument('--finetune', '--ft', help="finetune backbone", action='store_true')
    parser.add_argument('--lr_decay', help="use lr_decay", action='store_true')
    parser.add_argument('--ties', help="use ties from comparisons", action='store_true')
    parser.add_argument('--rank_w', help="nonties rank loss weight", default=1, type=float)
    parser.add_argument('--ties_w', help="ties rank loss weight", default=1, type=float)
    parser.add_argument('--ranking_margin', help="ranking loss margin term", default=0, type=float)
    parser.add_argument('--ranking_margin_ties', help="ranking loss margin term for ties", default=0, type=float)
    parser.add_argument('--seed', help="seed", default=5, type=int)

    parser.add_argument('--log_console', help="Log results to console.", action='store_true')
    parser.add_argument('--no-log_console', dest='log_console', action='store_false')
    parser.set_defaults(log_console=True)
    parser.add_argument('--log_wandb', help="Log results to WandB.", action='store_true')
    parser.add_argument('--no-log_wandb', dest='log_wandb', action='store_false')
    parser.set_defaults(log_wandb=True)
    parser.add_argument('--full_accuracy', help="Log full accuracy (including ties within margins).", action='store_true')

    return parser


def read_data(args):
    """
    Returns a cleaned pd.Dataframe with pairwise comparisons data.

        Parameters:
             args (): Program arguments

        Returns:
                comparisons_df (pd.DataFrame): DataFrame with pairwise comparisons data
    """
    # Read dataframe
    try:
        comparisons_df = pickle.load(open(args.comparisons, 'rb'))  # Read data
    except:
        comparisons_df = pd.read_pickle(args.comparisons)

    # Real Data
    if 'berlin' in args.comparisons:
        # Select only berlin images from the 'normal' set
        comparisons_df = comparisons_df[comparisons_df['dataset'] == 'berlin']
        # Use only images and score columns
        comparisons_df = comparisons_df[['score', 'image_l', 'image_r']]
        # Add image extension to image names
        comparisons_df['image_l'] = comparisons_df['image_l'].apply(lambda x: x + '.jpg')
        comparisons_df['image_r'] = comparisons_df['image_r'].apply(lambda x: x + '.jpg')

    # Mix of datasets (may contain synthetic and real data, or real data from different cities)
    elif 'mixed' in args.comparisons:
        pass

    # semi-Realistic Data (synthetic images)
    else:
        # Rename columns to match real dataset
        comparisons_df = comparisons_df.rename(columns={'scene_i': 'image_l', 'scene_j': 'image_r'})
        # Add image extension to image names
        comparisons_df['image_l'] = comparisons_df['image_l'].apply(lambda x: x + '.jpg')
        comparisons_df['image_r'] = comparisons_df['image_r'].apply(lambda x: x + '.jpg')

    # If not to include ties
    if not args.ties:
        comparisons_df = comparisons_df[comparisons_df['score'] != 0]  # Remove draws
        # Labels should start from 0in classification: [-1,+1]->[0,1]
        comparisons_df['score_classification'] = comparisons_df['score'].replace({-1: 0})

    else:
        # Labels should start from 0 in classification: [-1,0,+1]->[0,1,2]
        comparisons_df['score_classification'] = comparisons_df['score'] + 1

    return comparisons_df


def initialize_logging():
    """Initialize run logs."""
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s', filename=f'logs/{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.INFO)  # set the minimum level of message logging
    logger.info('HELLO')

    return logger


def initialize_wandb(args):
    """Initialize WandB run logs."""
    wandb.init(
        # set the wandb project where this run will be logged
        project="SubjectiveCyclingSafety",

        # track hyperparameters and run metadata
        config={
            'dataset': args.comparisons,
            'ties': args.ties,
            'ties_w': args.ties_w,
            'rank_w': args.rank_w,
            'rank_margin': args.ranking_margin,
            'rank_margin_ties': args.ranking_margin_ties,
            'seed': args.seed,
            'epochs': args.max_epochs,
            'batch_size': args.batch_size,
            'architecture_backbone': args.backbone,
            'architecture_model': args.model,
            'finetune_backbone': args.finetune,
            'learning_rate_decay': args.lr_decay,
            'resume': args.resume,
            'resume_epoch': args.epoch,
            'resume_realimages': args.resume_realimages,
            'checkpoint': os.path.join(args.model_dir, '{}'.format(args.resume_checkpoint)),
        }
    )

    wandb.define_metric("batch")
    wandb.define_metric("epoch")

    wandb.define_metric("accuracy_train", step_metric="batch")
    wandb.define_metric("accuracy_train", step_metric="epoch")
    wandb.define_metric("accuracy_validation", step_metric="epoch")
    wandb.define_metric("accuracy_test", step_metric="epoch")

    wandb.define_metric("loss_train", step_metric="epoch")
    wandb.define_metric("loss_validation", step_metric="epoch")
    wandb.define_metric("loss_test", step_metric="epoch")

    wandb.define_metric("max_accuracy_train", step_metric="epoch")
    wandb.define_metric("max_accuracy_validation", step_metric="epoch")
    wandb.define_metric("max_accuracy_test", step_metric="epoch")


if __name__ == '__main__':
    # =============================================================================================== #
    # INITIALIZATION
    # =============================================================================================== #
    args = arg_parse().parse_args()
    print(args, '\n')

#    args.ranking_margin_ties = args.ranking_margin

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Logging
    logger = initialize_logging()

    # WandB logging
    if args.log_wandb:
        initialize_wandb(args)
    
    # =============================================================================================== #
    # DATA
    # =============================================================================================== #
    print('Reading input data.')
    # Read data
    comparisons_df = read_data(args)
    print(comparisons_df)
    # Split data into train, validation, and test
    # Train: 0.7
    # Val:   0.1
    # Test:  0.2
    X_train, X_test = train_test_split(comparisons_df, test_size=0.2, random_state=args.seed)
    X_train, X_val  = train_test_split(X_train       , test_size=0.13, random_state=args.seed)
    print('Splits:', 
          '\n  -Train:', X_train.shape[0], '[{0:.2f}]'.format(X_train.shape[0]/comparisons_df.shape[0]),
          '\n  -Val:  ', X_val.shape[0], '[{0:.2f}]'.format(X_val.shape[0]/comparisons_df.shape[0]),
          '\n  -Test: ', X_test.shape[0], '[{0:.2f}]'.format(X_test.shape[0]/comparisons_df.shape[0]),
          )
    #print('Splits:',
    #      '\n  -Train:', X_train.score.value_counts(normalize=True).mul(100).round(1).astype(str) + '%',
    #      '\n  -Val:  ', X_val.score.value_counts(normalize=True).mul(100).round(1).astype(str) + '%',
    #      '\n  -Test: ', X_test.score.value_counts(normalize=True).mul(100).round(1).astype(str) + '%',
    #      )
    print('Initialize data & val sample loader.')
    transforms = transforms.Compose([CustomTransform(transforms.Resize(256)),
                                     CustomTransform(transforms.CenterCrop(224)),
                                     CustomTransform(transforms.ToTensor()),
                                     CustomTransform(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])),
                                     ])

    # =============================================================================================== #
    # DATA LOADING
    # =============================================================================================== #
    # Load the train dataset
    train_set = ComparisonsDataset(dataframe=X_train,
                                   root_dir=args.dataset, #'/home/mncosta/data/images/',#'/mnt/datasets/berlin_synthetic/images/',
                                   transform=transforms,
                                   logger=logger,)
    dataloader = torch.utils.data.DataLoader(train_set,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             #pin_memory=True,
                                             num_workers=4,
                                             drop_last=True)

    # Load the validation dataset
    val_set = ComparisonsDataset(dataframe=X_val,
                                 root_dir=args.dataset, #'/home/mncosta/data/images/',#'/mnt/datasets/berlin_synthetic/images/',
                                 transform=transforms,
                                 logger=logger,)
    val_loader = DataLoader(val_set, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=4, 
                            #pin_memory=True,
                            drop_last=True)

    # Load the test dataset
    test_set = ComparisonsDataset(dataframe=X_test,
                                  root_dir=args.dataset, #'/home/mncosta/data/images/',#'/mnt/datasets/berlin_synthetic/images/',
                                  transform=transforms,
                                  logger=logger,)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              batch_size=args.batch_size,
                                              shuffle=True, 
                                              #pin_memory=True,
                                              num_workers=4,
                                              drop_last=True)
    print()

    # =============================================================================================== #
    # MODEL
    # =============================================================================================== #
    # Define cpu/gpu device
    if args.cuda:
        device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print('Device:', device)
    
    print('Parsing model.')
    # Import model specific modules
    if args.backbone in ['deit_base', 'deit_small', 'deit_base_distilled']:
        from nets.transformer import Transformer as Net
    elif args.backbone in ['alex', 'vgg', 'dense', 'resnet']:
        from nets.cnn import CNN as Net
    else:
        raise Exception('Invalid model. To check available models run with -h.')
    # Define models available
    backbones = {
        'alex': models.alexnet,
        'vgg': models.vgg19,
        'dense': models.densenet121,
        'resnet': models.resnet50,
        'deit_base': torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True),
        'deit_small': torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True),
        'deit_base_distilled': torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224',
                                              pretrained=True)
    }

    # Parse model
    if args.backbone in ['deit_base', 'deit_small', 'deit_base_distilled']:
        net = Net(
            backbone=backbones[args.backbone],
            model=args.model,
            num_classes=3 if args.ties else 2,
            )
    elif args.backbone in ['alex', 'vgg', 'dense', 'resnet']:
        net = Net(backbone=backbones[args.backbone],
                  model=args.model,
                  finetune=args.finetune,
                  num_classes=3 if args.ties else 2,  # If ties are included, # of classes are 3
                  )
    else:
        raise Exception('Invalid model. To check available models run with -h.')

    # Resume training if requested
    if args.resume:
        print()
        print('Resuming training.')
        checkpoint_name = os.path.join(args.model_dir, '{}'.format(args.resume_checkpoint))
        print('Loading model:', checkpoint_name)
        
        net.load_state_dict(torch.load(checkpoint_name))
    print()

    # =============================================================================================== #
    # TRAINING
    # =============================================================================================== #
    print('Training:', wandb.run.name if args.log_wandb else '')
    train(device, net, dataloader, val_loader, test_loader, args, logger)
    
