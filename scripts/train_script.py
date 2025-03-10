import sys
import torch.nn as nn
import torch.optim as optim
import torch

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from random import randint
from timeit import default_timer as timer
import wandb

# from utils.ranking import *
from utils.log import log
from utils.losses import compute_loss
from utils.accuracy import RankAccuracy, RankAccuracy_withMargin, RankAccuracy_ties


def train(device, net, dataloader, val_loader, test_loader, args, logger):
    # =============================================================================================== #
    # TRAIN
    # =============================================================================================== #
    # Training subroutine
    def update(engine, data):
        if logger:
            start = timer()

        # Load input data
        input_left, input_right = data['image_l'], data['image_r']
        input_left, input_right = input_left.to(device), input_right.to(device)

        # Load label data
        label_r, label_c = data['score_r'], data['score_c']
        label_r, label_c = label_r.to(device), label_c.to(device)
        label_r, label_c = label_r.float(), label_c.float()
        labels = {'label_r': label_r, 'label_c': label_c}

        # Reset optimizer
        optimizer.zero_grad()

        # Forward pass the training sample
        forward_dict = net(input_left, input_right)

        # Compute loss
        loss = compute_loss(args, forward_dict, labels)

        # Backward step
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if logger:
            logger.info(f'TRAIN_STEP, {timer()-start:.4f}')

        # Ranking Model Only
        if args.model == 'rcnn':
            return {'loss': loss.item(),
                    'rank_left': forward_dict['left']['output'],
                    'rank_right': forward_dict['right']['output'],
                    'label': label_r
                    }

        # Classification Model Only
        elif args.model == 'sscnn':
            return {'loss': loss.item(),
                    'logits': forward_dict['logits']['output'],
                    'label': label_c.long(),
                    }

        # Classification + Ranking Model
        elif args.model == 'rsscnn':
            return {'loss': loss.item(),
                    'rank_left': forward_dict['left']['output'],
                    'rank_right': forward_dict['right']['output'],
                    'logits': forward_dict['logits']['output'],
                    'label_r': label_r,
                    'label_c': label_c
                    }

    # =============================================================================================== #
    # INFERENCE
    # =============================================================================================== #
    # Inference subroutine
    def inference(engine, data):
        with torch.no_grad():
            # Load input data
            input_left, input_right = data['image_l'], data['image_r']
            input_left, input_right = input_left.to(device), input_right.to(device)

            # Load label data
            label_r, label_c = data['score_r'], data['score_c']
            label_r, label_c = label_r.to(device), label_c.to(device)
            label_r, label_c = label_r.float(), label_c.float()
            labels = {'label_r': label_r, 'label_c': label_c}

            # Forward pass the sample
            forward_dict = net(input_left, input_right)

            # Compute loss
            loss = compute_loss(args, forward_dict, labels)

            # Ranking Model Only
            if args.model == 'rcnn':
                return {'loss': loss.item(),
                        'rank_left': forward_dict['left']['output'],
                        'rank_right': forward_dict['right']['output'],
                        'label': label_r
                        }

            # Classification Model Only
            elif args.model == 'sscnn':
                return {'loss': loss.item(),
                        'logits': forward_dict['logits']['output'],
                        'label': label_c.long(),
                        }

            # Classification + Ranking Model
            elif args.model == 'rsscnn':
                return {'loss': loss.item(),
                        'rank_left': forward_dict['left']['output'],
                        'rank_right': forward_dict['right']['output'],
                        'logits': forward_dict['logits']['output'],
                        'label_r': label_r,
                        'label_c': label_c
                        }

    # =============================================================================================== #
    # MODEL & OPTIMIZER
    # =============================================================================================== #
    # Define model, loss, optimizer and scheduler
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-09)
    if args.lr_decay:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.995, last_epoch=-1)
    else:
        scheduler = None

    # =============================================================================================== #
    # ENGINES
    # =============================================================================================== #
    # Engine specific parameters
    trainer = Engine(update)
    evaluator = Engine(inference)
    evaluator_test = Engine(inference)

    # =============================================================================================== #
    # METRICS
    # =============================================================================================== #
    # Attach losses and accuracies to trainer and evaluator
    for engine in [trainer, evaluator, evaluator_test]:
        # Ranking Model Only
        if args.model == 'rcnn':
            RunningAverage(output_transform=lambda x: x['loss'],
                           device=device).attach(engine, 'loss')
            if args.full_accuracy:
                # Ranking for non-ties, with margin
                RankAccuracy_withMargin(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label'],
                                                                    args.ranking_margin),
                                        device=device).attach(engine, 'acc')
                # Ranking for ties with margin
                if args.ties:
                    RankAccuracy_ties(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label'],
                                                                  args.ranking_margin),
                                      device=device).attach(engine, 'acc_ties')
            else:
                # Rank for non-ties, without margin
                RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']),
                             device=device).attach(engine, 'acc')

        # Classification Model Only
        elif args.model == 'sscnn':
            RunningAverage(output_transform=lambda x: x['loss'],
                           device=device).attach(engine, 'loss')
            RunningAverage(Accuracy(output_transform=lambda x: (x['logits'], x['label']))).attach(engine, 'acc')

        # Classification + Ranking Model
        elif args.model == 'rsscnn':
            # Loss
            RunningAverage(output_transform=lambda x: x['loss'],
                           device=device).attach(engine, 'loss')
            # Ranking accuracy
            if args.full_accuracy:
                # Ranking for non-ties, with margin
                RankAccuracy_withMargin(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label_r'],
                                                                    args.ranking_margin),
                                        device=device).attach(engine, 'acc')
                # Ranking for ties with margin
                if args.ties:
                    RankAccuracy_ties(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label_r'],
                                                                  args.ranking_margin),
                                      device=device).attach(engine, 'acc_ties')
            else:
                # Rank for non-ties, without margin
                RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label_r']),
                             device=device).attach(engine, 'acc')
            # Classification accuracy
            RunningAverage(Accuracy(output_transform=lambda x: (x['logits'], x['label_c']))).attach(engine, 'c_acc')

        else:
            raise Exception('Model type unknown: {}'.format(args.model))

    # =============================================================================================== #
    # LOG: EPOCH
    # =============================================================================================== #
    # Log training parameters after every epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        net.eval()
        evaluator.run(val_loader)
        evaluator_test.run(test_loader)
        trainer.state.metrics['val_acc'] = evaluator.state.metrics['acc']
        net.train()
        
        if hasattr(net, 'partial_eval'):
            net.partial_eval()

        metrics = {
            'accuracy_train': trainer.state.metrics['acc'],
            'accuracy_validation': evaluator.state.metrics['acc'],
            'accuracy_test': evaluator_test.state.metrics['acc'],
            'loss_train': trainer.state.metrics['loss'],
            'loss_validation': evaluator.state.metrics['loss'],
            'loss_test': evaluator_test.state.metrics['loss'],
            'time': f'{timer()-start_training:.3f}',
            'epoch': trainer.state.epoch,
            'iteration': trainer.state.iteration,
            'max_accuracy_validation': 0,  # placeholder for tracking epoch with the best validation accuracy
            'max_accuracy_train': 0,       # placeholder for tracking epoch with the best validation accuracy
            'max_accuracy_test': 0,        # placeholder for tracking epoch with the best validation accuracy
            }
        if args.full_accuracy and args.ties:
            if args.model == 'rcnn' or args.model == 'rsscnn':
                metrics.update({
                    'accuracy_train_ties': trainer.state.metrics['acc_ties'],
                    'accuracy_validation_ties': evaluator.state.metrics['acc_ties'],
                    'accuracy_test_ties': evaluator_test.state.metrics['acc_ties'],
                })
        # Ranking Model Only
        if args.model == 'rcnn':
            pass

        # Classification Model Only
        elif args.model == 'sscnn':
            pass

        # Classification + Ranking Model
        elif args.model == 'rsscnn':
            metrics.update({
                'c_accuracy_train': trainer.state.metrics['c_acc'],
                'c_accuracy_validation': evaluator.state.metrics['c_acc'],
                'c_accuracy_test': evaluator_test.state.metrics['c_acc'],
            })

        else:
            raise Exception('Model type unknown: {}'.format(args.model))

        log(args, metrics)

    # =============================================================================================== #
    # LOG: TRAIN
    # =============================================================================================== #
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(trainer):
        # Log training every 10th iteration
        if trainer.state.iteration % 10 == 0:
            metrics = {
                    'loss_train': trainer.state.metrics['loss'],
                    'time': f'{timer()-start_training:.3f}',
                    'epoch': trainer.state.epoch,
                    'iteration': trainer.state.iteration,
                }

            log(args, metrics)

    # =============================================================================================== #
    # CHECKPOINT
    # =============================================================================================== #
    # Model Checkpoint
    handler = ModelCheckpoint(args.model_dir,  # save model in directory
                              '{}_{}'.format(args.model, args.backbone), # model name
                              n_saved=10,
                              create_dir=True,
                              save_as_state_dict=True,
                              require_empty=False,
                              score_function=lambda engine: engine.state.metrics['val_acc'],  # save validation accuracy
                              global_step_transform=lambda *_: trainer.state.epoch,
                              )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, 
                              handler, 
                              {'model': net})

    # Best Model Checkpoint
    handler_best = ModelCheckpoint(args.model_dir,  # save model in directory
                                   '{}'.format(wandb.run.name),
                                   n_saved=1,
                                   create_dir=True,
                                   save_as_state_dict=True,
                                   require_empty=False,
                                   score_function=lambda engine: engine.state.metrics['val_acc'],  # save val accuracy
                                   global_step_transform=lambda *_: trainer.state.epoch,
                                   )
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              handler_best,
                              {'model': net})

    handler_last = ModelCheckpoint(args.model_dir,  # save model in directory
                                   '{}'.format(wandb.run.name),
                                   n_saved=1,
                                   create_dir=True,
                                   save_as_state_dict=True,
                                   require_empty=False,
#                                   score_function=lambda engine: engine.state.metrics['val_acc'],  # save val accuracy
                                   global_step_transform=lambda *_: trainer.state.epoch,
                                   )
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              handler_last,
                              {'model': net})


    # =============================================================================================== #
    # RESUME TRAINING
    # =============================================================================================== #
    # Set initial epoch to resume training on and maximum epochs to run
    if args.resume:
        def start_epoch(engine):
            engine.state.epoch = args.epoch
        def max_epoch(engine):
            engine.state.max_epochs = args.max_epochs
        trainer.add_event_handler(Events.STARTED, start_epoch)
        evaluator.add_event_handler(Events.STARTED, start_epoch)
        evaluator_test.add_event_handler(Events.STARTED, start_epoch)
        trainer.add_event_handler(Events.STARTED, max_epoch)
        evaluator.add_event_handler(Events.STARTED, max_epoch)
        evaluator_test.add_event_handler(Events.STARTED, max_epoch)
    
    # =============================================================================================== #
    # RUN
    # =============================================================================================== #
    if logger: start_training = timer()
    trainer.run(dataloader, 
                max_epochs=args.max_epochs, 
                )
    
    # Close WandB
    if args.log_wandb:
        wandb.finish()

    
if __name__ == '__main__':
    from nets.cnn import MyCnn
    import torchvision.models as models

    net = MyCnn(models.resnet50)
    x = torch.randn([3, 244, 244]).unsqueeze(0)
    y = torch.randn([3, 244, 244]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)
