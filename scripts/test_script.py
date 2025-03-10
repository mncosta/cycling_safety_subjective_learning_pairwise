import os
from os import path
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from utils.log import log
from utils.accuracy import RankAccuracy, RankAccuracy_withMargin, RankAccuracy_ties



def test(device, net, dataloader, args, logger):
    # =============================================================================================== #
    # INFERENCE SUBROUTINE
    # =============================================================================================== #
    def inference(engine, data):
        with torch.no_grad():
            # Load input data
            input_left, input_right = data['image_l'], data['image_r']
            input_left, input_right = input_left.to(device), input_right.to(device)

            # Load input data
            input_left_name, input_right_name = data['image_l_name'], data['image_r_name']

            # Load label data
            label_r, label_c = data['score_r'], data['score_c']
            label_r, label_c = label_r.to(device), label_c.to(device)
            label_r, label_c = label_r.float(), label_c.float()
            labels = {'label_r': label_r, 'label_c': label_c}

            # Forward pass the sample
            forward_dict = net(input_left, input_right)

            # Ranking Model Only
            if args.model == 'rcnn':
                rank_left = forward_dict['left']['output'].squeeze().cpu().detach().numpy()
                rank_right = forward_dict['right']['output'].squeeze().cpu().detach().numpy()
                forward_pass = {'rank_left': rank_left,
                                'rank_right': rank_right,
                                }
                returnable_dict = {
                    'rank_left': forward_dict['left']['output'],
                    'rank_right': forward_dict['right']['output'],
                    'label_r': label_r,
                }

            # Classification Model Only
            elif args.model == 'sscnn':
                logits = forward_dict['logits']['output'].cpu().detach().numpy()
                if args.ties:
                    forward_pass = {'logits_l': logits[:, 0],
                                    'logits_0': logits[:, 1],
                                    'logits_r': logits[:, 2],
                                    }
                else:
                    forward_pass = {'logits_l': logits[:, 0],
                                    'logits_r': logits[:, 1],
                                    }
                returnable_dict = {
                    'logits': forward_dict['logits']['output'],
                    'label_c': label_c
                }

            # Classification + Ranking Model
            elif args.model == 'rsscnn':
                rank_left = forward_dict['left']['output'].squeeze().cpu().detach().numpy()
                rank_right = forward_dict['right']['output'].squeeze().cpu().detach().numpy()
                logits = forward_dict['logits']['output'].cpu().detach().numpy()

                if args.ties:
                    forward_pass = {'rank_left': rank_left,
                                    'rank_right': rank_right,
                                    'logits_l': logits[:, 0],
                                    'logits_0': logits[:, 1],
                                    'logits_r': logits[:, 2],
                                    }
                else:
                    forward_pass = {'rank_left': rank_left,
                                    'rank_right': rank_right,
                                    'logits_l': logits[:, 0],
                                    'logits_r': logits[:, 1],
                                    }
                returnable_dict = {
                    'rank_left': forward_dict['left']['output'],
                    'rank_right': forward_dict['right']['output'],
                    'logits': forward_dict['logits']['output'],
                    'label_r': label_r,
                    'label_c': label_c
                }

            output_dict = {
                'image_left': input_left_name,
                'image_right': input_right_name,
                'label_r': data['score_r'],
                'label_c': data['score_c']
            }
            output_dict.update(forward_pass)
            pd.DataFrame(output_dict).to_pickle(path.join('outputs', '{}_{}'.format(path.basename(args.load_model), engine.state.iteration, )+'.pkl'))
            pbar.update(1)

            return returnable_dict

    # =============================================================================================== #
    # MODEL
    # =============================================================================================== #
    # Define model
    net = net.to(device)

    # Engine specific parameters
    evaluator = Engine(inference)

    # =============================================================================================== #
    # LOG: EPOCH
    # =============================================================================================== #
    # Log training parameters after every epoch

    @evaluator.on(Events.COMPLETED)
    def log_validation_results(evaluator):
        metrics = {
            'accuracy_validation': evaluator.state.metrics['acc'],
            'epoch': evaluator.state.epoch,
            'iteration': evaluator.state.iteration,
        }
        if args.full_accuracy and args.ties and args.model != 'sscnn':
            metrics.update({
                'accuracy_validation_ties': evaluator.state.metrics['acc_ties'],
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
                'c_accuracy_validation': evaluator.state.metrics['c_acc'],
            })

        else:
            raise Exception('Model type unknown: {}'.format(args.model))

        log(args, metrics)

    # @evaluator.on(Events.COMPLETED)
    # def log_validation_results(evaluator):
    #     metrics = {
    #         'accuracy_validation': evaluator.state.metrics['acc'],
    #         'epoch': evaluator.state.epoch,
    #         'iteration': evaluator.state.iteration,
    #     }
    #     if args.full_accuracy and args.ties:
    #         metrics.update({
    #             'accuracy_validation_ties': evaluator.state.metrics['acc_ties'],
    #         })
    #     # Ranking Model Only
    #     if args.model == 'rcnn':
    #         pass
    #
    #     # Classification Model Only
    #     elif args.model == 'sscnn':
    #         pass
    #
    #     # Classification + Ranking Model
    #     elif args.model == 'rsscnn':
    #         metrics.update({
    #             'c_accuracy_validation': evaluator.state.metrics['c_acc'],
    #         })
    #
    #     else:
    #         raise Exception('Model type unknown: {}'.format(args.model))
    #
    #     log(args, metrics)


    # =============================================================================================== #
    # METRICS
    # =============================================================================================== #
    # Attach losses and accuracies to trainer and evaluator

    for engine in [evaluator]:
        # Ranking Model Only
        if args.model == 'rcnn':
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

        # Classification Model Only
        elif args.model == 'sscnn':
            RunningAverage(Accuracy(output_transform=lambda x: (x['logits'], x['label_c']))).attach(engine, 'acc')

        # Classification + Ranking Model
        elif args.model == 'rsscnn':
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
    # for engine in [evaluator]:
    #     # Ranking Model Only
    #     if args.model == 'rcnn':
    #         if args.full_accuracy:
    #             # Ranking for non-ties, with margin
    #             RankAccuracy_withMargin(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label'],
    #                                                                 args.ranking_margin),
    #                                     device=device).attach(engine, 'acc')
    #             # Ranking for ties with margin
    #             if args.ties:
    #                 RankAccuracy_ties(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label'],
    #                                                               args.ranking_margin),
    #                                   device=device).attach(engine, 'acc_ties')
    #         else:
    #             # Rank for non-ties, without margin
    #             RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label']),
    #                          device=device).attach(engine, 'acc')
    #
    #     # Classification Model Only
    #     elif args.model == 'sscnn':
    #         RunningAverage(Accuracy(output_transform=lambda x: (x['logits'], x['label']))).attach(engine, 'acc')
    #
    #     # Classification + Ranking Model
    #     elif args.model == 'rsscnn':
    #         # Ranking accuracy
    #         if args.full_accuracy:
    #             # Ranking for non-ties, with margin
    #             RankAccuracy_withMargin(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label_r'],
    #                                                                 args.ranking_margin),
    #                                     device=device).attach(engine, 'acc')
    #             # Ranking for ties with margin
    #             if args.ties:
    #                 RankAccuracy_ties(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label_r'],
    #                                                               args.ranking_margin),
    #                                   device=device).attach(engine, 'acc_ties')
    #         else:
    #             # Rank for non-ties, without margin
    #             RankAccuracy(output_transform=lambda x: (x['rank_left'], x['rank_right'], x['label_r']),
    #                          device=device).attach(engine, 'acc')
    #         # Classification accuracy
    #         RunningAverage(Accuracy(output_transform=lambda x: (x['logits'], x['label_c']))).attach(engine, 'c_acc')
    #
    #     else:
    #         raise Exception('Model type unknown: {}'.format(args.model))


    # =============================================================================================== #
    # RUN
    # =============================================================================================== #
    pbar = tqdm(total=len(dataloader))
    net.eval()
    evaluator.run(dataloader)
    pbar.close()

    # =============================================================================================== #
    # RESULTS
    # =============================================================================================== #
    # Collect all batch results into a single file
    batch_result_files = glob(path.join('outputs', '{}_*'.format(path.basename(args.load_model)) + '.pkl'))
    batch_results = []
    for batch_result_file in batch_result_files:
        batch_results.append(pd.read_pickle(batch_result_file))

    # Delete temporary batch pickle files
    for batch_result_file in batch_result_files:
        os.remove(batch_result_file)

    # Join batches and write global result to pickle
    global_df = pd.concat(batch_results, axis=0)
    global_df.to_pickle(path.join('outputs', 'saved', '{}_{}_results'.format(args.notes, path.basename(args.load_model)) + '.pkl'))
    print(global_df)
    print(global_df.shape)
