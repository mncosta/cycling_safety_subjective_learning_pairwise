import json
import wandb


def log_wandb(metrics):
    try:
        if metrics['accuracy_validation'] > wandb.summary['max_accuracy_validation']:
            metrics['max_accuracy_train'] = metrics['accuracy_train']
            metrics['max_accuracy_validation'] = metrics['accuracy_validation']
            metrics['max_accuracy_test'] = metrics['accuracy_test']
        else:
            metrics['max_accuracy_train'] = wandb.summary['max_accuracy_train']
            metrics['max_accuracy_validation'] = wandb.summary['max_accuracy_validation']
            metrics['max_accuracy_test'] = wandb.summary['max_accuracy_test']
    except KeyError:
        pass

    wandb.log(metrics)


def log_console(metrics):
    print(f"Results - Epoch: {metrics['epoch']} - Step: {metrics['iteration']}")
    print(json.dumps(metrics, indent=2))


def log(args, metrics):
    if args.log_wandb:
        log_wandb(metrics)

    if args.log_console:
        log_console(metrics)