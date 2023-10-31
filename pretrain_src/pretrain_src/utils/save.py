"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
import torch


def save_training_meta(args):
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'logs', 'training_args.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(os.path.join(args.output_dir, 'logs', 'model_config.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None):
        output_model_file = os.path.join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.cpu()
            else:
                state_dict[k] = v
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')

