"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
from math import ceil


def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def get_lr_sched(global_step, opts):
    # learning rate scheduling
    lr_this_step = opts.learning_rate * warmup_linear(
        global_step, opts.warmup_steps, opts.num_train_steps)
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step
