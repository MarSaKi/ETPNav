import argparse
import sys
import json


def load_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--vlnbert', choices=['cmt'])
    parser.add_argument(
        "--model_config", type=str, help="path to model structure config json"
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to model checkpoint (*.pt)"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    # training parameters
    parser.add_argument(
        "--train_batch_size",
        default=4096,
        type=int,
        help="Total batch size for training. ",
    )
    parser.add_argument(
        "--val_batch_size",
        default=4096,
        type=int,
        help="Total batch size for validation. ",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumualte before "
        "performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--valid_steps", default=1000, type=int, help="Run validation every X steps"
    )
    parser.add_argument("--log_steps", default=1000, type=int)
    parser.add_argument(
        "--num_train_steps",
        default=100000,
        type=int,
        help="Total number of training updates to perform.",
    )
    parser.add_argument(
        "--optim",
        default="adamw",
        choices=["adam", "adamax", "adamw"],
        help="optimizer",
    )
    parser.add_argument(
        "--betas", default=[0.9, 0.98], nargs="+", help="beta for adam optimizer"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="tune dropout regularization"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="weight decay (L2) regularization",
    )
    parser.add_argument(
        "--grad_norm",
        default=2.0,
        type=float,
        help="gradient clipping (-1 for no clipping)",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10000,
        type=int,
        help="Number of training steps to perform linear " "learning rate warmup for.",
    )

    # device parameters
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of data workers"
    )
    parser.add_argument("--pin_mem", action="store_true", help="pin memory")

    # distributed computing
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank for distributed training on gpus",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Id of the node",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of GPUs across all nodes",
    )

    # can use config files
    parser.add_argument("--config", required=True, help="JSON config files")

    return parser


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args
