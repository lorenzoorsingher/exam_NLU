import argparse
import random


def generate_id(len=5):
    STR_KEY_GEN = "ABCDEFGHIJKLMNOPQRSTUVWXYzabcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(STR_KEY_GEN) for _ in range(len))


def get_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Get the params""",
    )

    parser.add_argument(
        "-j",
        "--json",
        type=str,
        help="Load params from json",
        default="",
    )

    parser.add_argument(
        "-NL",
        "--no-log",
        action="store_true",
        help="Do not log run",
    )

    parser.add_argument(
        "-trs",
        "--train-batch-size",
        type=int,
        help="Set the train batch size",
        default=256,
        metavar="",
    )

    parser.add_argument(
        "-tes",
        "--test-batch-size",
        type=int,
        help="Set the test batch size",
        default=1024,
        metavar="",
    )

    parser.add_argument(
        "-des",
        "--dev-batch-size",
        type=int,
        help="Set the dev batch size",
        default=1024,
        metavar="",
    )

    parser.add_argument(
        "-w",
        "--wandb-secret",
        type=str,
        help="Set wandb secret",
        default="",
        metavar="",
    )

    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        help="Set checkpoint save path",
        default="assignment_1/checkpoints/",
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
