import argparse
import sys

sys.path.append("..")

from RFFI_OSR_SFC.model.sfcr import SFCR


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def get_model_from_args(args, **kwargs):
    return get_grnet_center_model_from_args(args, **kwargs)

def get_grnet_center_model_from_args(args, num_classes=10, data_shape=(1, 16384, 2)):
    return SFCR(num_classes=num_classes, data_shape=data_shape, loss_weight=args.gamma)

