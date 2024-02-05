import argparse
import sys
import importlib
import torch
import logging

logger = logging.getLogger(__name__)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", type=str, help="name of the model module")
    parser.add_argument("--model_storage", type=str, help="Path to the model storage")
    parser.add_argument("--evaluation_module", type=str, help="name of the evaluation module")
    parser.add_argument("--evaluation_storage", type=str, help="name of the evaluation storage")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    parser.add_argument("--number_of_shots", type=int, help="Number of shots", default=1)
    return parser.parse_args(args)

def try_import(modelname):
    try:
        return importlib.import_module(modelname)
    except ImportError:
        logger.error('could not import %s', modelname)
        raise

def main(args):
    modelmod = try_import(args.model_module) 
    evalmodule = try_import(args.evaluation_module) 

    torch.set_default_device('cuda')
    
    model, tokenizer = modelmod.load_from_checkpoint(args.model_storage)
    evalmodule.load_from_storage(args.evaluation_storage)
    
    import numpy as np
    import random
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    result = evalmodule.evaluate(model, tokenizer, shots=args.number_of_shots, genargs=dict(top_p=0.9, temperature=0.75, do_sample=True))
    print(result)

if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        main(args)
    except Exception as e:
        logger.error('could not run experiment: %s', e, exc_info=True)