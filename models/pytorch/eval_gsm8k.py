import argparse
import sys
import importlib

import logging

logger = logging.getLogger(__name__)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_module", type=str, help="name of the model module")
    parser.add_argument("--model_storage", type=str, help="Path to the model storage")
    parser.add_argument("--evaluation_loader", type=str, help="name of the evaluation module")
    parser.add_argument("--evaluation_storage", type=str, help="name of the evaluation storage")
    parser.add_argument("--output_folder", type=str, help="Path to the output folder")
    parser.add_argument("--number_of_shots", type=int, help="Number of shots")
    return parser.parse_args(args)

def try_import(modelname):
    try:
        return importlib.import_module(modelname)
    except ImportError:
        logger.error('could not import %s', modelname)
        raise

def main(args):
    modelmod = try_import(args.model_module) 
    datasetmod = try_import(args.dataset_module) 

    model = modelmod.load_from_checkpoint(args.model_storage)
    dataset  = datasetmod.load_from_storage(args.dataset_storage)

    for sample_batch in dataset.sample_batch_it(batch_size=8):
        result = model.predict(sample_batch)
        metricmod.compute(model, sample_batch)


if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        main(args)
    except Exception as e:
        logger.error('could not run experiment', e, exc_info=True)