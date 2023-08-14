# Trains a transformer model of news binary classifier.
# ./train.py --dataset=labeled.csv --debug=False

import argparse
from model import data_layer, news_classifier


def str_to_bool(value) -> bool:
  if isinstance(value, bool):
    return value
  elif isinstance(value, str) and value.lower() in ("true"):
    return True
  elif isinstance(value, str) and value.lower() in ("false"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean literal expected.")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset",
                      type=str,
                      required=True,
                      help="Dataset in csv format")
  parser.add_argument("--debug", type=str_to_bool, required=True, help="Debug run")

  args = parser.parse_args()
  print(f'Train model with args ' +
        ', '.join(f'{arg}:{getattr(args, arg)}' for arg in vars(args)))

  model_path = news_classifier.TrainOn(data_layer.FormDataset(
      data_layer.LoadLabeledData(args.dataset)),
                                       epochs=50,
                                       debug=args.debug)
  print(f'Training finished with model saved in {model_path}')


if __name__ == "__main__":
  main()
