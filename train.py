# ./train.py --dataset=labeled.csv --debug=False

import argparse
from model import data_layer, news_classifier


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset",
                      type=str,
                      required=True,
                      help="Dataset in csv format")
  parser.add_argument("--debug", type=bool, required=True, help="Debug run")

  args = parser.parse_args()

  model_path = news_classifier.TrainOn(data_layer.FormDataset(
      data_layer.LoadLabeledData(args.dataset)),
                                       epochs=50,
                                       debug=args.debug)
  print(f'Training finished with model saved in {model_path}')


if __name__ == "__main__":
  main()
