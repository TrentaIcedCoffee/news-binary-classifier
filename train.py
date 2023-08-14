# ./train.py --dataset=labeled.csv --debug=False

import argparse
from model import transformer, data


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset",
                      type=str,
                      required=True,
                      help="Dataset in csv format")
  parser.add_argument("--debug", type=bool, required=True, help="Debug run")

  args = parser.parse_args()

  model_path = transformer.TrainOn(data.FormDataset(
      data.LoadLabeledData(args.dataset)),
                                   epochs=50,
                                   debug=args.debug)
  print(f'Training finished with model saved in {model_path}')


if __name__ == "__main__":
  main()
