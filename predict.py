'''Classify crawled data if it's news.

It takes a Google Cloud Storage URL of crawled data, and generates the output artifact in the same directory.
'''

from absl import app
from absl import flags
from model import news_classifier
import pandas as pd
import logging
from google.cloud import storage
import tempfile
import tqdm

_DATA_URL = flags.DEFINE_string("data_url", '',
                                'Data CSV Google Cloud Storage URL.')
_DRY_RUN = flags.DEFINE_bool('dry_run', True, 'Dry run to process few rows.')
_MODEL_URL = flags.DEFINE_string(
    'model_url', './model.pth',
    'Url of model checkpoint. Could either be a local file or GCS URL.')


def parse_gcs_url(url: str) -> tuple[str, str]:
  '''Returns bucket and object id from GCS URL.'''
  gs_schema = 'gs://'

  if not url.startswith(gs_schema):
    raise ValueError(f'Invalid GCS URL schema, {url}.')
  path = url[len(gs_schema):].split('/')
  return path[0], '/'.join(path[1:])


def download_from_gcs(url: str) -> str:
  bucket, object = parse_gcs_url(url)
  with tempfile.NamedTemporaryFile(delete=False) as f:
    storage.Client().bucket(bucket).blob(object).download_to_file(f)
    logging.info(f'Successfully downloaded {url} to {f.name}')
    return f.name


def upload_to_gcs(url: str, file_path: str) -> None:
  bucket, object = parse_gcs_url(url)
  with open(file_path, 'rb') as file:
    storage.Client().bucket(bucket).blob(object).upload_from_file(file)
    logging.info(f'Successfully uploaded {file_path} to {url}')


def _load_checkpoint(url: str) -> str:
  if url.startswith("gs://"):
    return download_from_gcs(url)
  else:
    return url


def main(_):
  logging.info(
      f'Classify data with data_url: {_DATA_URL.value}, dry_run: {_DRY_RUN.value}, model_url: {_MODEL_URL.value}'
  )

  model = news_classifier.NewsBinaryClassifier(
      _load_checkpoint(_MODEL_URL.value))

  data = pd.read_csv(download_from_gcs(_DATA_URL.value), header=0)
  data.dropna(inplace=True)

  if _DRY_RUN.value:
    data = data[:100]

  tqdm.tqdm.pandas()
  data['is_news'] = data['Url'].progress_apply(
      lambda url: bool(model.Predict(url)))

  news = data[data['is_news']].copy().reset_index(drop=True)
  news.dropna(inplace=True)
  logging.info(f'Found {len(news)} news from {len(data)} rows.')

  with tempfile.NamedTemporaryFile(delete=False, mode='w',
                                   encoding='utf-8') as f:
    news.to_csv(f, header=True, index=False)

  upload_to_gcs('/'.join(_DATA_URL.value.split('/')[:-1] + ['news.csv']),
                f.name)


if __name__ == '__main__':
  app.run(main)
