import flask
from model import transformer as news_classifier

app = flask.Flask(__name__)

model = news_classifier.NewsBinaryClassifier('./model_00_02_17.pth')


@app.route('/predict', methods=['POST'])
def predict():
  try:
    req_json = flask.request.json
    url = req_json.get('url')
    if url is None:
      return flask.jsonify({'error': 'url not found in payload'}), 400
    return flask.jsonify({'is_news': model.Predict(url)})
  except Exception as e:
    print(f'Internal error {e}')
    return flask.jsonify({'error': 'Internal error'}), 500


@app.route('/isalive', methods=['GET'])
def is_alive():
  return flask.Response(status=200)


if __name__ == '__main__':
  app.run(debug=False, host="0.0.0.0", port=8080)