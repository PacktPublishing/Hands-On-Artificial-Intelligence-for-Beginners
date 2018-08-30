from flask import Flask, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
CORS(app)

## Hit the Predict API
@app.route('/predict', methods=['GET'])
def sendPredict():
    response = predict()
    print(response[0])
    return jsonify(response[0])

if __name__ == '__main__':
    app.run(debug=True)
