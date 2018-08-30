from flask import Flask, jsonify
from flask_restful import Api
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
api = Api(app)
CORS(app)

## Hit the Predict API
@app.route('/predict', methods=['GET'])
def sendPredict():
    '''Function for returning a list of node names based on type'''
    response = predict()
    print(response[0])
    return jsonify(response[0])

if __name__ == '__main__':
    app.run(debug=True)
