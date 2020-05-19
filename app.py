from flask import Flask, request
import torch
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

net = torch.jit.load('RNN.zip')


@app.route('/')
def hello():
    return "Hello World!"


@app.route("/predict", methods=['POST'])
def predict():
    
    context = request.json
    frase = context['text']
    pred = net(frase)
    pred_probas = torch.softmax(pred, axis=0)
    return {
        'Fake': pred_probas[1].item(),
        'Real': pred_probas[0].item()
    }


if __name__ == "__main__":
    app.run(debug=True)
