from flask import Flask
import torch as Torch

app = Flask(__name__)


def load_bert_model():
    map_location = Torch.device('cpu')
    model = Torch.load('/home/orange/PycharmProjects/BertProject/model.pkl', map_location)
    model.eval()
    print('load model successfully')


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    load_bert_model()
    app.run()
