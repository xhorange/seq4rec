import torch
from flask import Flask, request, render_template
import torch as Torch

app = Flask(__name__)
app.debug = True
model = None


def load_bert_model():
    global model
    map_location = Torch.device('cpu')
    model = Torch.load('/home/orange/PycharmProjects/BertProject/model.pkl', map_location)
    model.eval()
    print('load model successfully')


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/movies_result', methods=["POST", "GET"])
def bert_result():
    global model
    if request.method == 'POST':
        seq_data = request.form.get('movies_id').split(',')
        if seq_data:
            result_str = compute_rank(seq_data)
            return render_template('answer.html', result_str=result_str)
    else:
        return render_template('index.html')


def compute_rank(seq_data):
    seq_data = list(map(int, seq_data))
    seq_data = seq_data + [3707]
    seq_data = seq_data[-100:]
    padding_len = 100 - len(seq_data)
    seq_data = [0] * padding_len + seq_data
    seq_data = [seq_data]
    seq = torch.LongTensor(seq_data)
    scores = model(seq)
    scores = scores[:, -1, :]  # B x V
    top_k_scores, top_k_id = torch.topk(scores, 100, sorted=True)
    rank = top_k_id.tolist()
    rank_str = ','.join(map(str, rank))
    return rank_str


if __name__ == '__main__':
    load_bert_model()
    app.run()
