from flask import Flask, jsonify, request
from model import CaptionBot

app = Flask(__name__)




@app.route('/',methods=['GET'])
def index():
    return 'Machine Learning'


@app.route('/predict',methods=['POST'])
def fetch_result():
    if 'file' not in request.files:
        return "Please try again image does not exist"
    file = request.files.get('file')
    if not file:
        return
    img_bytes = file.read()
    c = CaptionBot(img_bytes)
    caption = c.run()
    return jsonify(prediction = caption)


if __name__ == '__main__':
    app.run(debug=True)