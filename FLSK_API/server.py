from flask import Flask, request
import LC
import json

app = Flask(__name__)
model_center = LC.Model_center()

@app.route('/')
def hello_world():
    return 'hello world'


@app.route('/langchain', methods=['POST'])
def register():
    data = json.loads(request.get_data())
    quz = data['question']
    print(quz)
    history = data['history']
    print(history)
    e,ans = model_center.qa_chain_self_answer(quz,history)
    print(ans)
    print(ans[0][1])
    return {"anwser":ans[0][1]}


if __name__ == '__main__':
    app.run(port=5001)