import tensorflow as tf
import base64
import json

from flask import Flask, render_template, request
from cnn import ConvolutionalNeuralNetwork
from io import BytesIO

app = Flask(__name__)

def calculate_operation(operation):
    def operate(fb, sb, op):
        if operator == '+':
            result = int(first_buffer) + int(second_buffer)
        elif operator == '-':
            result = int(first_buffer) - int(second_buffer)
        elif operator == 'x':
            result = int(first_buffer) * int(second_buffer)
        return result

    if not operation or not operation[0].isdigit():
        return -1

    operator = ''
    first_buffer = ''
    second_buffer = ''

    for i in range(len(operation)):
        if operation[i].isdigit():
            if len(second_buffer) == 0 and len(operator) == 0:
                first_buffer += operation[i]
            else:
                second_buffer += operation[i]
        else:
            if len(second_buffer) != 0:
                result = operate(first_buffer, second_buffer, operator)
                first_buffer = str(result)
                second_buffer = ''
            operator = operation[i]

    result = int(first_buffer)
    if len(second_buffer) != 0 and len(operator) != 0:
        result = operate(first_buffer, second_buffer, operator)

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    operation = BytesIO(base64.urlsafe_b64decode(request.form['operation']))
    CNN = ConvolutionalNeuralNetwork()
    operation = CNN.predict(operation)

    return json.dumps({
        'operation': operation,
        'solution': calculate_operation(operation)
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
