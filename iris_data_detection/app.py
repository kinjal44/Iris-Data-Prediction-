from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def main():
   return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    sw = request.form['sw']
    sl = request.form['sl']
    pw = request.form['pw']
    pl = request.form['pl']
    #arr = np.array([sw, sl, pw, pl])
    arr = np.array([[sw, sl, pw, pl]])
    #print(arr.ndim)
    arr = arr.astype(np.float64)
    pred = model.predict(arr)
    print(pred)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)
