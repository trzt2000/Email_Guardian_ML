from sklearn.tree import DecisionTreeClassifier
from joblib import load
from attrib import extract_features
from flask_cors import CORS

# load the saved model
classifier: DecisionTreeClassifier = load("decision_tree_model.joblib")



def check_phishing(url: str):
    x_data = []  # list of 30 elements
    x_data = extract_features(url)
    for n in range(len(x_data)-1):
        if x_data[n] is None:
            x_data[n] = 0
    print(f"\n>Extracted Features : {x_data}")
    # make a prediction
    predictions = classifier.predict([x_data])
    # print(predictions)
    if predictions[0] == 1:
        return "legitimate "
    return "phishing "


def check_phishing_2(url: str):
    x_data = []  # list of 30 elements
    x_data = extract_features(url)
    print(f">Extracted Features : {x_data}")
    # make a prediction
    predictions = classifier.predict_proba([x_data])
    # print(predictions)
    if predictions[0][1] > 0.5:
        return f"legitimate ({predictions[0][1] * 100}%)"
    return f"phishing ({predictions[0][0] * 100}%)"


# API endpoint here =========================================================
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)



@app.route('/phishing_detection', methods=['POST'])
def receive_data():
    data = request.json  
    print("*"*80)
    print(f">Received data : {data}\n")
    print(">Processing data".center(80,'.'))
    result = check_phishing(data['url'])

    print("\n>>URL : ", data['url'], " is ", result)
    return jsonify({'status': 'success', 'result': result})


if __name__ == '__main__':
    app.run(debug=False, port=5002)
