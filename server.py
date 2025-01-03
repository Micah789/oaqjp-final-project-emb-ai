''' 
Executing this function initiates the application of sentiment    
analysis to be executed over the Flask channel and deployed on
localhost:5000.
'''
from flask import Flask, request, render_template
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion detection")

@app.route('/')
def render_index_page():
    """
    Render and load the home page
    """
    return render_template("index.html")

@app.route("/emotionDetector")
def sent_emotion():
    """
    the emotionDetector which is trigger from clicking the submit button
    """
    text_to_analyze = request.args.get('textToAnalyze')

    response = emotion_detector(text_to_analyze)

    if response['dominant_emotion'] is None:
        return "Invalid text! Please try again!."
    return f"""For the given statement,
    the system response is 'anger': {response['anger']}, 'disgust': {response['disgust']}, 
    'fear': {response['fear']}, 'joy': {response['joy']} and 'sadness': {response['sadness']}. 
    The dominant emotion is {response['dominant_emotion']}."""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
