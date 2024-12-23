import requests
import json

def emotion_detector(text_to_analyse):
    """
    Emotion detector using the watson emotion api

    Parameters
    ----------
    text_to_analyse : string
                      The text from the input

    Returns
    -------
    dict
        the response from the api
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = { "raw_document": { "text": text_to_analyse } }
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    response_object = {}

    if response.status_code == 200:    
        # convert the object to dict
        formatted_response = json.loads(response.text)

        # extract the prediction
        emotion = list(map(lambda x: x['emotion'], formatted_response['emotionPredictions']))[0]

        response_object = {
            'anger': emotion['anger'],
            'disgust': emotion['disgust'],
            'fear': emotion['fear'],
            'joy': emotion['joy'],
            'sadness': emotion['sadness'],
            'dominant_emotion': max(emotion, key=emotion.get)
        }
    elif response.status_code == 400 or response.status_code == 500:
        response_object = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }
    return response_object



    