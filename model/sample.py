import nltk
import json

nltk.download('punkt')


def load_responses():
    with open('responses.json', 'r') as f:
        responses = json.load(f)
    return responses


responses = load_responses()


def respond_to(text):
    """주어진 텍스트에 대한 응답을 반환합니다."""
    words = nltk.word_tokenize(text)
    keywords = ["송금", "시간", "돈", "금액", "국내", "해외"]
    found_keywords = [word for word in words if word in keywords]
    missing_keywords = [keyword for keyword in keywords if keyword not in found_keywords]
    if len(missing_keywords) > 0:
        return f"{', '.join(missing_keywords)}에 대한 정보를 제공해주세요."
    else:
        return responses[found_keywords[0]]

while True:
    text = input("> ")
    response = respond_to(text)
    print(response)
