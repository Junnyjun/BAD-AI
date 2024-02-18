import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# 학습 데이터를 불러옵니다.
with open('conversations.json', 'r') as f:
    conversations = json.load(f)

questions = []
answers = []

# 질문과 응답을 분리합니다.
for conversation in conversations:
    questions.append(conversation['question'])
    answers.append(conversation['answer'])

# 텍스트를 토큰화하고, 시퀀스로 변환합니다.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
total_words = len(tokenizer.word_index) + 1

questions_sequences = tokenizer.texts_to_sequences(questions)
answers_sequences = tokenizer.texts_to_sequences(answers)

# 시퀀스를 동일한 길이로 패딩합니다.
max_sequence_length = max([len(x) for x in questions_sequences])
questions_padded = pad_sequences(questions_sequences, maxlen=max_sequence_length, padding='post')
answers_padded = pad_sequences(answers_sequences, maxlen=max_sequence_length, padding='post')

# 모델을 정의하고 학습합니다.
model = keras.Sequential([
    keras.layers.Embedding(total_words, 100, input_length=max_sequence_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(questions_padded, np.array(answers_padded), epochs=100)

def respond_to(text):
    """주어진 텍스트에 대한 응답을 반환합니다."""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')
    prediction = model.predict(padded)
    predicted_word_index = np.argmax(prediction)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "미안해요, 이해하지 못했어요."

while True:
    text = input("> ")
    response = respond_to(text)
    print(response)