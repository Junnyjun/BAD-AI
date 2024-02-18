from rasa.shared.nlu.training_data import loading
from rasa.shared.core.training_data import loading
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.training_data import TrainingData
from rasa.shared.core.training_data import DialogueTrainingData
from rasa.shared.nlu.interpreter import RasaNLUInterpreter
from rasa.core.agent import Agent

# 챗봇 모델 로드
nlu_interpreter = RasaNLUInterpreter.load("models/nlu")
domain = Domain.load("models/domain.yml")
agent = Agent.load("models/core")

# 사용자 질문 입력
user_message = input("질문을 입력하세요: ")

# 챗봇 응답 생성
prediction = nlu_interpreter.parse(user_message)
intent = prediction["intent"]["name"]
entities = prediction["entities"]

# 송금 종류 및 정보 확인
if intent == "송금_상태_확인":
    transfer_type, transfer_amount, transfer_date, transfer_time = get_transfer_info(entities)

    # 송금 정보 검색
    transfer_info = search_transfer_info(transfer_type, transfer_amount, transfer_date, transfer_time)

    # 송금 정보 출력
    if transfer_info is None:
        print("송금 정보를 찾을 수 없습니다.")
    else:
        print(f"송금 종류: {transfer_info['transfer_type']}")
        print(f"송금 금액: {transfer_info['transfer_amount']}")
        print(f"송금 날짜: {transfer_info['transfer_date']}")
        print(f"송금 시간: {transfer_info['transfer_time']}")
        print(f"송금 상태: {transfer_info['transfer_status']}")

# 송금 상황 및 진행 상황 확인
# ...

# 송금 관련 문제 해결 지원
# ...

# 챗봇 응답 출력

# 송금 정보 추출 함수
def get_transfer_info(entities):
    transfer_type = None
    transfer_amount = None
    transfer_date = None
    transfer_time = None

    if "송금_종류" in entities:
        transfer_type = entities["송금_종류"][0]["value"]
    if "송금_금액" in entities:
        transfer_amount = entities["송금_금액"][0]["value"]
    if "송금_날짜" in entities:
        transfer_date = entities["송금_날짜"][0]["value"]
    if "송금_시간" in entities:
        transfer_time = entities["송금_시간"][0]["value"]

    return transfer_type, transfer_amount, transfer_date, transfer_time

# 송금 정보 검색 함수
def search_transfer_info(transfer_type, transfer_amount, transfer_date, transfer_time):
    # ...

    return transfer_info
