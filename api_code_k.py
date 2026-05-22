import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

url = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
params = {
    'ServiceKey': 'd65455079f3d0eedb6add1ed34d344a6bb98472668f2dded077757196a37bd5e',
    'pageNo': '1',
    'numOfRows': '100',
    'dataType': 'JSON',
    'regId': '11B00000 ', # 예보구역번호
    'tmFc': '202605150500', # 발표시각
}

# response = requests.get(url, params=params)

# if response.status_code == 200:
#     data = response.json()
#     print(data)
# else:
#     print(f"API 요청 실패: {response.status_code}")

def get_session():
    session = requests.Session()
    # 502, 503, 504 에러가 나면 최대 3번까지 자동으로 재시도
    retry = Retry(
        total=3, 
        backoff_factor=1, 
        status_forcelist=[502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 이제 requests 대신 session을 쓰면 돼!
session = get_session()
try:
    response = session.get(url, params=params, timeout=10)
    data = response.json()
    print("성공적으로 데이터를 가져왔어!")
    print(data)
except Exception as e:
    print(f"결국 실패했어... 내일 다시 해보자: {e}")