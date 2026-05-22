import joblib
import numpy as np
from my_pipeline import predict_and_classify_risk


# 1. 하드디스크에 저장해 둔 모델 파일을 쏙 불러온다.
loaded_model = joblib.load('tsutsugamushi_model.joblib')

# 2. 다시 학습할 필요 없이 바로 새로운 데이터(X_new)를 넣고 예측시킨다!
y_pred = loaded_model.predict(X_new)

def predict_risk_level(model, X_new):
    # 모델이 예측한 환자 수
    pred_patient_count = model.predict(X_new)
    # 음수 출력 방어
    pred_patient_count = np.clip(pred_patient_count, 0, None)

    # 예측된 환자 수를 인구수로 나눠서 10만 명당 환자 수로 환산
    pred_ratio = (pred_patient_count / X_new['population']) * 100000

    conditions = [(pred_ratio >= 3), (pred_ratio >= 0.5) & (pred_ratio < 3)]
    choices = ['위험', '주의']

    final_risk_level = np.select(conditions, choices, default='안전')

    return final_risk_level, pred_patient_count
