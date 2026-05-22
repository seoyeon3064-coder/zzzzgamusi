"""
쯔쯔가무시 발병 위험도 예측 베이스 코드.

프로젝트 목표:
최근 1~3주 전 날씨 데이터를 이용해서 이번 주가 위험한지 예측합니다.

학생들은 TODO 부분을 직접 완성하면 됩니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path("data")
TARGET_COLUMN = "patient_count"

ENGLISH_COLUMNS = [
    "region",
    "year",
    "month",
    "week",
    "patient_count",
    "avg_temp",
    "max_temp",
    "min_temp",
    "rainfall",
    "avg_wind",
    "avg_humidity",
    "lag1_avg_temp",
    "lag1_max_temp",
    "lag1_min_temp",
    "lag1_rainfall",
    "lag1_avg_wind",
    "lag1_avg_humidity",
    "lag2_avg_temp",
    "lag2_max_temp",
    "lag2_min_temp",
    "lag2_rainfall",
    "lag2_avg_wind",
    "lag2_avg_humidity",
    "lag3_avg_temp",
    "lag3_max_temp",
    "lag3_min_temp",
    "lag3_rainfall",
    "lag3_avg_wind",
    "lag3_avg_humidity",
    "population",
]

def find_data_file(data_dir: Path) -> Path:
    """data 폴더에서 사용할 CSV 파일을 찾습니다."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the data folder.")

    # TODO: 학생들이 직접 사용할 CSV 파일을 선택해도 됩니다.
    return Path("D:/보충플젝/통합_AI_인구데이터.csv")

def load_data(path: Path) -> pd.DataFrame:
    """CSV 데이터를 불러오고 컬럼명을 영어로 바꿉니다."""

    # TODO: CSV 파일을 pandas로 읽어오세요.
    # (만약 한글 깨짐 에러가 나면 encoding='cp949'를 괄호 안에 추가하세요)
    df = pd.read_csv(path, encoding='cp949')

    # TODO: 컬럼명을 ENGLISH_COLUMNS로 바꾸세요.
    df.columns = ENGLISH_COLUMNS

    return df

def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """주차(week)를 이용해 계절 관련 feature를 추가합니다."""
    # TODO: 가을철 고위험 기간 feature를 추가하세요.
    # 1. 넓은 유행기 (직접 확인하신 42~51주차)
    df["fall_peak"] = df["month"].between(10,11).astype(int)
    
#     # 2. 집중 유행기 (선택 사항)
#     # 42~51주 안에서도 환자 수가 폭발적으로 많은 '최정점' 구간이 있다면 
#     # 눈으로 확인하신 후 아래 숫자를 바꿔주세요. (예: 45~48주)
#     # 만약 구분이 어렵다면 이 줄은 삭제하셔도 무방합니다.
#     df["fall_core_peak"] = df["week"].between(44, 48).astype(int)
    return df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """기존 날씨 데이터를 조합하여 더 강력한 힌트를 만듭니다."""

    # 1. 일교차 (최고기온 - 최저기온): 가을철 큰 일교차가 진드기 활동에 영향을 줄 수 있음
    df['temp_diff'] = df['max_temp'] - df['min_temp']
    df['lag1_temp_diff'] = df['lag1_max_temp'] - df['lag1_min_temp']
    
    # 2. 최근 3주간의 '평균' 기온 흐름 (얼마나 꾸준히 따뜻했나?)
    df['avg_temp_3w'] = (df['lag1_avg_temp'] + df['lag2_avg_temp'] + df['lag3_avg_temp']) / 3
    
    # 3. 최근 3주간의 '누적' 강수량 (땅이 얼마나 습한가?)
    df['total_rain_3w'] = df['lag1_rainfall'] + df['lag2_rainfall'] + df['lag3_rainfall']

    # 25년 9월부터 신고기준 변경으로 환자수 급감
    df['is_after_2025'] = (df['year'] >= 2025).astype(int)
    
    return df

def select_features(df: pd.DataFrame):
    """모델에 사용할 입력 feature와 정답 target을 선택합니다."""
    # 이번 주 환자수를 알기 전에 사용할 수 있는 정보만 입력값으로 사용합니다.
    # patient_count는 정답을 만들 때만 사용하고, 입력 feature로 사용하면 안 됩니다.
    # 조기 예측이 목표라면 이번 주 날씨 데이터도 입력 feature에서 제외하는 것이 좋습니다.

    feature_columns = [
        "year",
        "month",
        "week",
        "lag1_avg_temp",
        "lag1_max_temp",
        "lag1_min_temp",
        "lag1_rainfall",
        "lag1_avg_wind",
        "lag1_avg_humidity",
        "lag2_avg_temp",
        "lag2_max_temp",
        "lag2_min_temp",
        "lag2_rainfall",
        "lag2_avg_wind",
        "lag2_avg_humidity",
        "lag3_avg_temp",
        "lag3_max_temp",
        "lag3_min_temp",
        "lag3_rainfall",
        "lag3_avg_wind",
        "lag3_avg_humidity",
        "region",
        "lag1_temp_diff",
        "avg_temp_3w",
        "total_rain_3w",
        "fall_peak",
        "is_after_2025",
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # TODO: region 같은 문자형 컬럼을 숫자형 컬럼으로 변환하세요.

    X = pd.get_dummies(X, columns=["region"])

    return X, y


# 2. 학습
def train_model(X_train, y_train):
    """최신 부스팅 알고리즘인 LightGBM 모델로 학습합니다."""

    model = LGBMRegressor(
        n_estimators=100,        # 이어달리기를 100번 반복합니다.
        max_depth=7,             # 과대적합 방지를 위해 나무 깊이를 제한합니다.
        learning_rate=0.05,      # 얼마나 꼼꼼히 학습할지 (보통 0.05 ~ 0.1 사이를 씁니다)
        class_weight="balanced", # 🌟 놓치기 쉬운 '위험(1)'을 틀리면 벌점을 크게 줍니다!
        random_state=42,
        n_jobs=-1,
        force_row_wise=True      # 경고 메시지 방지용 설정
    )

    # 모델 학습
    model.fit(X_train, y_train, sample_weight=weights)
    return model

def main():
    # 1. 데이터 준비
    data_path = find_data_file(DATA_DIR)
    df = load_data(data_path)

    # 2. 데이터 가공

    weights = X_train['is_after_2025'].apply(lambda x: 3.0 if x == 1 else 1.0)

    df = add_season_features(df)
    df = add_advanced_features(df)
    X, y = select_features(df)

    # 3. 데이터 쪼개기 (학습용 80%, 시험용 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=False # 시계열 데이터이므로 섞지 않고 순서대로 나눕니다.
    )

    # 4. 모델 학습 및 평가
    model = train_model(X_train, y_train)

    # 1. 모델이 예측한 0 또는 1의 결과 (y_pred)
    y_pred = model.predict(X_test)
    evaluate_regression_model(y_test, y_pred)

    joblib.dump(model, './tsutsugamushi_model1.joblib')
    print("✅ 모델 저장 완료!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 2. 모델 평가 (회귀 모델용)
def evaluate_regression_model(y_true, y_pred):
    """
    회귀 모델의 평가지표(MAE, RMSE, R2)를 출력합니다.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) # RMSE: 오차의 단위가 환자 수와 같아져서 해석하기 제일 좋아!
    r2 = r2_score(y_true, y_pred)
    
    print("====================================")
    print("📈 회귀 모델 평가 성적표 📈")
    print("====================================")
    print(f"✅ MAE (평균 오차): {mae:.2f} 명")
    print(f"✅ RMSE (표준 오차): {rmse:.2f} 명")
    print(f"✅ R2 Score (설명력): {r2:.4f}")
    print("====================================")
    print("💡 해석 가이드:")
    print(" - MAE/RMSE가 작을수록 모델이 실제 환자 수를 잘 맞추고 있다는 뜻이야!")
    print(" - R2 Score는 1에 가까울수록 모델이 데이터를 아주 잘 설명하고 있다는 뜻이야.")

if __name__ == "__main__":
    main()


