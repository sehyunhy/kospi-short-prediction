# 📉 KOSPI Top 50 공매도 영향 예측

> KOSPI 시가총액 상위 50개 종목의 공매도 데이터를 기반으로,  
> **내일의 공매도 압력 방향(고압력 / 저압력)** 을 머신러닝으로 예측하는 프로젝트입니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 분석 대상 | KOSPI 시가총액 Top 50 종목 |
| 예측 방식 | Extreme-State Binary Classification (고압력 / 저압력) |
| 타겟 정의 | 공매도 영향 지수 상위 30% → 고압력(1), 하위 30% → 저압력(0) |
| 중립 구간 | 중간 40% 제거 (신호 불명확 구간 학습 제외) |
| 출력 | 내일 공매도 방향 예측 확률 + SHAP 해석 + Streamlit 대시보드 |

---

## 🏗️ 프로젝트 구조

```
kospi_short_selling/
│
├── app.py                        ← Streamlit 대시보드 진입점
├── main.py                       ← CLI 파이프라인 진입점
├── requirements.txt              ← 패키지 의존성
│
├── config/
│   └── config.py                 ← 전역 설정 (경로, 하이퍼파라미터 등)
│
├── src/
│   ├── data/
│   │   └── loader.py             ← 데이터 로드 / Top N 선정 / 결측치 처리
│   │
│   ├── features/
│   │   └── engineer.py           ← 파생 피처 / 일별 집계 / 시계열 피처 / 타겟 생성
│   │
│   ├── models/
│   │   ├── definitions.py        ← 모델 인스턴스 팩토리
│   │   ├── trainer.py            ← 교차검증 / 최종 학습 / 앙상블
│   │   ├── predictor.py          ← 내일 예측
│   │   └── explainer.py          ← SHAP 분석
│   │
│   └── utils/
│       ├── visualizer.py         ← 모든 시각화 함수
│       └── logger.py             ← 공통 로거
│
├── data/                         ← ⚠️ CSV 파일을 여기에 넣으세요 (Git 미포함)
│   ├── df_short.csv
│   └── df_price.csv
│
├── outputs/                      ← 결과 이미지 / 로그 자동 저장
│
└── .vscode/
    ├── settings.json             ← Python 경로, 포매터 설정
    └── launch.json               ← F5 실행 설정
```

---

## 🤖 사용 모델

| 구분 | 모델 |
|---|---|
| 선형 | Logistic Regression, LDA |
| 비선형 | Random Forest, XGBoost, LightGBM, CatBoost |
| 앙상블 | Soft Voting, Stacking (메타: LogisticRegression) |
| 해석 | SHAP (TreeExplainer, LinearExplainer) |

---

## 🚀 실행 방법 (Mac 기준)

### 0. 사전 준비

- Python 3.10 이상
- Homebrew 설치 ([brew.sh](https://brew.sh))

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/kospi_short_selling.git
cd kospi_short_selling
```

### 2. 가상환경 생성 & 활성화

```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ 터미널 앞에 `(venv)` 가 붙으면 성공입니다.

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 한글 폰트 설치

```bash
brew install font-nanum
```

### 5. 데이터 파일 준비

`data/` 폴더 안에 아래 두 파일을 넣어주세요.

```
data/
├── df_short.csv    ← 공매도 데이터
└── df_price.csv    ← 주가 데이터
```

> ⚠️ 데이터 파일은 용량 문제로 Git에 포함되지 않습니다.  
> 팀 공유 드라이브 또는 슬랙에서 받아주세요.

### 6-A. CLI로 실행 (차트 파일 저장)

```bash
python main.py
```

결과물은 `outputs/` 폴더에 저장됩니다.

```
outputs/
├── short_selling_v3.png      ← 메인 대시보드 차트
├── shap_waterfall_v3.png     ← SHAP Waterfall 차트
└── run.log                   ← 실행 로그
```

### 6-B. Streamlit 대시보드로 실행

```bash
python -m streamlit run app.py
```

브라우저에서 **http://localhost:8501** 로 접속하면 됩니다.

| 탭 | 내용 |
|---|---|
| 🎯 예측 결과 | 내일 방향 카드 + 모델별 확률 테이블 |
| 📊 모델 성능 | CV F1 비교 + 요일별 분포 차트 |
| 🔍 SHAP 해석 | 모델 선택 후 피처 중요도 인터랙티브 차트 |
| 📈 데이터 탐색 | 공매도 지수 시계열 + 원본 데이터 |

종료하려면 터미널에서 `Ctrl + C`를 누르세요.

---

## ⚙️ 주요 설정 변경

`config/config.py` 에서 아래 값들을 바꿀 수 있습니다.

```python
TOP_N    = 50      # 시가총액 상위 N개 종목
Q_LOW    = 0.30    # 저압력 분위수 기준 (하위 30%)
Q_HIGH   = 0.70    # 고압력 분위수 기준 (상위 70%)
N_SPLITS = 5       # 시계열 교차검증 fold 수
```

Streamlit에서는 사이드바 슬라이더로 실시간 조정도 가능합니다.

---

## ❓ 자주 겪는 문제

**Q. `source .venv/bin/activate` 가 안 돼요.**

가상환경 폴더 이름을 확인하세요. `.venv`가 아니라 `venv`로 만들었다면:
```bash
source venv/bin/activate
```

**Q. 한글이 깨져서 차트에 □□□ 로 나와요.**

폰트 설치 후 matplotlib 캐시를 삭제해주세요:
```bash
rm -rf ~/.matplotlib/fontlist*.json
```
그 다음 다시 실행하면 됩니다.

**Q. `ModuleNotFoundError` 가 떠요.**

가상환경이 활성화되어 있는지 확인하세요. 터미널 앞에 `(venv)` 가 없다면:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 주요 패키지 버전

| 패키지 | 버전 |
|---|---|
| Python | 3.10 이상 |
| scikit-learn | 1.4 이상 |
| XGBoost | 2.0 이상 |
| LightGBM | 4.3 이상 |
| CatBoost | 1.2 이상 |
| SHAP | 0.45 이상 |
| Streamlit | 1.35 이상 |

---

## 👥 팀

| 이름 | 역할 |
|---|---|
| 이세현 | PM / 모델링 |
| (팀원 추가) | (역할 추가) |
