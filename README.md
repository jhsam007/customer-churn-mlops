Customer Churn Prediction System (Production-Grade MLOps)

An end-to-end, production-ready machine learning system for predicting customer churn using structured telecom data.

This project demonstrates a complete ML lifecycle implementation — from data preprocessing to deployment — following MLOps best practices.

Business Problem

Customer churn significantly impacts subscription-based businesses and recurring revenue models.

Early identification of high-risk customers enables:

Targeted retention strategies

Reduced revenue loss

Improved customer lifetime value (CLV)

This system predicts the probability of customer churn and applies configurable business decision logic to generate actionable churn labels.

Key Features

End-to-end ML pipeline (training → evaluation → deployment)

Experiment tracking with MLflow

Model versioning and registry

Configurable decision threshold layer

Model explainability using SHAP

REST API serving with FastAPI

Docker containerization

CI/CD with GitHub Actions

Unit testing with Pytest

Reproducible and modular project structure

System Architecture

                ┌──────────────────┐
                │  Raw CSV Dataset │
                └──────────┬───────┘
                           │
                    Data Loader
                           │
                    Preprocessing
                           │
                      ML Pipeline
                           │
                    MLflow Tracking
                           │
               Model Registry (Versioned)
                           │
                   Probability Output
                           │
              Decision Layer (Threshold)
                           │
                 Explainability (SHAP)
                           │
                   FastAPI Inference
                           │
                     Docker Container

Architectural Design Principles

Separation of prediction and business decision logic

Centralized configuration management

Version-controlled model artifacts

Reproducible training and inference pipeline

Production-ready API serving

Model Explainability

Model interpretability is implemented using SHAP.

Capabilities:

Global feature importance analysis

Local explanation for individual predictions

Business insight extraction

Transparent decision support

Explainability improves trust, auditability, and real-world usability of the system.

Model Performance

Algorithms:

Logistic Regression

Random Forest (configurable)

Evaluation Metrics:

ROC-AUC

Precision

Recall

F1-score

(Replace with actual metrics from your experiments.)

Testing

Unit tests cover:

Data preprocessing

Model training

Pipeline integration

Decision threshold logic

Configuration consistency

Run tests:

pytest

Tech Stack

| Category         | Tools          |
| ---------------- | -------------- |
| Language         | Python 3.10    |
| ML               | scikit-learn   |
| Explainability   | SHAP           |
| Tracking         | MLflow         |
| API              | FastAPI        |
| Serving          | Uvicorn        |
| Testing          | Pytest         |
| CI/CD            | GitHub Actions |
| Containerization | Docker         |

Project Structure

customer-churn/
├── src/
│   └── customer_churn/
├── tests/
├── notebooks/
├── models/
├── .github/workflows/
├── app.py
├── Dockerfile
├── Makefile
└── pyproject.toml

How to Run

1️⃣ Install Dependencies
pip install -e .[dev]

2️⃣ Train the Model
python -m src.customer_churn.pipeline

3️⃣ Run the API
uvicorn app:app --reload


API will be available at:

http://localhost:8000

4️⃣ Run with Docker

Build image:

docker build -t churn-api .


Run container:

docker run -p 8000:8000 churn-api

🔄 CI/CD

GitHub Actions automatically runs:

Unit tests

Linting

Coverage checks

Ensuring code quality and reliability on every push.

Model Registry

Models are logged and versioned using MLflow Model Registry.

Each training run stores:

Parameters

Metrics

Artifacts

Serialized pipeline

This enables reproducibility and production deployment management.

Production Considerations

Configurable decision threshold for business flexibility

Clear separation between probability prediction and decision policy

Explainability integrated for compliance and transparency

Containerized deployment for portability

Modular structure for scalability

👤 Author

Hasan Jahid
ハサン・ジャヒド
🇯🇵 日本語版
顧客解約予測システム（本番運用対応・MLOps設計）

本プロジェクトは、通信業界の顧客データを用いた解約予測モデルを構築し、
本番運用を想定したエンドツーエンドの機械学習システムを実装したものです。

データ前処理からAPIデプロイまで、
MLライフサイクル全体をMLOpsベストプラクティスに基づいて設計しています。

ビジネス課題

顧客解約（Churn）は、サブスクリプション型ビジネスにおいて
収益に大きな影響を与える重要指標です。

本システムは：

解約確率の予測

設定可能な閾値による意思決定

解約リスクの可視化

を実現します。

主な特徴

再現性のあるMLパイプライン設計

MLflowによる実験管理・モデル管理

設定可能な閾値ロジック（Decision Layer）

SHAPによるモデル解釈性

FastAPIによるREST API提供

Dockerによるコンテナ化

GitHub ActionsによるCI/CD

Pytestによるユニットテスト

システム構成

CSVデータ
   ↓
データ読み込み
   ↓
前処理
   ↓
機械学習パイプライン
   ↓
MLflow実験管理
   ↓
モデルレジストリ
   ↓
確率出力
   ↓
閾値判定（Decision Layer）
   ↓
SHAPによる説明
   ↓
FastAPI推論API
   ↓
Dockerコンテナ

モデル解釈性（Explainability）

SHAPを用いて以下を実現：

特徴量の重要度分析

個別予測の説明

ビジネスインサイト抽出

予測結果の透明性向上

実務利用を想定した説明可能なAI設計です。

テスト

以下を対象にユニットテストを実装：

前処理

モデル学習

パイプライン統合

閾値ロジック

設定値整合性確認

pytest

本番運用を想定した設計

予測と意思決定ロジックの分離

モデルのバージョン管理

コンテナベースのデプロイ

スケーラブルな構成

再現性のある実験管理