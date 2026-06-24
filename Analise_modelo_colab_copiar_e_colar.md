# Google Colab - Copiar e Colar

Cole cada bloco abaixo em uma celula separada no Google Colab.

## Celula 1 - Instalacao

Na primeira execucao, o Colab vai reiniciar o runtime automaticamente. Depois da reconexao, rode a mesma celula mais uma vez; quando aparecer `Dependencias prontas.`, siga para a Celula 2.

```python
from pathlib import Path

SETUP_FLAG = Path("/content/.reidi_colab_deps_v2")

if not SETUP_FLAG.exists():
    %pip install -q --upgrade "numpy>=1.26,<3" "pandas>=2.2,<3" "scipy>=1.13,<2" "scikit-learn>=1.4,<2" "openpyxl>=3.1.5" "joblib>=1.5"
    SETUP_FLAG.touch()
    from google.colab import runtime
    runtime.restart_runtime()

print("Dependencias prontas.")
```

## Celula 2 - Imports e funcoes

```python
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
from google.colab import files
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

POSITIVE_LABEL = "sim"
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: str) -> str:
    label = normalize_text(str(value))
    if label not in {"sim", "nao"}:
        raise ValueError(f"Rotulo inesperado: {value!r}")
    return label


def display_label(value: str) -> str:
    return "Sim" if value == "sim" else "Nao"


def feature_columns() -> list[str]:
    return ["descricao_norm", "cnae_2", "cnae_3", "cnae_4", "cnae_5", "cnae_7"]


def prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["cnaePrincipal"] = (
        working["cnaePrincipal"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str.zfill(7)
    )
    working["descricao"] = working["descricao"].astype(str).str.strip()
    working["descricao_norm"] = working["descricao"].map(normalize_text)
    working["cnae_2"] = working["cnaePrincipal"].str[:2]
    working["cnae_3"] = working["cnaePrincipal"].str[:3]
    working["cnae_4"] = working["cnaePrincipal"].str[:4]
    working["cnae_5"] = working["cnaePrincipal"].str[:5]
    working["cnae_7"] = working["cnaePrincipal"]
    return working


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 3:
        raise ValueError("O arquivo precisa ter pelo menos 3 colunas.")

    raw_columns = list(df.columns)
    df = df.rename(
        columns={
            raw_columns[0]: "cnaePrincipal",
            raw_columns[1]: "descricao",
            raw_columns[2]: "label",
        }
    )
    df = df[["cnaePrincipal", "descricao", "label"]].dropna().copy()
    df = prepare_base_frame(df)
    df["label"] = df["label"].map(normalize_label)
    return df.reset_index(drop=True)


def build_inference_frame(cnae: str, description: str) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "cnaePrincipal": str(cnae).strip(),
                "descricao": str(description).strip(),
            }
        ]
    )
    return prepare_base_frame(df)


def build_inference_dataframe(raw_df: pd.DataFrame, cnae_column: str, description_column: str) -> pd.DataFrame:
    df = raw_df[[cnae_column, description_column]].copy()
    df = df.rename(columns={cnae_column: "cnaePrincipal", description_column: "descricao"})
    return prepare_base_frame(df)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    strip_accents="unicode",
                    min_df=1,
                    sublinear_tf=True,
                ),
                "descricao_norm",
            ),
            (
                "cnae",
                OneHotEncoder(handle_unknown="ignore"),
                ["cnae_2", "cnae_3", "cnae_4", "cnae_5", "cnae_7"],
            ),
        ]
    )
    classifier = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.0, dual="auto"),
        cv=3,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=POSITIVE_LABEL,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["nao", "sim"]).ravel()
    return {
        "accuracy": float(accuracy),
        "precision_sim": float(precision),
        "recall_sim": float(recall),
        "f1_sim": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def cross_validate_model(df: pd.DataFrame, folds: int, seed: int) -> tuple[list[dict], dict]:
    X = df[feature_columns()]
    y = df["label"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_metrics: list[dict] = []
    for train_idx, test_idx in cv.split(X, y):
        model = build_pipeline()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = model.predict(X.iloc[test_idx])
        fold_metrics.append(compute_metrics(y.iloc[test_idx], pd.Series(fold_pred, index=y.iloc[test_idx].index)))

    pred = cross_val_predict(build_pipeline(), X, y, cv=cv, method="predict")
    global_metrics = compute_metrics(y, pd.Series(pred, index=y.index))
    return fold_metrics, global_metrics


def print_evaluation(df: pd.DataFrame, folds: int, seed: int) -> dict[str, float | int]:
    fold_metrics, global_metrics = cross_validate_model(df, folds=folds, seed=seed)

    baseline_label = df["label"].mode().iloc[0]
    baseline_pred = pd.Series([baseline_label] * len(df), index=df.index)
    baseline_metrics = compute_metrics(df["label"], baseline_pred)

    print("Avaliacao por validacao cruzada")
    print(f"- linhas: {len(df)}")
    print(f"- folds: {folds}")
    print(f"- seed: {seed}")
    print(f"- baseline_majoritaria: {display_label(baseline_label)}")
    print(f"- baseline_accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"- modelo_accuracy: {global_metrics['accuracy']:.4f}")
    print(f"- modelo_precision_sim: {global_metrics['precision_sim']:.4f}")
    print(f"- modelo_recall_sim: {global_metrics['recall_sim']:.4f}")
    print(f"- modelo_f1_sim: {global_metrics['f1_sim']:.4f}")
    print(
        "- matriz_confusao: "
        f"TP={global_metrics['tp']} TN={global_metrics['tn']} "
        f"FP={global_metrics['fp']} FN={global_metrics['fn']}"
    )
    for index, metrics in enumerate(fold_metrics, start=1):
        print(
            f"  fold_{index}: accuracy={metrics['accuracy']:.4f} "
            f"f1_sim={metrics['f1_sim']:.4f}"
        )
    return global_metrics


def train_and_save(df: pd.DataFrame, model_path: Path, folds: int, seed: int) -> tuple[Pipeline, dict]:
    metrics = print_evaluation(df, folds=folds, seed=seed)
    model = build_pipeline()
    model.fit(df[feature_columns()], df["label"])
    metadata = {
        "rows": len(df),
        "folds": folds,
        "seed": seed,
        "metrics": metrics,
    }
    payload = {
        "model": model,
        "metadata": metadata,
    }
    joblib.dump(payload, model_path)
    print(f"Modelo salvo em: {model_path}")
    return model, metadata


def load_model(model_path: Path) -> tuple[Pipeline, dict]:
    payload = joblib.load(model_path)
    return payload["model"], payload.get("metadata", {})


def predict_one(model: Pipeline, cnae: str, description: str) -> dict[str, float | str]:
    X = build_inference_frame(cnae, description)[feature_columns()]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    class_to_proba = dict(zip(model.classes_, proba))

    result = {
        "previsao": display_label(pred),
        "prob_sim": float(class_to_proba.get("sim", 0.0)),
        "prob_nao": float(class_to_proba.get("nao", 0.0)),
    }
    print(f"Previsao: {result['previsao']}")
    print(f"- prob_sim: {result['prob_sim']:.4f}")
    print(f"- prob_nao: {result['prob_nao']:.4f}")
    return result


def predict_dataframe(model: Pipeline, raw_df: pd.DataFrame, cnae_column: str, description_column: str) -> pd.DataFrame:
    inference_df = build_inference_dataframe(raw_df, cnae_column=cnae_column, description_column=description_column)
    X = inference_df[feature_columns()]
    pred = model.predict(X)
    proba = model.predict_proba(X)
    class_to_index = {label: index for index, label in enumerate(model.classes_)}

    result = raw_df.copy()
    result["previsao"] = [display_label(value) for value in pred]
    result["prob_sim"] = proba[:, class_to_index.get("sim", 0)]
    result["prob_nao"] = proba[:, class_to_index.get("nao", 0)]
    return result


def read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Formato nao suportado. Use .xlsx ou .csv.")
```

## Celula 3 - Upload da base de treinamento

```python
uploaded = files.upload()
xlsx_files = [name for name in uploaded if name.lower().endswith(".xlsx")]

if not xlsx_files:
    raise ValueError("Envie pelo menos um arquivo .xlsx.")

if len(xlsx_files) > 1:
    print(f"Arquivos encontrados: {xlsx_files}. Usando o primeiro arquivo.")

DATA_FILE = Path(xlsx_files[0])
print(f"Arquivo selecionado: {DATA_FILE}")
```

## Celula 4 - Avaliacao do modelo

```python
FOLDS = 5
SEED = 42

df = load_dataframe(DATA_FILE)
display(df.head())
print(f"Linhas validas: {len(df)}")
print_evaluation(df, folds=FOLDS, seed=SEED)
```

## Celula 5 - Treino e salvamento

```python
MODEL_FILE = Path("modelo_classificacao_reidi_colab.joblib")
model, metadata = train_and_save(df, model_path=MODEL_FILE, folds=FOLDS, seed=SEED)
metadata
```

## Celula 6 - Download do modelo

```python
files.download(str(MODEL_FILE))
```

## Celula 7 - Previsao manual

```python
CNAE = "3511501"
DESCRICAO = "geracao de energia eletrica"

predict_one(model, cnae=CNAE, description=DESCRICAO)
```

## Celula 8 - Classificacao em lote

```python
uploaded_batch = files.upload()
valid_batch_files = [name for name in uploaded_batch if name.lower().endswith((".xlsx", ".csv"))]

if not valid_batch_files:
    raise ValueError("Envie um arquivo .xlsx ou .csv para classificacao em lote.")

BATCH_FILE = Path(valid_batch_files[0])
batch_df = read_tabular_file(BATCH_FILE)
print(f"Colunas encontradas: {list(batch_df.columns)}")

CNAE_COLUMN = batch_df.columns[0]
DESCRICAO_COLUMN = batch_df.columns[1]

resultado = predict_dataframe(
    model,
    batch_df,
    cnae_column=CNAE_COLUMN,
    description_column=DESCRICAO_COLUMN,
)

RESULT_FILE = Path("predicoes_reidi.xlsx")
resultado.to_excel(RESULT_FILE, index=False)
display(resultado.head())
files.download(str(RESULT_FILE))
```

## Celula 9 - Carregar modelo ja treinado

```python
uploaded_model = files.upload()
joblib_files = [name for name in uploaded_model if name.lower().endswith(".joblib")]

if not joblib_files:
    raise ValueError("Envie um arquivo .joblib.")

loaded_model_file = Path(joblib_files[0])
model, metadata = load_model(loaded_model_file)
print("Metadata do modelo:")
metadata
```
