from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


DATA_FILE = Path(__file__).with_name("classificacao_manual_modelagem.xlsx")
MODEL_FILE = Path(__file__).with_name("modelo_classificacao_reidi.joblib")
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
    return "Sim" if value == "sim" else "N\u00e3o"


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
    df["cnaePrincipal"] = (
        df["cnaePrincipal"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str.zfill(7)
    )
    df["descricao"] = df["descricao"].astype(str).str.strip()
    df["descricao_norm"] = df["descricao"].map(normalize_text)
    df["label"] = df["label"].map(normalize_label)
    df["cnae_2"] = df["cnaePrincipal"].str[:2]
    df["cnae_3"] = df["cnaePrincipal"].str[:3]
    df["cnae_4"] = df["cnaePrincipal"].str[:4]
    df["cnae_5"] = df["cnaePrincipal"].str[:5]
    df["cnae_7"] = df["cnaePrincipal"]
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
    df["cnaePrincipal"] = df["cnaePrincipal"].str.extract(r"(\d+)")[0].str.zfill(7)
    df["descricao_norm"] = df["descricao"].map(normalize_text)
    df["cnae_2"] = df["cnaePrincipal"].str[:2]
    df["cnae_3"] = df["cnaePrincipal"].str[:3]
    df["cnae_4"] = df["cnaePrincipal"].str[:4]
    df["cnae_5"] = df["cnaePrincipal"].str[:5]
    df["cnae_7"] = df["cnaePrincipal"]
    return df


def feature_columns() -> list[str]:
    return ["descricao_norm", "cnae_2", "cnae_3", "cnae_4", "cnae_5", "cnae_7"]


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


def train_and_save(df: pd.DataFrame, model_path: Path, folds: int, seed: int) -> Pipeline:
    metrics = print_evaluation(df, folds=folds, seed=seed)
    model = build_pipeline()
    model.fit(df[feature_columns()], df["label"])
    payload = {
        "model": model,
        "metadata": {
            "data_file": DATA_FILE.name,
            "rows": len(df),
            "folds": folds,
            "seed": seed,
            "metrics": metrics,
        },
    }
    joblib.dump(payload, model_path)
    print(f"Modelo salvo em: {model_path}")
    return model


def load_model(model_path: Path) -> tuple[Pipeline, dict]:
    payload = joblib.load(model_path)
    return payload["model"], payload.get("metadata", {})


def predict_one(model: Pipeline, cnae: str, description: str) -> None:
    X = build_inference_frame(cnae, description)[feature_columns()]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    class_to_proba = dict(zip(model.classes_, proba))

    print(f"Previsao: {display_label(pred)}")
    print(f"- prob_sim: {class_to_proba.get('sim', 0.0):.4f}")
    print(f"- prob_nao: {class_to_proba.get('nao', 0.0):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina e usa um classificador com pandas + scikit-learn para prever Sim/Nao."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DATA_FILE,
        help="Arquivo xlsx com colunas cnaePrincipal, Descricao e Dentro_escopo.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=MODEL_FILE,
        help="Arquivo joblib usado para salvar ou carregar o modelo treinado.",
    )
    subparsers = parser.add_subparsers(dest="command")

    evaluate_parser = subparsers.add_parser("evaluate", help="Avalia o modelo com validacao cruzada.")
    evaluate_parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    evaluate_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    train_parser = subparsers.add_parser("train", help="Treina e salva o modelo final.")
    train_parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    predict_parser = subparsers.add_parser("predict", help="Faz previsao para um novo registro.")
    predict_parser.add_argument("--cnae", required=True, help="CNAE principal com 7 digitos.")
    predict_parser.add_argument("--description", required=True, help="Descricao da atividade.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        if not args.model_file.exists():
            parser.error(f"Modelo nao encontrado em {args.model_file}. Rode o comando train antes.")
        model, _ = load_model(args.model_file)
        predict_one(model, cnae=args.cnae, description=args.description)
        return

    df = load_dataframe(args.data_file)

    if args.command == "evaluate":
        print_evaluation(df, folds=args.folds, seed=args.seed)
        return

    if args.command == "train":
        train_and_save(df, model_path=args.model_file, folds=args.folds, seed=args.seed)
        return

    train_and_save(df, model_path=args.model_file, folds=DEFAULT_FOLDS, seed=DEFAULT_SEED)


if __name__ == "__main__":
    main()
