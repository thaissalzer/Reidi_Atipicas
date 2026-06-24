from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from Analise_modelo import normalize_text
from gerar_painel_alertas import (
    ALERT_META,
    HIGH_ALERT_THRESHOLD,
    INPUT_FILE,
    MEDIUM_ALERT_THRESHOLD,
    MODEL_FILE,
    build_summary,
    canonicalize_input,
    format_currency,
    format_percent,
    read_input_file,
    score_companies,
)


st.set_page_config(
    page_title="Painel de Atipicidade REIDI",
    page_icon=":bar_chart:",
    layout="wide",
)


def alert_label(keyword: str) -> str:
    for label in ALERT_META:
        if keyword in normalize_text(label):
            return label
    raise ValueError(f"Rotulo de alerta nao encontrado para: {keyword}")


ALERT_HIGH = alert_label("alto")
ALERT_MEDIUM = alert_label("medio")
ALERT_LOW = alert_label("baixo")
ALERT_ORDER = [ALERT_HIGH, ALERT_MEDIUM, ALERT_LOW]


def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def render_metric_cards(scored: pd.DataFrame) -> None:
    alto = int((scored["nivel_alerta"] == ALERT_HIGH).sum())
    medio = int((scored["nivel_alerta"] == ALERT_MEDIUM).sum())
    baixo = int((scored["nivel_alerta"] == ALERT_LOW).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Empresas na fila", f"{len(scored)}")
    col2.metric("Beneficio total", format_currency(scored["valor_beneficio"].sum()))
    col3.metric("Prob. media de Nao/Atipico", format_percent(scored["prob_nao_atipico"].mean()))
    col4.metric("Fila alto / medio / baixo", f"{alto} / {medio} / {baixo}")


def render_queue_cards(summary: pd.DataFrame) -> None:
    columns = st.columns(3)
    for index, (_, row) in enumerate(summary.iterrows()):
        label = str(row["nivel_alerta"])
        meta = ALERT_META[label]
        with columns[index % 3]:
            st.markdown(
                f"""
                <div style="
                    border:2px solid {meta['color']};
                    background:{meta['bg']};
                    border-radius:18px;
                    padding:18px 18px 14px;
                    min-height:150px;
                ">
                    <div style="font-size:18px;font-weight:700;color:{meta['color']};">{label}</div>
                    <div style="font-size:30px;font-weight:700;color:#0f3d5e;margin-top:10px;">{int(row['empresas'])} empresas</div>
                    <div style="font-size:14px;color:#475467;margin-top:8px;">Beneficio total: {format_currency(row['beneficio_total'])}</div>
                    <div style="font-size:14px;color:#475467;margin-top:6px;">Prob. media de Nao/Atipico: {format_percent(row['prob_media_nao'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_flow() -> None:
    st.markdown("### Fluxo de uso")
    st.markdown(
        "O resultado do modelo gera uma fila de revisao tecnica. Quanto maior a probabilidade "
        "de `Nao/Atipico`, maior o nivel de alerta para a empresa daquele setor que recebeu o beneficio."
    )
    col1, col2, col3 = st.columns(3)
    col1.info("1. Carregar lote com empresa, CNAE, atividade e valor do beneficio.")
    col2.info("2. O modelo calcula a probabilidade de `Nao/Atipico` para cada empresa.")
    col3.info("3. A fila e priorizada em alerta alto, medio e baixo para revisao tecnica.")


def make_download_csv(scored: pd.DataFrame) -> bytes:
    export = scored[
        [
            "ordem_fila",
            "empresa",
            "cnae",
            "atividade",
            "prob_nao_atipico",
            "nivel_alerta",
            "valor_beneficio",
        ]
    ].copy()
    return export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def style_alerts(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        label = row["nivel_alerta"]
        meta = ALERT_META.get(label, {"bg": "#ffffff"})
        return [f"background-color: {meta['bg']}" for _ in row]

    styled = df.style.apply(row_style, axis=1)
    styled = styled.format(
        {
            "prob_nao_atipico": lambda value: format_percent(value),
            "valor_beneficio": lambda value: format_currency(value),
        }
    )
    return styled


def main() -> None:
    st.title("Painel de Revisao Tecnica por Atipicidade")
    st.caption(
        "Produto visivel para triagem de empresas beneficiadas. "
        "A fila e organizada pela probabilidade de `Nao/Atipico`."
    )

    with st.sidebar:
        st.header("Entrada")
        uploaded = st.file_uploader(
            "Enviar planilha CSV/XLSX",
            type=["csv", "xlsx", "xls"],
            help="Colunas esperadas: empresa, cnae, atividade, valor_beneficio.",
        )
        use_demo = st.checkbox("Usar base demo do projeto", value=uploaded is None)
        st.markdown("---")
        st.header("Regras de alerta")
        st.write(f"{ALERT_HIGH}: probabilidade >= {format_percent(HIGH_ALERT_THRESHOLD)}")
        st.write(
            f"{ALERT_MEDIUM}: {format_percent(MEDIUM_ALERT_THRESHOLD)} "
            f"ate {format_percent(HIGH_ALERT_THRESHOLD)}"
        )
        st.write(f"{ALERT_LOW}: probabilidade < {format_percent(MEDIUM_ALERT_THRESHOLD)}")

    if uploaded is not None:
        raw_df = load_uploaded_dataframe(uploaded)
        source_name = uploaded.name
    elif use_demo:
        raw_df = read_input_file(INPUT_FILE)
        source_name = INPUT_FILE.name
    else:
        st.info("Envie uma planilha ou marque a opcao de usar a base demo.")
        return

    try:
        input_df = canonicalize_input(raw_df)
        scored = score_companies(input_df, MODEL_FILE)
    except Exception as exc:
        st.error(f"Nao foi possivel processar o arquivo: {exc}")
        return

    summary = build_summary(scored)

    render_metric_cards(scored)
    st.markdown("")
    render_flow()
    st.markdown("### Fila de alerta")
    render_queue_cards(summary)

    st.markdown("### Filtros")
    col1, col2, col3 = st.columns([1.1, 1.1, 1.4])
    selected_alerts = col1.multiselect(
        "Nivel de alerta",
        options=ALERT_ORDER,
        default=ALERT_ORDER,
    )
    min_probability = col2.slider(
        "Probabilidade minima de Nao/Atipico",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    search_term = col3.text_input("Buscar empresa, CNAE ou atividade")

    filtered = scored.copy()
    if selected_alerts:
        filtered = filtered[filtered["nivel_alerta"].isin(selected_alerts)]
    filtered = filtered[filtered["prob_nao_atipico"] >= min_probability]
    if search_term.strip():
        search = search_term.strip().lower()
        mask = (
            filtered["empresa"].str.lower().str.contains(search, na=False)
            | filtered["cnae"].str.lower().str.contains(search, na=False)
            | filtered["atividade"].str.lower().str.contains(search, na=False)
        )
        filtered = filtered[mask]

    st.markdown("### Tabela operacional")
    st.caption(f"Fonte atual: {source_name} | Modelo: {MODEL_FILE.name}")

    view = filtered[
        [
            "ordem_fila",
            "empresa",
            "cnae",
            "atividade",
            "prob_nao_atipico",
            "nivel_alerta",
            "valor_beneficio",
        ]
    ].rename(
        columns={
            "ordem_fila": "Fila",
            "empresa": "Empresa",
            "cnae": "CNAE",
            "atividade": "Atividade",
            "prob_nao_atipico": "Probabilidade de Nao/Atipico",
            "nivel_alerta": "Nivel de alerta",
            "valor_beneficio": "Valor do beneficio",
        }
    )
    st.dataframe(style_alerts(view), use_container_width=True, hide_index=True)

    csv_bytes = make_download_csv(filtered)
    st.download_button(
        "Baixar fila filtrada em CSV",
        data=csv_bytes,
        file_name="fila_alertas_filtrada.csv",
        mime="text/csv",
        use_container_width=False,
    )

    with st.expander("Visualizar base recebida"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
