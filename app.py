from __future__ import annotations

import html
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from Analise_modelo import load_model
from gerar_painel_alertas import (
    ALERT_META,
    HIGH_ALERT_THRESHOLD,
    INPUT_FILE,
    MEDIUM_ALERT_THRESHOLD,
    MODEL_FILE,
    build_feature_frame,
    build_summary,
    canonicalize_input,
    classify_alert,
    format_currency,
    format_percent,
    read_input_file,
)


st.set_page_config(
    page_title="Painel de Atipicidade REIDI",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


ALERT_ORDER = sorted(ALERT_META, key=lambda label: ALERT_META[label]["rank"], reverse=True)

EXPORT_COLUMNS = [
    "ordem_fila",
    "empresa",
    "cnae",
    "atividade",
    "prob_nao_atipico",
    "nivel_alerta",
    "valor_beneficio",
]

DISPLAY_COLUMNS = {
    "ordem_fila": "Fila",
    "empresa": "Empresa",
    "cnae": "CNAE",
    "atividade": "Atividade",
    "prob_nao_atipico": "Prob. Nao/Atipico",
    "nivel_alerta": "Alerta",
    "valor_beneficio": "Beneficio",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #112036;
                --muted: #5f6b7a;
                --line: #d0d5dd;
                --paper: #fffdfa;
                --sand: #f7f3ea;
                --accent: #0f3d5e;
                --accent-soft: #174d73;
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(11, 110, 153, 0.12), transparent 28%),
                    linear-gradient(180deg, #f7f3ea 0%, #fcfbf7 48%, #ffffff 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1380px;
                padding-top: 1.8rem;
                padding-bottom: 2.5rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f9f7f0 0%, #ffffff 100%);
                border-right: 1px solid rgba(17, 32, 54, 0.08);
            }

            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p {
                color: var(--ink);
            }

            .hero-panel {
                background: linear-gradient(135deg, rgba(15, 61, 94, 0.98), rgba(20, 79, 117, 0.96));
                border-radius: 28px;
                padding: 2rem 2.1rem;
                box-shadow: 0 22px 50px rgba(17, 32, 54, 0.18);
                color: #ffffff;
                position: relative;
                overflow: hidden;
                margin-bottom: 1.1rem;
            }

            .hero-panel::after {
                content: "";
                position: absolute;
                inset: auto -80px -120px auto;
                width: 320px;
                height: 320px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.22) 0%, rgba(255, 255, 255, 0) 72%);
            }

            .hero-kicker {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.24);
                background: rgba(255, 255, 255, 0.09);
                font-size: 0.76rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.95rem;
            }

            .hero-title {
                font-family: Georgia, "Times New Roman", serif;
                font-size: clamp(2.1rem, 2.4vw, 2.8rem);
                line-height: 1.08;
                margin: 0 0 0.85rem 0;
                max-width: 880px;
            }

            .hero-copy {
                font-size: 1.05rem;
                line-height: 1.6;
                max-width: 980px;
                color: rgba(255, 255, 255, 0.92);
                margin: 0;
            }

            .hero-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
                margin-top: 1rem;
            }

            .hero-tag {
                padding: 0.5rem 0.8rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.14);
                font-size: 0.82rem;
            }

            .metric-card,
            .panel-card,
            .queue-card,
            .mini-card {
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(17, 32, 54, 0.08);
                border-radius: 22px;
                box-shadow: 0 12px 30px rgba(17, 32, 54, 0.06);
            }

            .metric-card {
                min-height: 146px;
                padding: 1.15rem 1.1rem 1rem;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.8rem;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                margin-bottom: 0.85rem;
            }

            .metric-value {
                color: var(--accent);
                font-size: clamp(1.65rem, 2vw, 2rem);
                font-weight: 700;
                line-height: 1.1;
            }

            .metric-sub {
                margin-top: 0.8rem;
                color: var(--muted);
                line-height: 1.45;
                font-size: 0.9rem;
            }

            .section-heading {
                margin-top: 1rem;
                margin-bottom: 0.55rem;
                color: var(--accent);
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.45rem;
                line-height: 1.2;
            }

            .section-copy {
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.65;
                margin-bottom: 1rem;
            }

            .panel-card {
                padding: 1.2rem 1.15rem 1.1rem;
                margin-bottom: 1rem;
            }

            .flow-step {
                min-height: 170px;
            }

            .flow-number {
                width: 2rem;
                height: 2rem;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(15, 61, 94, 0.1);
                color: var(--accent);
                font-weight: 700;
                margin-bottom: 0.85rem;
            }

            .panel-card h3,
            .queue-card h3 {
                margin: 0 0 0.55rem 0;
                color: var(--accent);
                font-size: 1.05rem;
                line-height: 1.3;
            }

            .panel-card p,
            .queue-card p,
            .mini-card p {
                margin: 0;
                color: var(--muted);
                line-height: 1.55;
                font-size: 0.93rem;
            }

            .queue-card {
                padding: 1.2rem 1.15rem 1rem;
                min-height: 184px;
            }

            .queue-label {
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }

            .queue-count {
                font-size: 1.85rem;
                font-weight: 700;
                color: var(--accent);
                margin-bottom: 0.35rem;
            }

            .queue-detail {
                margin-top: 0.5rem;
                color: var(--muted);
                font-size: 0.92rem;
            }

            .mini-card {
                padding: 0.95rem 1rem 0.9rem;
                min-height: 108px;
            }

            .mini-card strong {
                display: block;
                color: var(--accent);
                font-size: 1.25rem;
                margin-top: 0.4rem;
            }

            .table-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(17, 32, 54, 0.08);
                border-radius: 24px;
                box-shadow: 0 16px 40px rgba(17, 32, 54, 0.06);
                padding: 1rem 1rem 1.15rem;
            }

            .table-shell {
                overflow: auto;
                border-radius: 18px;
                border: 1px solid rgba(17, 32, 54, 0.08);
                background: #ffffff;
            }

            .ops-table {
                width: 100%;
                min-width: 1080px;
                border-collapse: collapse;
                font-size: 0.94rem;
            }

            .ops-table thead th {
                position: sticky;
                top: 0;
                z-index: 1;
                background: #f7f9fb;
                color: var(--accent);
                text-align: left;
                padding: 0.9rem 0.85rem;
                border-bottom: 2px solid rgba(17, 32, 54, 0.12);
                font-size: 0.82rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .ops-table tbody td {
                padding: 0.9rem 0.85rem;
                border-bottom: 1px solid rgba(17, 32, 54, 0.06);
                color: var(--ink);
                vertical-align: top;
            }

            .ops-table tbody tr:last-child td {
                border-bottom: none;
            }

            .table-rank {
                font-weight: 700;
                color: var(--accent);
            }

            .pill {
                display: inline-block;
                padding: 0.36rem 0.7rem;
                border-radius: 999px;
                border: 1px solid currentColor;
                font-size: 0.78rem;
                font-weight: 700;
                white-space: nowrap;
            }

            .mono {
                font-family: Consolas, "Courier New", monospace;
                font-size: 0.88rem;
                white-space: nowrap;
            }

            .clip {
                display: inline-block;
                max-width: 100%;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                vertical-align: top;
            }

            .table-note {
                margin-top: 0.85rem;
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.55;
            }

            div[data-testid="stFileUploader"] section {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 16px;
                border: 1px dashed rgba(15, 61, 94, 0.26);
            }

            .stButton > button,
            .stDownloadButton > button {
                background: linear-gradient(135deg, #0f3d5e, #174d73);
                color: #ffffff;
                border: none;
                border-radius: 999px;
                padding: 0.65rem 1rem;
                font-weight: 600;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background: linear-gradient(135deg, #0d3550, #123f5f);
                color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, subtext: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-sub">{html.escape(subtext)}</div>
    </div>
    """


def queue_card(label: str, count: int, total_benefit: float, avg_probability: float) -> str:
    meta = ALERT_META[label]
    return f"""
    <div class="queue-card" style="border-color:{meta['color']}; background:{meta['bg']};">
        <div class="queue-label" style="color:{meta['color']};">{html.escape(label)}</div>
        <div class="queue-count">{count} empresas</div>
        <div class="queue-detail">Beneficio total: {html.escape(format_currency(total_benefit))}</div>
        <div class="queue-detail">Prob. media de Nao/Atipico: {html.escape(format_percent(avg_probability))}</div>
    </div>
    """


def flow_card(number: int, title: str, description: str) -> str:
    return f"""
    <div class="panel-card flow-step">
        <div class="flow-number">{number}</div>
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(description)}</p>
    </div>
    """


def mini_card(label: str, value: str, description: str) -> str:
    return f"""
    <div class="mini-card">
        <p>{html.escape(label)}</p>
        <strong>{html.escape(value)}</strong>
        <p>{html.escape(description)}</p>
    </div>
    """


def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.getvalue()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(payload))
    return pd.read_csv(io.BytesIO(payload))


@st.cache_resource(show_spinner=False)
def get_model_bundle(model_path: str):
    return load_model(Path(model_path))


@st.cache_data(show_spinner=False)
def score_dataframe(input_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    model, metadata = get_model_bundle(str(MODEL_FILE.resolve()))
    features = build_feature_frame(input_df)
    probabilities = pd.DataFrame(model.predict_proba(features), columns=model.classes_)

    scored = input_df.copy()
    scored["prob_nao_atipico"] = probabilities.get("nao", 0.0)
    scored["prob_sim_tipico"] = probabilities.get("sim", 0.0)
    scored["nivel_alerta"] = scored["prob_nao_atipico"].map(classify_alert)
    scored["alert_rank"] = scored["nivel_alerta"].map(lambda label: ALERT_META[label]["rank"])
    scored["valor_beneficio"] = scored["valor_beneficio"].round(2)
    scored = scored.sort_values(
        ["alert_rank", "prob_nao_atipico", "valor_beneficio", "empresa"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    scored["ordem_fila"] = scored.index + 1
    scored["modelo_base"] = metadata.get("data_file", MODEL_FILE.name)
    return scored, str(metadata.get("data_file", MODEL_FILE.name))


def make_download_csv(scored: pd.DataFrame) -> bytes:
    export = scored[EXPORT_COLUMNS].copy()
    return export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def format_probability_column(probability: float) -> str:
    return format_percent(float(probability))


def hero(source_name: str, model_name: str, row_count: int) -> None:
    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-kicker">REIDI | Revisao tecnica assistida por modelo</div>
            <h1 class="hero-title">Painel de Revisao Tecnica por Atipicidade</h1>
            <p class="hero-copy">
                Interface Streamlit baseada no mock SVG para transformar a saida do modelo em uma fila
                operacional de revisao. Quanto maior a probabilidade de "Nao/Atipico", maior o nivel de
                alerta e mais cedo a empresa entra na triagem tecnica.
            </p>
            <div class="hero-meta">
                <span class="hero-tag">Fonte atual: {html.escape(source_name)}</span>
                <span class="hero-tag">Modelo: {html.escape(model_name)}</span>
                <span class="hero-tag">{row_count} empresas processadas</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(scored: pd.DataFrame) -> None:
    high_count = int((scored["nivel_alerta"] == ALERT_ORDER[0]).sum())
    medium_count = int((scored["nivel_alerta"] == ALERT_ORDER[1]).sum())
    low_count = int((scored["nivel_alerta"] == ALERT_ORDER[2]).sum())

    cards = [
        (
            "Empresas na fila",
            f"{len(scored)}",
            "Ordenadas por alerta, probabilidade prevista e valor do beneficio.",
        ),
        (
            "Beneficio total",
            format_currency(float(scored["valor_beneficio"].sum())),
            "Soma do lote atualmente exibido na fila de trabalho.",
        ),
        (
            "Prob. media de Nao/Atipico",
            format_percent(float(scored["prob_nao_atipico"].mean())),
            "Sinal medio de atipicidade estimado pelo classificador.",
        ),
        (
            "Fila alto / medio / baixo",
            f"{high_count} / {medium_count} / {low_count}",
            "Distribuicao por nivel de alerta para priorizacao tecnica.",
        ),
    ]

    columns = st.columns(4)
    for column, (label, value, subtext) in zip(columns, cards):
        with column:
            st.markdown(metric_card(label, value, subtext), unsafe_allow_html=True)


def render_flow() -> None:
    st.markdown('<div class="section-heading">Fluxo de uso</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-copy">
            O layout segue a estrutura do SVG: primeiro os indicadores executivos, depois a logica de
            priorizacao e por fim a tabela operacional para consulta e exportacao da fila.
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "Carregar o lote",
            "Receba um CSV ou XLSX com empresa, CNAE, atividade e valor do beneficio para processamento.",
        ),
        (
            "Pontuar a atipicidade",
            "O modelo combina descricao de atividade e hierarquia do CNAE para estimar a probabilidade de Nao/Atipico.",
        ),
        (
            "Gerar a fila de revisao",
            "A fila final sobe os casos com maior risco para alerta alto, seguido por medio e baixo.",
        ),
    ]

    columns = st.columns(3)
    for index, (column, (title, description)) in enumerate(zip(columns, steps), start=1):
        with column:
            st.markdown(flow_card(index, title, description), unsafe_allow_html=True)


def render_queue_summary(summary: pd.DataFrame) -> None:
    st.markdown('<div class="section-heading">Fila de alerta</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-copy">
            Regra usada neste deploy: alerta alto para probabilidade maior ou igual a
            <strong>{html.escape(format_percent(HIGH_ALERT_THRESHOLD))}</strong>, alerta medio entre
            <strong>{html.escape(format_percent(MEDIUM_ALERT_THRESHOLD))}</strong> e
            <strong>{html.escape(format_percent(HIGH_ALERT_THRESHOLD))}</strong>, e alerta baixo abaixo
            de <strong>{html.escape(format_percent(MEDIUM_ALERT_THRESHOLD))}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_lookup = {str(row["nivel_alerta"]): row for _, row in summary.iterrows()}
    columns = st.columns(3)
    for column, label in zip(columns, ALERT_ORDER):
        row = summary_lookup.get(label)
        if row is None:
            card_html = queue_card(label, 0, 0.0, 0.0)
        else:
            card_html = queue_card(
                label=label,
                count=int(row["empresas"]),
                total_benefit=float(row["beneficio_total"]),
                avg_probability=float(row["prob_media_nao"]),
            )
        with column:
            st.markdown(card_html, unsafe_allow_html=True)


def filter_dataframe(scored: pd.DataFrame) -> pd.DataFrame:
    st.markdown('<div class="section-heading">Filtros operacionais</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-copy">
            Use os filtros abaixo para refinar a fila antes de exportar o CSV ou revisar os registros na tabela.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.1, 1.2, 1.7])
    with col1:
        selected_alerts = st.multiselect(
            "Nivel de alerta",
            options=ALERT_ORDER,
            default=ALERT_ORDER,
        )
    with col2:
        min_probability = st.slider(
            "Probabilidade minima",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
    with col3:
        search_term = st.text_input("Buscar por empresa, CNAE ou atividade")

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

    filtered = filtered.reset_index(drop=True)
    return filtered


def render_filtered_snapshot(filtered: pd.DataFrame, total_rows: int) -> None:
    columns = st.columns(3)
    cards = [
        (
            "Empresas visiveis",
            f"{len(filtered)}",
            f"{len(filtered) / total_rows:.0%} da fila total apos aplicar os filtros.",
        ),
        (
            "Beneficio filtrado",
            format_currency(float(filtered["valor_beneficio"].sum())) if not filtered.empty else "R$ 0,00",
            "Montante financeiro representado pelos registros em tela.",
        ),
        (
            "Prob. media filtrada",
            format_percent(float(filtered["prob_nao_atipico"].mean())) if not filtered.empty else "0,00%",
            "Media das probabilidades da selecao corrente.",
        ),
    ]

    for column, (label, value, description) in zip(columns, cards):
        with column:
            st.markdown(mini_card(label, value, description), unsafe_allow_html=True)


def render_table(filtered: pd.DataFrame, source_name: str, model_name: str) -> None:
    st.markdown('<div class="section-heading">Tabela operacional</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-copy">
            Fonte em uso: <strong>{html.escape(source_name)}</strong> |
            Modelo: <strong>{html.escape(model_name)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("Nenhum registro atende aos filtros selecionados.")
        return

    rows_html: list[str] = []
    for _, row in filtered.iterrows():
        meta = ALERT_META[row["nivel_alerta"]]
        rows_html.append(
            (
                f'<tr style="background:{meta["bg"]}">'
                f'<td class="table-rank">{int(row["ordem_fila"])}</td>'
                f'<td><span class="clip" title="{html.escape(str(row["empresa"]), quote=True)}">{html.escape(str(row["empresa"]))}</span></td>'
                f'<td><span class="mono">{html.escape(str(row["cnae"]))}</span></td>'
                f'<td><span class="clip" title="{html.escape(str(row["atividade"]), quote=True)}">{html.escape(str(row["atividade"]))}</span></td>'
                f'<td>{html.escape(format_probability_column(row["prob_nao_atipico"]))}</td>'
                f'<td><span class="pill" style="color:{meta["color"]}; background:{meta["bg"]};">{html.escape(str(row["nivel_alerta"]))}</span></td>'
                f'<td>{html.escape(format_currency(float(row["valor_beneficio"])))}</td>'
                "</tr>"
            )
        )

    headers = "".join(f"<th>{html.escape(label)}</th>" for label in DISPLAY_COLUMNS.values())
    table_html = (
        '<div class="table-card">'
        '<div class="table-shell">'
        '<table class="ops-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
        '<div class="table-note">'
        "A coloracao acompanha o mock SVG para acelerar a leitura visual da fila: "
        "vermelho para alta prioridade, amarelo para media e verde para baixa."
        "</div>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_sidebar() -> tuple[object | None, bool]:
    with st.sidebar:
        st.markdown("## Entrada do lote")
        uploaded = st.file_uploader(
            "Enviar CSV ou XLSX",
            type=["csv", "xlsx", "xls"],
            help="Colunas esperadas: empresa, cnae, atividade e valor_beneficio.",
        )
        use_demo = st.checkbox("Usar base demo do projeto", value=uploaded is None)
        st.download_button(
            "Baixar CSV de exemplo",
            data=INPUT_FILE.read_bytes(),
            file_name=INPUT_FILE.name,
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("## Regras de alerta")
        st.write(f"Alto: probabilidade >= {format_percent(HIGH_ALERT_THRESHOLD)}")
        st.write(
            f"Medio: entre {format_percent(MEDIUM_ALERT_THRESHOLD)} e {format_percent(HIGH_ALERT_THRESHOLD)}"
        )
        st.write(f"Baixo: probabilidade < {format_percent(MEDIUM_ALERT_THRESHOLD)}")

        st.markdown("---")
        st.markdown("## Campos esperados")
        st.code("empresa\ncnae\natividade\nvalor_beneficio")

    return uploaded, use_demo


def load_source_data(uploaded, use_demo: bool) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded is not None:
        return load_uploaded_dataframe(uploaded), uploaded.name
    if use_demo:
        return read_input_file(INPUT_FILE), INPUT_FILE.name
    return None, None


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <h3>Nenhum lote carregado</h3>
            <p>
                Envie uma planilha no menu lateral ou marque a opcao de usar a base demo para visualizar
                o painel completo antes do deploy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    uploaded, use_demo = render_sidebar()
    raw_df, source_name = load_source_data(uploaded, use_demo)

    if raw_df is None or source_name is None:
        hero("aguardando envio", MODEL_FILE.name, 0)
        render_empty_state()
        return

    try:
        input_df = canonicalize_input(raw_df)
        if input_df.empty:
            st.error("O arquivo foi lido, mas nenhum registro valido restou apos a padronizacao.")
            return

        with st.spinner("Processando lote e montando a fila de alerta..."):
            scored, model_name = score_dataframe(input_df)
    except Exception as exc:
        st.error(f"Nao foi possivel processar o lote informado: {exc}")
        return

    summary = build_summary(scored)
    hero(source_name, model_name, len(scored))
    render_metrics(scored)
    render_flow()
    render_queue_summary(summary)

    st.markdown("")
    filtered = filter_dataframe(scored)
    render_filtered_snapshot(filtered, len(scored))

    action_col1, action_col2 = st.columns([1.25, 2.75])
    with action_col1:
        st.download_button(
            "Baixar fila filtrada",
            data=make_download_csv(filtered),
            file_name="fila_alertas_filtrada.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with action_col2:
        st.caption(
            "O CSV exportado respeita os filtros aplicados e pode ser usado como insumo para revisao tecnica."
        )

    render_table(filtered, source_name, model_name)

    with st.expander("Visualizar dados processados"):
        view = filtered[EXPORT_COLUMNS].rename(columns=DISPLAY_COLUMNS)
        st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander("Visualizar base de entrada padronizada"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
