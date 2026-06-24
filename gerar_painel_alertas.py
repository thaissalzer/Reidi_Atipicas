from __future__ import annotations

import argparse
import html
import math
import re
import textwrap
from pathlib import Path

import pandas as pd

from Analise_modelo import load_model, normalize_text


MODEL_FILE = Path(__file__).with_name("modelo_classificacao_reidi_v2.joblib")
INPUT_FILE = Path(__file__).with_name("empresas_beneficio_demo.csv")
OUTPUT_HTML = Path(__file__).with_name("painel_alertas_atipicidade.html")
OUTPUT_SVG = Path(__file__).with_name("painel_alertas_atipicidade.svg")
OUTPUT_CSV = Path(__file__).with_name("fila_alertas_atipicidade.csv")

HIGH_ALERT_THRESHOLD = 0.80
MEDIUM_ALERT_THRESHOLD = 0.55

ALERT_META = {
    "Alerta alto": {"rank": 3, "color": "#b42318", "bg": "#fef3f2"},
    "Alerta médio": {"rank": 2, "color": "#b54708", "bg": "#fffaeb"},
    "Alerta baixo": {"rank": 1, "color": "#027a48", "bg": "#ecfdf3"},
}


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%".replace(".", ",")


def format_currency(value: float) -> str:
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(name))


def read_input_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def canonicalize_input(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {}
    for original in df.columns:
        normalized = normalize_column_name(original)
        if normalized == "empresa":
            column_map[original] = "empresa"
        elif normalized in {"cnae", "cnaeprincipal"}:
            column_map[original] = "cnae"
        elif normalized in {"atividade", "descricao"}:
            column_map[original] = "atividade"
        elif normalized in {"valorbeneficio", "valordobeneficio", "beneficio", "valor"}:
            column_map[original] = "valor_beneficio"

    parsed = df.rename(columns=column_map)
    required = ["empresa", "cnae", "atividade", "valor_beneficio"]
    missing = [column for column in required if column not in parsed.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Arquivo de entrada sem as colunas obrigatorias: {missing_text}. "
            "Esperado: empresa, cnae, atividade, valor_beneficio."
        )

    parsed = parsed[required].copy()
    parsed["empresa"] = parsed["empresa"].astype(str).str.strip()
    parsed["cnae"] = parsed["cnae"].astype(str).str.extract(r"(\d+)")[0].str.zfill(7)
    parsed["atividade"] = parsed["atividade"].astype(str).str.strip()
    parsed["valor_beneficio"] = pd.to_numeric(parsed["valor_beneficio"], errors="coerce")
    parsed = parsed.dropna(subset=["empresa", "cnae", "atividade", "valor_beneficio"]).reset_index(drop=True)
    return parsed


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()
    features["descricao_norm"] = df["atividade"].map(normalize_text)
    features["cnae_2"] = df["cnae"].str[:2]
    features["cnae_3"] = df["cnae"].str[:3]
    features["cnae_4"] = df["cnae"].str[:4]
    features["cnae_5"] = df["cnae"].str[:5]
    features["cnae_7"] = df["cnae"]
    return features


def classify_alert(prob_nao: float) -> str:
    if prob_nao >= HIGH_ALERT_THRESHOLD:
        return "Alerta alto"
    if prob_nao >= MEDIUM_ALERT_THRESHOLD:
        return "Alerta médio"
    return "Alerta baixo"


def score_companies(input_df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    model, metadata = load_model(model_path)
    features = build_feature_frame(input_df)
    probabilities = pd.DataFrame(model.predict_proba(features), columns=model.classes_)

    scored = input_df.copy()
    scored["prob_nao_atipico"] = probabilities["nao"]
    scored["prob_sim_tipico"] = probabilities["sim"]
    scored["nivel_alerta"] = scored["prob_nao_atipico"].map(classify_alert)
    scored["alert_rank"] = scored["nivel_alerta"].map(lambda label: ALERT_META[label]["rank"])
    scored["valor_beneficio"] = scored["valor_beneficio"].round(2)
    scored["ordem_fila"] = (
        scored.sort_values(
            ["alert_rank", "prob_nao_atipico", "valor_beneficio", "empresa"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
        .index
        + 1
    )
    scored = scored.sort_values(
        ["alert_rank", "prob_nao_atipico", "valor_beneficio", "empresa"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    scored["ordem_fila"] = scored.index + 1
    scored["modelo_base"] = metadata.get("data_file", model_path.name)
    return scored


def build_summary(scored: pd.DataFrame) -> pd.DataFrame:
    order = pd.CategoricalDtype(["Alerta alto", "Alerta médio", "Alerta baixo"], ordered=True)
    summary = (
        scored.groupby("nivel_alerta", observed=False)
        .agg(
            empresas=("empresa", "size"),
            beneficio_total=("valor_beneficio", "sum"),
            prob_media_nao=("prob_nao_atipico", "mean"),
        )
        .reset_index()
    )
    summary["nivel_alerta"] = summary["nivel_alerta"].astype(order)
    summary = summary.sort_values("nivel_alerta").reset_index(drop=True)
    return summary


def render_html(scored: pd.DataFrame, summary: pd.DataFrame, html_path: Path, source_name: str) -> None:
    total_benefit = scored["valor_beneficio"].sum()
    avg_probability = scored["prob_nao_atipico"].mean()
    high_count = int((scored["nivel_alerta"] == "Alerta alto").sum())
    medium_count = int((scored["nivel_alerta"] == "Alerta médio").sum())
    low_count = int((scored["nivel_alerta"] == "Alerta baixo").sum())

    rows_html = []
    for _, row in scored.iterrows():
        meta = ALERT_META[row["nivel_alerta"]]
        rows_html.append(
            f"""
            <tr style="background:{meta['bg']}">
              <td>{int(row['ordem_fila'])}</td>
              <td>{html.escape(row['empresa'])}</td>
              <td><code>{html.escape(row['cnae'])}</code></td>
              <td>{html.escape(row['atividade'])}</td>
              <td><strong>{format_percent(row['prob_nao_atipico'])}</strong></td>
              <td><span class="badge" style="color:{meta['color']};background:{meta['bg']};border-color:{meta['color']}">{html.escape(row['nivel_alerta'])}</span></td>
              <td>{format_currency(row['valor_beneficio'])}</td>
            </tr>
            """
        )

    queue_cards = []
    for _, row in summary.iterrows():
        meta = ALERT_META[str(row["nivel_alerta"])]
        queue_cards.append(
            f"""
            <div class="queue-card" style="background:{meta['bg']};border-color:{meta['color']}">
              <div class="queue-title" style="color:{meta['color']}">{html.escape(str(row['nivel_alerta']))}</div>
              <div class="queue-number">{int(row['empresas'])} empresas</div>
              <div class="queue-detail">Benefício total: {format_currency(row['beneficio_total'])}</div>
              <div class="queue-detail">Prob. média de Não/Atípico: {format_percent(row['prob_media_nao'])}</div>
            </div>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Painel de Revisão Técnica por Atipicidade</title>
  <style>
    :root {{
      --ink: #112036;
      --muted: #5f6b7a;
      --line: #d0d5dd;
      --card: #f8fafc;
      --accent: #0f3d5e;
      --accent-2: #0b6e99;
      --paper: #fffdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top right, rgba(11,110,153,0.15), transparent 28%),
        linear-gradient(180deg, #f7f3ea 0%, #ffffff 45%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15,61,94,0.96), rgba(11,110,153,0.9));
      color: white;
      padding: 28px;
      border-radius: 24px;
      box-shadow: 0 24px 60px rgba(17,32,54,0.18);
      margin-bottom: 22px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      max-width: 980px;
      font-size: 17px;
      line-height: 1.55;
      color: rgba(255,255,255,0.92);
    }}
    .meta {{
      margin-top: 14px;
      font-size: 14px;
      color: rgba(255,255,255,0.8);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
    }}
    .metric {{
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(17,32,54,0.08);
      border-radius: 20px;
      padding: 18px 18px 16px;
      box-shadow: 0 10px 25px rgba(17,32,54,0.06);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 10px;
      font-size: 28px;
      font-weight: 700;
      color: var(--accent);
    }}
    .metric-sub {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .section {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(17,32,54,0.08);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 12px 30px rgba(17,32,54,0.06);
      margin-bottom: 18px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 24px;
      color: var(--accent);
    }}
    .section p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 15px;
    }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .flow-step {{
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 18px;
      padding: 18px;
    }}
    .flow-step strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--accent);
      font-size: 16px;
    }}
    .queue-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .queue-card {{
      border: 2px solid var(--line);
      border-radius: 18px;
      padding: 18px;
    }}
    .queue-title {{
      font-size: 16px;
      font-weight: 700;
    }}
    .queue-number {{
      font-size: 26px;
      font-weight: 700;
      margin-top: 10px;
    }}
    .queue-detail {{
      margin-top: 8px;
      font-size: 14px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 14px;
    }}
    thead th {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 2px solid var(--line);
      color: var(--accent);
      background: #f7f9fb;
      position: sticky;
      top: 0;
    }}
    tbody td {{
      padding: 12px 10px;
      border-bottom: 1px solid rgba(17,32,54,0.08);
      vertical-align: top;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid rgba(17,32,54,0.08);
      border-radius: 18px;
      background: white;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid currentColor;
      font-size: 12px;
      font-weight: 700;
    }}
    .note {{
      margin-top: 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
    }}
    @media (max-width: 980px) {{
      .metrics, .flow, .queue-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Painel de Revisão Técnica por Probabilidade de “Não/Atípico”</h1>
      <p>
        Este painel transforma o resultado do modelo em uma fila operacional de revisão técnica.
        Quanto maior a probabilidade de “Não/Atípico”, maior o nível de alerta e mais cedo a empresa
        deve entrar na análise do benefício recebido.
      </p>
      <div class="meta">Fonte de entrada: {html.escape(source_name)} | Modelo: {html.escape(scored['modelo_base'].iloc[0])}</div>
    </section>

    <section class="metrics">
      <div class="metric">
        <div class="metric-label">Empresas na fila</div>
        <div class="metric-value">{len(scored)}</div>
        <div class="metric-sub">Ordenadas por alerta, probabilidade e valor do benefício</div>
      </div>
      <div class="metric">
        <div class="metric-label">Benefício total</div>
        <div class="metric-value">{format_currency(total_benefit)}</div>
        <div class="metric-sub">Base usada para o demonstrador visual</div>
      </div>
      <div class="metric">
        <div class="metric-label">Probabilidade média de Não/Atípico</div>
        <div class="metric-value">{format_percent(avg_probability)}</div>
        <div class="metric-sub">Sinal médio de atipicidade observado no lote</div>
      </div>
      <div class="metric">
        <div class="metric-label">Distribuição da fila</div>
        <div class="metric-value">{high_count} / {medium_count} / {low_count}</div>
        <div class="metric-sub">Alto, médio e baixo, respectivamente</div>
      </div>
    </section>

    <section class="section">
      <h2>Fluxo de uso</h2>
      <p>
        O resultado do modelo nao deve ser lido apenas como uma classificacao binaria. Ele gera uma
        fila de priorizacao para revisao tecnica, na qual a probabilidade de “Nao/Atipico” orienta
        o nivel de alerta e a ordem de tratamento das empresas que receberam o beneficio.
      </p>
      <div class="flow">
        <div class="flow-step">
          <strong>1. Carregar o lote</strong>
          Importar uma base com empresa, CNAE, atividade e valor do beneficio para pontuacao automatica.
        </div>
        <div class="flow-step">
          <strong>2. Pontuar a atipicidade</strong>
          O modelo estima a probabilidade de “Nao/Atipico” para cada atividade combinando texto e hierarquia do CNAE.
        </div>
        <div class="flow-step">
          <strong>3. Gerar a fila de revisao</strong>
          Probabilidades maiores sobem para alerta alto; probabilidades intermediarias vao para alerta medio; probabilidades menores vao para alerta baixo.
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Fila de revisao tecnica</h2>
      <p>
        Regra aplicada neste demonstrador: <strong>Alerta alto</strong> para probabilidade de “Nao/Atipico”
        maior ou igual a {format_percent(HIGH_ALERT_THRESHOLD)}, <strong>Alerta medio</strong> entre
        {format_percent(MEDIUM_ALERT_THRESHOLD)} e {format_percent(HIGH_ALERT_THRESHOLD)}, e
        <strong>Alerta baixo</strong> abaixo de {format_percent(MEDIUM_ALERT_THRESHOLD)}.
      </p>
      <div class="queue-grid">
        {''.join(queue_cards)}
      </div>
    </section>

    <section class="section">
      <h2>Tabela operacional</h2>
      <p>
        A tabela abaixo e o produto visivel para triagem. Ela mostra a empresa, o CNAE, a atividade,
        a probabilidade prevista de “Nao/Atipico”, o nivel de alerta e o valor do beneficio.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Fila</th>
              <th>Empresa</th>
              <th>CNAE</th>
              <th>Atividade</th>
              <th>Probabilidade de “Nao/Atipico”</th>
              <th>Nivel de alerta</th>
              <th>Valor do beneficio</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
      <div class="note">
        Nomes de empresa e valores do beneficio deste painel demo sao ilustrativos. A mecanica de priorizacao e reutilizavel em lotes reais.
      </div>
    </section>
  </div>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def shorten_text(value: str, width: int) -> str:
    return textwrap.shorten(str(value), width=width, placeholder="...")


def render_svg(scored: pd.DataFrame, summary: pd.DataFrame, svg_path: Path, source_name: str) -> None:
    width = 1640
    row_height = 44
    base_height = 420
    height = base_height + row_height * len(scored)

    def escape(value: str) -> str:
        return html.escape(str(value), quote=True)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><style>'
        '.title{font:700 34px Georgia, serif; fill:#ffffff;}'
        '.subtitle{font:400 16px Georgia, serif; fill:#e9f1f7;}'
        '.section{font:700 18px Georgia, serif; fill:#0f3d5e;}'
        '.metric-label{font:600 12px Arial, sans-serif; fill:#5f6b7a;}'
        '.metric-value{font:700 24px Arial, sans-serif; fill:#0f3d5e;}'
        '.body{font:400 13px Arial, sans-serif; fill:#112036;}'
        '.small{font:400 12px Arial, sans-serif; fill:#5f6b7a;}'
        '.table-head{font:700 12px Arial, sans-serif; fill:#0f3d5e;}'
        '.table-cell{font:400 12px Arial, sans-serif; fill:#112036;}'
        '</style></defs>',
        '<rect x="0" y="0" width="1640" height="{0}" fill="#f7f3ea"/>'.format(height),
        '<rect x="24" y="24" rx="24" ry="24" width="1592" height="128" fill="#0f3d5e"/>',
        '<text x="52" y="74" class="title">Painel de Revisao Tecnica por Atipicidade</text>',
        '<text x="52" y="106" class="subtitle">Quanto maior a probabilidade de "Nao/Atipico", maior o alerta para revisao tecnica da empresa que recebeu o beneficio.</text>',
        f'<text x="52" y="132" class="subtitle">Fonte: {escape(source_name)}</text>',
    ]

    metric_specs = [
        ("Empresas na fila", str(len(scored))),
        ("Beneficio total", format_currency(scored["valor_beneficio"].sum())),
        ("Prob. media de Nao/Atipico", format_percent(scored["prob_nao_atipico"].mean())),
        ("Fila alto/medio/baixo", f"{int((scored['nivel_alerta'] == 'Alerta alto').sum())} / {int((scored['nivel_alerta'] == 'Alerta médio').sum())} / {int((scored['nivel_alerta'] == 'Alerta baixo').sum())}"),
    ]

    for index, (label, value) in enumerate(metric_specs):
        x = 24 + index * 398
        svg_parts.append(f'<rect x="{x}" y="176" rx="20" ry="20" width="380" height="96" fill="#ffffff" stroke="#d0d5dd"/>')
        svg_parts.append(f'<text x="{x + 20}" y="206" class="metric-label">{escape(label)}</text>')
        svg_parts.append(f'<text x="{x + 20}" y="244" class="metric-value">{escape(value)}</text>')

    svg_parts.append('<text x="24" y="318" class="section">Fluxo de uso</text>')
    svg_parts.append('<text x="24" y="342" class="body">1. Carregar o lote | 2. Pontuar a probabilidade de "Nao/Atipico" | 3. Gerar fila de alerta alto, medio e baixo.</text>')

    for index, (_, row) in enumerate(summary.iterrows()):
        label = str(row["nivel_alerta"])
        meta = ALERT_META[label]
        x = 24 + index * 398
        svg_parts.append(
            f'<rect x="{x}" y="360" rx="18" ry="18" width="380" height="96" fill="{meta["bg"]}" stroke="{meta["color"]}" stroke-width="2"/>'
        )
        svg_parts.append(f'<text x="{x + 20}" y="392" class="section" fill="{meta["color"]}">{escape(label)}</text>')
        svg_parts.append(f'<text x="{x + 20}" y="418" class="body">{int(row["empresas"])} empresas | {format_currency(row["beneficio_total"])}</text>')
        svg_parts.append(f'<text x="{x + 20}" y="440" class="small">Prob. media de Nao/Atipico: {format_percent(row["prob_media_nao"])}</text>')

    table_y = 500
    svg_parts.append(f'<rect x="24" y="{table_y}" rx="18" ry="18" width="1592" height="{height - table_y - 24}" fill="#ffffff" stroke="#d0d5dd"/>')
    svg_parts.append(f'<text x="44" y="{table_y + 30}" class="section">Tabela operacional</text>')

    col_x = [44, 110, 330, 470, 1010, 1225, 1410]
    headers = ["Fila", "Empresa", "CNAE", "Atividade", "Prob. Nao/Atipico", "Alerta", "Beneficio"]
    header_y = table_y + 64
    svg_parts.append(f'<line x1="40" y1="{header_y + 10}" x2="1596" y2="{header_y + 10}" stroke="#d0d5dd"/>')
    for x, title in zip(col_x, headers):
        svg_parts.append(f'<text x="{x}" y="{header_y}" class="table-head">{escape(title)}</text>')

    for index, (_, row) in enumerate(scored.iterrows(), start=1):
        y = header_y + 22 + (index - 1) * row_height
        meta = ALERT_META[row["nivel_alerta"]]
        svg_parts.append(
            f'<rect x="36" y="{y - 18}" width="1568" height="{row_height - 2}" fill="{meta["bg"]}" opacity="0.95"/>'
        )
        values = [
            str(int(row["ordem_fila"])),
            shorten_text(row["empresa"], 30),
            row["cnae"],
            shorten_text(row["atividade"], 68),
            format_percent(row["prob_nao_atipico"]),
            row["nivel_alerta"],
            format_currency(row["valor_beneficio"]),
        ]
        for x, value in zip(col_x, values):
            svg_parts.append(f'<text x="{x}" y="{y + 6}" class="table-cell">{escape(value)}</text>')
        svg_parts.append(f'<line x1="40" y1="{y + 16}" x2="1596" y2="{y + 16}" stroke="#eaecf0"/>')

    svg_parts.append("</svg>")
    svg_path.write_text("".join(svg_parts), encoding="utf-8")


def save_scored_csv(scored: pd.DataFrame, output_path: Path) -> None:
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
    export.to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera um painel visivel de priorizacao por probabilidade de Nao/Atipico."
    )
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE, help="CSV ou Excel com empresa, cnae, atividade e valor_beneficio.")
    parser.add_argument("--model-file", type=Path, default=MODEL_FILE, help="Modelo joblib treinado.")
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML, help="Arquivo HTML do painel.")
    parser.add_argument("--output-svg", type=Path, default=OUTPUT_SVG, help="Arquivo SVG com o print do painel.")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="Arquivo CSV com a fila pontuada.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_df = canonicalize_input(read_input_file(args.input_file))
    scored = score_companies(input_df, args.model_file)
    summary = build_summary(scored)

    render_html(scored, summary, args.output_html, args.input_file.name)
    render_svg(scored, summary, args.output_svg, args.input_file.name)
    save_scored_csv(scored, args.output_csv)

    print(f"Painel HTML salvo em: {args.output_html}")
    print(f"Print SVG salvo em: {args.output_svg}")
    print(f"Fila CSV salva em: {args.output_csv}")


if __name__ == "__main__":
    main()
