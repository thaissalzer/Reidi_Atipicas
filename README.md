# Painel de Atipicidade REIDI

Interface em Streamlit para priorizacao de revisao tecnica com base na probabilidade prevista de `Nao/Atipico`.

## Arquivos principais

- `app.py`: ponto de entrada para execucao local e deploy no Streamlit Cloud.
- `requirements.txt`: dependencias do projeto.
- `.streamlit/config.toml`: configuracao visual base do app.
- `modelo_classificacao_reidi_v2.joblib`: modelo usado para pontuar os registros.
- `empresas_beneficio_demo.csv`: base demo para testar a interface.

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Colunas esperadas no arquivo de entrada

- `empresa`
- `cnae`
- `atividade`
- `valor_beneficio`

Aceita `CSV`, `XLSX` e `XLS`.

## Como publicar no Streamlit Cloud

1. Suba esta pasta para um repositorio no GitHub.
2. No Streamlit Cloud, clique em `New app`.
3. Selecione o repositorio e defina o arquivo principal como `app.py`.
4. Confirme o deploy.

## Observacao

Se nenhum arquivo for enviado, o app pode usar a base demo do projeto para exibicao inicial.
