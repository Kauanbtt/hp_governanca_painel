# HP — Painel Executivo do Piloto (Governança & Analytics)

Este projeto entrega um **dashboard público em Streamlit (somente leitura)** que consolida:
- Monitoramento do piloto (precisão, recall, FPR/FNR, tickets, devoluções, metas vs. realizado, tendência temporal)
- Operação (matriz de confusão simulada, tempo de detecção, fila de revisão humana e taxa de override)
- Fairness & Governança (desempenho por corte, diferenças de taxas, sinal de deriva, **Model Card** e **LGPD mini**)
- Negócio & ROI (**Com Ação vs. Sem Ação**, ROI/Payback e recomendação **go/no-go**)
- Mapa/Heatmap com incidência regional de suspeitas
- Downloads (CSV/PNG/PDF) e **Resumo executivo** gerado pelo painel
- Campo de **Anotações/Insights**

> Baseado e estendido a partir de *painel-executivo-piloto.py*.

## Como executar localmente

1. Crie um ambiente (opcional, mas recomendado):
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Linux/Mac
   # ou
   python -m venv .venv && .venv\Scripts\activate   # Windows
   ```

2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Rode o app:
   ```bash
   streamlit run app.py
   ```

4. Para publicar, você pode usar:
   - **Streamlit Community Cloud** (gratuito) enviando estes arquivos para um repositório no GitHub.
   - **Streamlit Enterprise** / servidor interno.

## Estrutura
```
hp_governanca_painel/
  ├── app.py
  ├── requirements.txt
  └── README.md
```

## Observações
- Os dados são **simulados** para fins de protótipo (podem ser substituídos por fonte real).
- As exportações **CSV/PNG/PDF** estão ativas (PNG via `kaleido`, PDF via `reportlab`).
- O painel inclui explicabilidade simples de modelo (regressão logística com pesos) e elementos de governança diretamente na UI.
