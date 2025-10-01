\
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

# --- Utils: Downloads ---
def download_csv(df, filename="dados.csv", label="Baixar CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

def download_plotly_png(fig, filename="grafico.png", label="Baixar PNG"):
    try:
        png_bytes = fig.to_image(format="png", width=1280, height=720, scale=2)
        st.download_button(label, png_bytes, file_name=filename, mime="image/png")
    except Exception as e:
        st.warning("Para exportar PNG, instale a dependência 'kaleido'. Detalhe: " + str(e))

def generate_pdf_summary(kpis: dict, insight_text: str) -> bytes:
    """Gera um PDF simples de 1 página com KPIs e um resumo executivo."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2*cm, "Resumo Executivo — Piloto HP (Governança & Analytics)")

    c.setFont("Helvetica", 11)
    y = height - 3.2*cm
    for k, v in kpis.items():
        c.drawString(2*cm, y, f"- {k}: {v}")
        y -= 0.6*cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y - 0.4*cm, "Insights")
    c.setFont("Helvetica", 11)
    text_obj = c.beginText(2*cm, y - 1.0*cm)
    for line in insight_text.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# --- Mock Data Generation (replace with real data in production) ---
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-06-30", freq="D")
regions = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
ufs = ["SP", "RJ", "MG", "BA", "RS", "PE", "PR", "SC", "GO", "DF"]
prod_lines = ["Cartucho", "Toner"]
channels = ["E-commerce", "Loja", "Suporte"]
orig_gen = ["Original", "Genérico"]
severities = ["Baixa", "Média", "Alta", "Crítica"]
tipos_erro = ["vazamento", "impressão falha", "manchas", "erro desconhecido"]

n = 3000
df = pd.DataFrame({
    "Data": np.random.choice(dates, n),
    "Região": np.random.choice(regions, n),
    "UF": np.random.choice(ufs, n),
    "Linha de Produto": np.random.choice(prod_lines, n),
    "Canal": np.random.choice(channels, n),
    "Tipo": np.random.choice(orig_gen, n),
    "Severidade": np.random.choice(severities, n),
    "Precisão": np.random.uniform(0.7, 0.99, n),
    "Recall": np.random.uniform(0.6, 0.98, n),
    "FPR": np.random.uniform(0.01, 0.2, n),
    "FNR": np.random.uniform(0.01, 0.2, n),
    "Tickets": np.random.randint(1, 10, n),
    "Devoluções": np.random.randint(0, 5, n),
    "Override": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "Tempo Detecção (min)": np.random.uniform(1, 60, n),
    "Com Ação": np.random.choice([True, False], n, p=[0.6, 0.4]),
    "NPS": np.random.randint(0, 100, n),
    "Custo Suporte": np.random.uniform(10, 120, n),
    "Suspeita": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    # features para explicabilidade (Sprint 2)
    "TipoErro": np.random.choice(tipos_erro, n),
    "TempoUsoDias": np.random.uniform(1, 90, n),
    "RegistradoHP": np.random.choice([0, 1], n, p=[0.3, 0.7]),
})

# --- Sidebar Filters ---
st.sidebar.header("Filtros")
periodo = st.sidebar.date_input("Período", [df["Data"].min(), df["Data"].max()])
regiao = st.sidebar.multiselect("Região/UF", sorted(df["Região"].unique()), default=sorted(df["Região"].unique()))
uf = st.sidebar.multiselect("UF", sorted(df["UF"].unique()), default=sorted(df["UF"].unique()))
linha_produto = st.sidebar.multiselect("Linha de Produto", prod_lines, default=prod_lines)
canal = st.sidebar.multiselect("Canal", channels, default=channels)
tipo = st.sidebar.multiselect("Original x Genérico", orig_gen, default=orig_gen)
severidade = st.sidebar.multiselect("Severidade do Chamado", severities, default=severities)

# Parâmetros de negócio para ROI
st.sidebar.header("Parâmetros de ROI")
custo_programa_mensal = st.sidebar.number_input("Custo do Programa (R$/mês)", min_value=0.0, value=5000.0, step=500.0)
valor_por_devolucao_ev = st.sidebar.number_input("Economia por Devolução Evitada (R$)", min_value=0.0, value=50.0, step=10.0)

# --- Filter Data ---
df_filt = df[
    (df["Data"] >= pd.to_datetime(periodo[0])) &
    (df["Data"] <= pd.to_datetime(periodo[1])) &
    (df["Região"].isin(regiao)) &
    (df["UF"].isin(uf)) &
    (df["Linha de Produto"].isin(linha_produto)) &
    (df["Canal"].isin(canal)) &
    (df["Tipo"].isin(tipo)) &
    (df["Severidade"].isin(severidade))
].copy()

# --- Title & Navigation ---
st.title("Painel Executivo do Piloto — Governança, Ética & Impacto de Negócio")
page = st.sidebar.radio(
    "Navegação",
    [
        "Visão Geral",
        "Detecção & Operação",
        "Fairness & Governança",
        "Negócio & ROI",
        "Mapa Regional",
        "Downloads & Evidências"
    ]
)

# --- Helper KPIs ---
def kpi_overview(df_):
    kpis = {
        "Precisão média": f"{df_['Precisão'].mean():.2%}" if len(df_) else "N/A",
        "Recall médio": f"{df_['Recall'].mean():.2%}" if len(df_) else "N/A",
        "FPR médio": f"{df_['FPR'].mean():.2%}" if len(df_) else "N/A",
        "FNR médio": f"{df_['FNR'].mean():.2%}" if len(df_) else "N/A",
        "Tickets": int(df_['Tickets'].sum()) if len(df_) else 0,
        "Devoluções": int(df_['Devoluções'].sum()) if len(df_) else 0,
    }
    return kpis

def resumo_executivo(df_):
    if len(df_) == 0:
        return "Sem dados no período/recortes selecionados."
    total_tickets = int(df_["Tickets"].sum())
    total_devol = int(df_["Devoluções"].sum())
    nps_medio = df_["NPS"].mean()
    reg_counts = df_["Região"].value_counts()
    regioes_top = reg_counts.idxmax() if not reg_counts.empty else "N/A"
    return (
        f"- Total de tickets: {total_tickets}\n"
        f"- Total de devoluções: {total_devol}\n"
        f"- NPS médio: {nps_medio:.1f}\n"
        f"- Região com mais ocorrências: {regioes_top}\n"
    )

# --- PAGES ---
if page == "Visão Geral":
    st.header("KPIs do Piloto")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Precisão", f"{df_filt['Precisão'].mean():.2%}" if len(df_filt) else "N/A")
    col2.metric("Recall", f"{df_filt['Recall'].mean():.2%}" if len(df_filt) else "N/A")
    col3.metric("FPR", f"{df_filt['FPR'].mean():.2%}" if len(df_filt) else "N/A")
    col4.metric("FNR", f"{df_filt['FNR'].mean():.2%}" if len(df_filt) else "N/A")
    col5.metric("Tickets", int(df_filt['Tickets'].sum()) if len(df_filt) else 0)
    col6.metric("Devoluções", int(df_filt['Devoluções'].sum()) if len(df_filt) else 0)

    st.subheader("Metas vs. Realizado")
    metas = {"Precisão": 0.90, "Recall": 0.85, "FPR": 0.05, "FNR": 0.05}
    kpi_real = {
        "Precisão": df_filt["Precisão"].mean() if len(df_filt) else 0.0,
        "Recall": df_filt["Recall"].mean() if len(df_filt) else 0.0,
        "FPR": df_filt["FPR"].mean() if len(df_filt) else 0.0,
        "FNR": df_filt["FNR"].mean() if len(df_filt) else 0.0,
    }
    meta_df = pd.DataFrame({
        "KPI": list(metas.keys()),
        "Meta": list(metas.values()),
        "Realizado": [kpi_real[k] for k in metas]
    })
    fig_meta = px.bar(meta_df, x="KPI", y=["Meta", "Realizado"], barmode="group")
    st.plotly_chart(fig_meta, use_container_width=True)
    download_plotly_png(fig_meta, filename="metas_vs_realizado.png", label="Baixar PNG — Metas vs. Realizado")

    st.subheader("Tendência Temporal")
    if len(df_filt):
        df_time = df_filt.copy()
        df_time["Mes"] = df_time["Data"].dt.to_period("M").astype(str)
        agg = df_time.groupby("Mes").agg(
            Tickets=("Tickets","sum"),
            Devoluções=("Devoluções","sum"),
            Precisão=("Precisão","mean"),
            Recall=("Recall","mean")
        ).reset_index()
        fig_trend = px.line(agg, x="Mes", y=["Tickets", "Devoluções"], markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        download_plotly_png(fig_trend, filename="tendencia_temporal.png", label="Baixar PNG — Tendência")
    else:
        st.info("Sem dados para o recorte atual.")

elif page == "Detecção & Operação":
    st.header("Matriz de Confusão (simulada)")
    if len(df_filt):
        y_true = df_filt["Com Ação"].astype(bool)
        y_pred = (df_filt["Suspeita"] > 0)
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        cm_df = pd.DataFrame(cm, index=["Sem Ação (Real)", "Com Ação (Real)"], columns=["Pred: Sem", "Pred: Com"])
        st.dataframe(cm_df)
    else:
        st.info("Sem dados para o recorte atual.")

    st.subheader("Tempo Médio de Detecção")
    st.metric("Tempo Médio (min)", f"{df_filt['Tempo Detecção (min)'].mean():.1f}" if len(df_filt) else "N/A")

    st.subheader("Fila de Revisão Humana (HITL)")
    fila = df_filt[df_filt["Override"] == 1].copy()
    st.write(f"Chamados em revisão: {len(fila)}")
    st.dataframe(fila[["Data", "Região", "UF", "Linha de Produto", "Canal", "Severidade"]].head(15))
    download_csv(fila, filename="fila_revisao.csv", label="Baixar CSV — Fila Revisão")

    st.subheader("Taxa de Override")
    taxa_override = (fila.shape[0] / max(1, df_filt.shape[0])) if len(df_filt) else 0.0
    st.metric("Taxa de Override", f"{taxa_override:.2%}")

elif page == "Fairness & Governança":
    st.header("Desempenho por Corte")
    corte = st.selectbox("Corte", ["Região", "Canal"])
    if len(df_filt):
        corte_df = df_filt.groupby(corte).agg({
            "Precisão": "mean",
            "Recall": "mean",
            "FPR": "mean",
            "FNR": "mean"
        }).reset_index()
        st.dataframe(corte_df)
        fig_ft = px.bar(corte_df, x=corte, y=["FPR","FNR"], barmode="group", title="Diferenças de Taxas (FPR/FNR)")
        st.plotly_chart(fig_ft, use_container_width=True)
        download_plotly_png(fig_ft, filename="diferencas_taxas.png", label="Baixar PNG — Diferenças de Taxas")
    else:
        st.info("Sem dados para o recorte atual.")

    st.subheader("Sinal de Deriva")
    if len(df_filt):
        df_m = df_filt.copy()
        df_m["Mes"] = df_m["Data"].dt.to_period("M")
        drift = df_m.groupby("Mes")["Precisão"].mean().diff().fillna(0)
        drift_df = drift.reset_index()
        drift_df["Mes"] = drift_df["Mes"].astype(str)
        fig_drift = px.line(drift_df, x="Mes", y="Precisão", title="Variação de Precisão (Δ mês a mês)")
        st.plotly_chart(fig_drift, use_container_width=True)
        download_plotly_png(fig_drift, filename="deriva_precisao.png", label="Baixar PNG — Deriva")
    else:
        st.info("Sem dados para deriva.")

    st.subheader("Como o modelo decide (explicabilidade simples)")
    # Pipeline simples com regressão logística nas features definidas na Sprint 2
    if len(df_filt):
        df_ml = df_filt[["Suspeita","Tickets","Devoluções","TempoUsoDias","RegistradoHP","TipoErro"]].copy()
        df_ml["RegistradoHP"] = df_ml["RegistradoHP"].astype(int)
        X = df_ml[["Tickets","Devoluções","TempoUsoDias","RegistradoHP","TipoErro"]]
        y = df_ml["Suspeita"]

        cat_cols = ["TipoErro"]
        num_cols = ["Tickets","Devoluções","TempoUsoDias","RegistradoHP"]
        preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
        model = Pipeline([("prep", preproc), ("lr", LogisticRegression(max_iter=200))])
        model.fit(X, y)

        # Importâncias baseadas em |coef|
        lr = model.named_steps["lr"]
        ohe = model.named_steps["prep"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = cat_names + num_cols
        coefs = lr.coef_[0]
        imp_df = pd.DataFrame({"feature": feature_names, "peso": coefs, "importancia_abs": np.abs(coefs)}).sort_values("importancia_abs", ascending=False)

        st.write("Pesos (regressão logística) — quanto maior |peso|, maior a influência.")
        st.dataframe(imp_df[["feature","peso"]])

        fig_imp = px.bar(imp_df.head(10).sort_values("importancia_abs"), x="importancia_abs", y="feature", orientation="h",
                         title="Top 10 — Importância (|peso|)")
        st.plotly_chart(fig_imp, use_container_width=True)
        download_plotly_png(fig_imp, filename="importancia_modelo.png", label="Baixar PNG — Importâncias")

        st.markdown("**Explicação de um caso**")
        idx = st.number_input("Índice do caso (linha do dataset filtrado)", min_value=0, max_value=len(X)-1, value=0, step=1)
        x_row = X.iloc[[idx]]
        # Obter vetor transformado e contribuições lineares
        x_vec = model.named_steps["prep"].transform(x_row)
        if hasattr(x_vec, "toarray"):
            x_vec = x_vec.toarray()
        contribs = x_vec[0] * coefs
        contrib_df = pd.DataFrame({"feature": feature_names, "contribuicao": contribs}).sort_values("contribuicao", ascending=False)
        st.dataframe(contrib_df)

        fig_expl = px.bar(contrib_df, x="contribuicao", y="feature", orientation="h", title="Contribuições para a suspeita (log-odds)")
        st.plotly_chart(fig_expl, use_container_width=True)
        download_plotly_png(fig_expl, filename="explicacao_caso.png", label="Baixar PNG — Caso explicado")
    else:
        st.info("Sem dados suficientes para treinar/exibir explicabilidade.")

    st.markdown("### Model Card")
    st.info(
        "**Objetivo:** Detectar e mitigar falsificações em suprimentos.\n"
        "**Dados usados:** Chamados, devoluções, NPS, logs de suporte e atributos sintetizados (TipoErro, TempoUsoDias, RegistradoHP) conforme protótipo de Sprint 2.\n"
        "**Limites de uso:** Não substitui revisão humana em casos críticos; não utilizar para decisões punitivas sem validação adicional.\n"
        "**Monitoramento:** KPIs de precisão/recall, FPR/FNR, deriva por mês e análise por cortes (Região/Canal)."
    )

    st.markdown("### LGPD mini")
    st.info(
        "**Base legal:** Execução de contrato e legítimo interesse.\n"
        "**Minimização:** Apenas dados essenciais; pseudonimização no treinamento.\n"
        "**Retenção:** 12 meses após o fim do piloto; revisões trimestrais.\n"
        "**Direitos do titular:** Transparência, revisão humana e canal de contestação para decisões automatizadas."
    )

elif page == "Negócio & ROI":
    st.header("Comparativo Com Ação vs. Sem Ação")
    if len(df_filt):
        comp = df_filt.groupby("Com Ação").agg({
            "Tickets": "sum",
            "Devoluções": "sum",
            "NPS": "mean",
            "Custo Suporte": "sum"
        }).rename(index={True: "Com Ação", False: "Sem Ação"})
        st.dataframe(comp)

        st.subheader("Estimativa de ROI/Payback")
        # Hipóteses de economia com base nas devoluções evitadas e custos de suporte
        custo_acao = float(comp.loc["Com Ação", "Custo Suporte"]) if "Com Ação" in comp.index else 0.0
        custo_sem = float(comp.loc["Sem Ação", "Custo Suporte"]) if "Sem Ação" in comp.index else 0.0
        devol_acao = float(comp.loc["Com Ação", "Devoluções"]) if "Com Ação" in comp.index else 0.0
        devol_sem = float(comp.loc["Sem Ação", "Devoluções"]) if "Sem Ação" in comp.index else 0.0
        economia = max(0.0, (devol_sem - devol_acao)) * float(valor_por_devolucao_ev)
        delta_custo = (custo_acao - custo_sem) + float(custo_programa_mensal)
        roi = (economia - delta_custo) / delta_custo if delta_custo > 0 else 0.0
        payback_meses = (delta_custo / economia) if economia > 0 else np.inf

        c1, c2 = st.columns(2)
        c1.metric("ROI Estimado", f"{roi:.1%}")
        c2.metric("Payback (meses)", "∞" if np.isinf(payback_meses) else f"{payback_meses:.1f}")

        st.subheader("Recomendação Go/No-Go")
        if roi > 0.10 and not np.isinf(payback_meses) and payback_meses <= 12:
            st.success("Recomendação: **Go** (seguir com o modelo)")
        else:
            st.warning("Recomendação: **No-Go** (rever estratégia ou parâmetros de custo/benefício)")
    else:
        st.info("Sem dados para o recorte atual.")

elif page == "Mapa Regional":
    st.header("Incidência Regional de Suspeitas")
    if len(df_filt):
        # Coordenadas aproximadas por UF (amostra das UFs usadas)
        uf_coords = {
            "SP": (-23.55, -46.63), "RJ": (-22.91, -43.17), "MG": (-19.92, -43.94),
            "BA": (-12.97, -38.51), "RS": (-30.03, -51.23), "PE": (-8.05, -34.90),
            "PR": (-25.42, -49.27), "SC": (-27.59, -48.55), "GO": (-16.68, -49.25),
            "DF": (-15.79, -47.88)
        }
        mapa_df = df_filt.groupby(["UF", "Região"]).agg({"Suspeita": "sum"}).reset_index()
        mapa_df["lat"] = mapa_df["UF"].map(lambda x: uf_coords.get(x, (-15.0, -47.0))[0])
        mapa_df["lon"] = mapa_df["UF"].map(lambda x: uf_coords.get(x, (-15.0, -47.0))[1])

        fig_map = px.scatter_mapbox(
            mapa_df, lat="lat", lon="lon", size="Suspeita", hover_name="UF", hover_data=["Região","Suspeita"],
            zoom=3.2, height=550
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
        download_plotly_png(fig_map, filename="mapa_suspeitas.png", label="Baixar PNG — Mapa")

        st.subheader("Hotspots de Falsificação")
        st.dataframe(mapa_df.sort_values("Suspeita", ascending=False).head(10))
        download_csv(mapa_df.sort_values("Suspeita", ascending=False), filename="hotspots.csv", label="Baixar CSV — Hotspots")
    else:
        st.info("Sem dados para o recorte atual.")

elif page == "Downloads & Evidências":
    st.header("Exportar Dados e Evidências")
    download_csv(df_filt, filename="dados_filtrados.csv", label="Baixar CSV — Dados Filtrados")

    # Gráfico auxiliar para export (sempre disponível aqui)
    if len(df_filt):
        aux = df_filt.groupby("UF").agg(Suspeitas=("Suspeita","sum")).reset_index()
        fig_aux = px.bar(aux.sort_values("Suspeitas", ascending=False).head(15), x="UF", y="Suspeitas", title="Top UFs por Suspeitas")
        st.plotly_chart(fig_aux, use_container_width=True)
        download_plotly_png(fig_aux, filename="top_ufs_suspeitas.png", label="Baixar PNG — Top UFs")
    else:
        st.info("Sem dados para gráficos de evidência.")

    st.subheader("Resumo Executivo (gerado)")
    resumo_txt = resumo_executivo(df_filt)
    st.text(resumo_txt)

    st.subheader("Anotações/Insights")
    insight = st.text_area(
        "Destaque achados (ex: 'Nordeste 3× mais ocorrências que Sul')",
        value="Nordeste apresenta incidência relativa maior que o Sul no período analisado."
    )
    st.write("Insight salvo:", insight)

    # PDF de 1 página com KPIs + insight
    kpis = kpi_overview(df_filt)
    pdf_bytes = generate_pdf_summary(kpis, insight + "\n\n" + resumo_txt)
    st.download_button("Baixar PDF — Resumo Executivo", pdf_bytes, file_name="resumo_executivo.pdf", mime="application/pdf")

# --- Footer de Governança ---
st.markdown("---")
st.caption("Este painel é um protótipo para decisão executiva (go/no-go). Inclui métricas do piloto, fairness, explicabilidade básica, LGPD mini e análise de ROI.")
