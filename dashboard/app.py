import pandas as pd
import numpy as np
import plotly.express as px

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc


# =========================
# Config
# =========================
PATIENTS_EXCEL_PATH = "data/pacientes_dashboard.xlsx"  # relativo a la carpeta dashboard/
THEME = dbc.themes.FLATLY

# Paleta corporativa (sobria)
PALETTE = ["#1F3A5F", "#2F5D8A", "#3B82A0", "#4C7A6B", "#8C6A3B", "#6B7280"]


# =========================
# Helpers datos
# =========================
def load_patients(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="Detalle_Pacientes")
    # Normaliza strings
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": np.nan, "None": np.nan})
    return df


def is_yes(series: pd.Series) -> pd.Series:
    # Mapea "Sí/No" a boolean
    s = series.fillna("").astype(str).str.lower().str.strip()
    return s.isin(["sí", "si", "1", "true", "verdadero", "yes"])


def safe_pct(x):
    try:
        return f"{float(x):.1f}%".replace(".", ",")
    except Exception:
        return ""


def fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)


def section_title(text):
    return html.H5(text, style={"fontWeight": 800, "marginBottom": "10px"})


def kpi_card(title, value, subtitle=None, color="secondary", icon=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(icon, style={"fontSize": "26px", "marginRight": "12px"}),
                html.Div([
                    html.Div(value, style={"fontSize": "28px", "fontWeight": 900, "lineHeight": "1.1"}),
                    html.Div(title, style={"fontSize": "13px", "opacity": 0.9}),
                    html.Div(subtitle, style={"fontSize": "12px", "opacity": 0.85}) if subtitle else None
                ])
            ], style={"display": "flex", "alignItems": "center"})
        ]),
        color=color,
        inverse=True,
        style={
            "borderRadius": "16px",
            "boxShadow": "0 10px 24px rgba(0,0,0,0.10)",
            "border": "1px solid rgba(255,255,255,0.12)"
        }
    )


def pro_bar_layout(fig, height=230, bottom=70):
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        texttemplate="%{text}",
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=bottom),
        xaxis_title="",
        yaxis_title="",
        bargap=0.35,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12, color="#243B53"),
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="rgba(36,59,83,0.10)",
        zerolinecolor="rgba(36,59,83,0.18)"
    )
    fig.update_xaxes(
        automargin=True,
        tickfont=dict(size=11, color="#243B53")
    )
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode="hide"
    )
    return fig


def risk_level_from_pct(p):
    # Reglas simples para el formato condicional de la tabla
    if p >= 25:
        return "Alto"
    if p >= 10:
        return "Moderado"
    return "Bajo"


def prematurity_risk_from_eg(eg_weeks):
    # Regla simple por semanas (solo demo / mockup)
    if pd.isna(eg_weeks):
        return ("Sin dato", None)
    eg = float(eg_weeks)
    if eg < 32:
        label = "Alto riesgo"
    elif eg < 37:
        label = "Moderado riesgo"
    else:
        label = "Bajo riesgo"
    prob = np.clip((37 - eg) / 10, 0, 1)  # heurístico 0–1
    return (label, prob)


# =========================
# App
# =========================
app = Dash(__name__, external_stylesheets=[THEME])
app.title = "Predicción de Prematurez"

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Predicción de Prematurez", style={"fontWeight": 900}),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Inicio", href="#")),
            dbc.NavItem(dbc.NavLink("Análisis", active=True, href="#")),
            dbc.NavItem(dbc.NavLink("Historial", href="#")),
            dbc.NavItem(dbc.NavLink("Configuración", href="#")),
        ], className="ms-auto", navbar=True),
    ]),
    color="primary",
    dark=True,
    style={"borderRadius": "0 0 16px 16px"}
)

app.layout = dbc.Container(
    [
        navbar,
        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(
                dbc.Button("Recargar datos (Excel)", id="btn-reload", color="secondary", outline=True),
                md="auto"
            ),
            dbc.Col(
                html.Div(id="last-load-msg", style={"paddingTop": "8px", "opacity": 0.85}),
                md=True
            )
        ], className="g-2"),

        html.Div(style={"height": "10px"}),

        dcc.Store(id="store-patients"),

        dcc.Tabs(
            id="tabs",
            value="tab-pop",
            children=[
                dcc.Tab(label="Vista Poblacional", value="tab-pop"),
                dcc.Tab(label="Vista por Paciente", value="tab-patient"),
            ]
        ),

        html.Div(style={"height": "12px"}),
        html.Div(id="tab-content")
    ],
    fluid=True,
    style={"backgroundColor": "#f4f7fb", "minHeight": "100vh", "paddingBottom": "34px"},
)


# =========================
# Cargar/recargar Excel
# =========================
@app.callback(
    Output("store-patients", "data"),
    Output("last-load-msg", "children"),
    Input("btn-reload", "n_clicks"),
)
def reload_excel(n_clicks):
    try:
        df = load_patients(PATIENTS_EXCEL_PATH)
        return df.to_dict("records"), f"Datos cargados desde: {PATIENTS_EXCEL_PATH} (Detalle_Pacientes)"
    except Exception as e:
        return [], f"Error cargando Excel: {e}"


# =========================
# Render tabs
# =========================
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("store-patients", "data"),
)
def render_tabs(tab, patients_store):
    if not patients_store:
        return dbc.Alert(
            "No hay datos cargados. Verifica que 'pacientes_dashboard.xlsx' esté en dashboard/data/ y presiona 'Recargar datos (Excel)'.",
            color="warning"
        )

    df = pd.DataFrame(patients_store)
    card_style = {"borderRadius": "16px", "boxShadow": "0 10px 24px rgba(0,0,0,0.10)"}

    if tab == "tab-pop":
        # =====================
        # KPIs desde Detalle_Pacientes
        # =====================
        total = len(df)
        counts_risk = df["nivel_riesgo"].value_counts(dropna=False)
        n_bajo = int(counts_risk.get("Bajo", 0))
        n_mod = int(counts_risk.get("Moderado", 0))
        n_alto = int(counts_risk.get("Alto", 0))

        kpis_row = dbc.Row([
            dbc.Col(kpi_card("Pacientes Totales", fmt_int(total), "(100,0%)", "secondary", "👶"), md=3),
            dbc.Col(kpi_card("Niños Sanos", fmt_int(n_bajo), f"({safe_pct(100*n_bajo/max(total,1))})", "success", "✅"), md=3),
            dbc.Col(kpi_card("Prematuros Riesgo Moderado", fmt_int(n_mod), f"({safe_pct(100*n_mod/max(total,1))})", "warning", "⚠️"), md=3),
            dbc.Col(kpi_card("Prematuros Alto Riesgo", fmt_int(n_alto), f"({safe_pct(100*n_alto/max(total,1))})", "danger", "🚨"), md=3),
        ], className="g-3")

        # =====================
        # Tabla indicadores clave (calculada)
        # =====================
        indicators = []

        # Factores Sí/No
        factor_cols = {
            "Hipertensión materna": "hipertension_materna",
            "Diabetes gestacional": "diabetes_gestacional",
            "Infección urinaria": "infeccion_urinaria",
            "Amenaza de parto prematuro": "amenaza_parto_prematuro",
            "Sangrado": "sangrado",
            "Embarazo múltiple": "embarazo_multiple",
            "Anemia": "anemia",
            "Enfermedad respiratoria": "enfermedad_respiratoria",
        }

        for name, col in factor_cols.items():
            if col in df.columns:
                m = is_yes(df[col])
                total_eval = int(m.notna().sum())
                affected = int(m.sum())
                pct = (affected / max(total_eval, 1)) * 100
                indicators.append({
                    "Indicador": name,
                    "Porcentaje (%)": round(pct, 1),
                    "Total Afectados": affected,
                    "Total Evaluados": total_eval,
                    "Nivel Riesgo": risk_level_from_pct(pct)
                })

        # Edad materna extrema
        if "edad_materna_anios" in df.columns:
            em = pd.to_numeric(df["edad_materna_anios"], errors="coerce")
            m = (em < 20) | (em > 35)
            total_eval = int(em.notna().sum())
            affected = int(m.sum())
            pct = (affected / max(total_eval, 1)) * 100
            indicators.append({
                "Indicador": "Edad materna extrema (<20 o >35 años)",
                "Porcentaje (%)": round(pct, 1),
                "Total Afectados": affected,
                "Total Evaluados": total_eval,
                "Nivel Riesgo": risk_level_from_pct(pct)
            })

        # Bajo peso al nacer <1500
        if "peso_nacer_g" in df.columns:
            pn = pd.to_numeric(df["peso_nacer_g"], errors="coerce")
            m = pn < 1500
            total_eval = int(pn.notna().sum())
            affected = int(m.sum())
            pct = (affected / max(total_eval, 1)) * 100
            indicators.append({
                "Indicador": "Bajo peso al nacer (<1500 g)",
                "Porcentaje (%)": round(pct, 1),
                "Total Afectados": affected,
                "Total Evaluados": total_eval,
                "Nivel Riesgo": risk_level_from_pct(pct)
            })

        # Prematurez <37 semanas
        if "edad_gestacional_semanas" in df.columns:
            eg = pd.to_numeric(df["edad_gestacional_semanas"], errors="coerce")
            m = eg < 37
            total_eval = int(eg.notna().sum())
            affected = int(m.sum())
            pct = (affected / max(total_eval, 1)) * 100
            indicators.append({
                "Indicador": "Edad gestacional < 37 semanas",
                "Porcentaje (%)": round(pct, 1),
                "Total Afectados": affected,
                "Total Evaluados": total_eval,
                "Nivel Riesgo": "Alto" if pct >= 50 else risk_level_from_pct(pct)
            })

        # Días promedio hospital
        dias_prom = "N/A"
        if "dias_hospital" in df.columns:
            dh = pd.to_numeric(df["dias_hospital"], errors="coerce")
            if dh.notna().any():
                dias_prom = str(round(float(dh.mean()), 1)).replace(".", ",")

        # Orden por % desc
        ind_df = pd.DataFrame(indicators).sort_values("Porcentaje (%)", ascending=False)

        indicators_table = dash_table.DataTable(
            columns=[
                {"name": "Indicador", "id": "Indicador"},
                {"name": "%", "id": "Porcentaje (%)"},
                {"name": "Total", "id": "Total Afectados"},
                {"name": "Evaluados", "id": "Total Evaluados"},
                {"name": "Riesgo", "id": "Nivel Riesgo"},
            ],
            data=ind_df.to_dict("records"),
            style_table={"borderRadius": "12px", "overflow": "hidden"},
            style_cell={
                "fontFamily": "Arial",
                "fontSize": "14px",
                "padding": "10px",
                "whiteSpace": "normal",
                "height": "auto"
            },
            style_header={"fontWeight": "900", "backgroundColor": "#eef3fb"},
            style_data_conditional=[
                {"if": {"filter_query": "{Nivel Riesgo} = 'Alto'", "column_id": "Nivel Riesgo"},
                 "backgroundColor": "#f8d7da", "color": "#842029", "fontWeight": "900"},
                {"if": {"filter_query": "{Nivel Riesgo} = 'Moderado'", "column_id": "Nivel Riesgo"},
                 "backgroundColor": "#fff3cd", "color": "#664d03", "fontWeight": "900"},
                {"if": {"filter_query": "{Nivel Riesgo} = 'Bajo'", "column_id": "Nivel Riesgo"},
                 "backgroundColor": "#d1e7dd", "color": "#0f5132", "fontWeight": "900"},
            ],
        )

        # =====================
        # Lactancia (porcentaje Sí)
        # =====================
        lact_points = []
        for label, col in [
            ("40 semanas EG", "lactancia_exclusiva_40sem"),
            ("3 meses EC", "lactancia_exclusiva_3m"),
            ("6 meses EC", "lactancia_exclusiva_6m"),
        ]:
            if col in df.columns:
                m = is_yes(df[col])
                pct = 100 * (m.sum() / max(m.notna().sum(), 1))
                lact_points.append({"Período": label, "Porcentaje (%)": round(pct, 1)})
        lact_df = pd.DataFrame(lact_points)

        fig_lact = px.bar(
            lact_df, x="Período", y="Porcentaje (%)", text="Porcentaje (%)",
            color="Período", color_discrete_sequence=PALETTE
        )
        fig_lact.update_layout(showlegend=False)
        fig_lact = pro_bar_layout(fig_lact, height=230, bottom=55)

        # =====================
        # Complicaciones prenatales (porcentaje Sí)
        # =====================
        comp_vars = [
            ("Infección urinaria", "infeccion_urinaria"),
            ("Sangrado", "sangrado"),
            ("Amenaza parto prematuro", "amenaza_parto_prematuro"),
            ("Edad extrema materna", None),  # calculada
            ("Embarazo múltiple", "embarazo_multiple"),
            ("Hipertensión/preeclampsia", "hipertension_materna"),
            ("Diabetes gestacional", "diabetes_gestacional"),
            ("Anemia", "anemia"),
            ("Enfermedad respiratoria", "enfermedad_respiratoria"),
        ]
        comp_points = []
        for name, col in comp_vars:
            if col is None:
                if "edad_materna_anios" in df.columns:
                    em = pd.to_numeric(df["edad_materna_anios"], errors="coerce")
                    m = (em < 20) | (em > 35)
                    pct = 100 * (m.sum() / max(em.notna().sum(), 1))
                    comp_points.append({"Complicación": name, "Porcentaje (%)": round(pct, 1)})
            else:
                if col in df.columns:
                    m = is_yes(df[col])
                    pct = 100 * (m.sum() / max(m.notna().sum(), 1))
                    comp_points.append({"Complicación": name, "Porcentaje (%)": round(pct, 1)})

        comp_df = pd.DataFrame(comp_points).sort_values("Porcentaje (%)", ascending=False)

        fig_comp = px.bar(
            comp_df, x="Complicación", y="Porcentaje (%)", text="Porcentaje (%)",
            color="Complicación", color_discrete_sequence=PALETTE
        )
        fig_comp.update_layout(showlegend=False)
        fig_comp = pro_bar_layout(fig_comp, height=230, bottom=85)
        fig_comp.update_xaxes(tickangle=35)

        # =====================
        # Demografía
        # =====================
        # Edad gestacional categorías
        eg = pd.to_numeric(df["edad_gestacional_semanas"], errors="coerce")
        eg_cat = pd.cut(
            eg,
            bins=[-np.inf, 28, 32, 34, 37, np.inf],
            labels=[
                "Extremo prematuro (<28 sem)",
                "Muy prematuro (28–31 sem)",
                "Moderado prematuro (32–33 sem)",
                "Tardío prematuro (34–36 sem)",
                "A término (≥37 sem)"
            ]
        )
        eg_dist = (eg_cat.value_counts(normalize=True) * 100).reset_index()
        eg_dist.columns = ["Categoría EG", "Porcentaje (%)"]

        fig_eg = px.bar(
            eg_dist, x="Categoría EG", y="Porcentaje (%)", text="Porcentaje (%)",
            color="Categoría EG", color_discrete_sequence=PALETTE
        )
        fig_eg.update_layout(showlegend=False)
        fig_eg = pro_bar_layout(fig_eg, height=240, bottom=95)
        fig_eg.update_xaxes(tickangle=25)

        # Sexo
        sx_dist = (df["sexo"].value_counts(normalize=True) * 100).reset_index()
        sx_dist.columns = ["Sexo", "Porcentaje (%)"]
        fig_sx = px.bar(
            sx_dist, x="Sexo", y="Porcentaje (%)", text="Porcentaje (%)",
            color="Sexo", color_discrete_sequence=PALETTE
        )
        fig_sx.update_layout(showlegend=False)
        fig_sx = pro_bar_layout(fig_sx, height=240, bottom=55)

        # Edad materna
        em = pd.to_numeric(df["edad_materna_anios"], errors="coerce")
        em_cat = pd.cut(
            em,
            bins=[-np.inf, 20, 26, 31, 36, np.inf],
            labels=["<20 años", "20–25 años", "26–30 años", "31–35 años", ">35 años"]
        )
        em_dist = (em_cat.value_counts(normalize=True) * 100).reset_index()
        em_dist.columns = ["Rango Edad Materna", "Porcentaje (%)"]

        fig_em = px.bar(
            em_dist, x="Rango Edad Materna", y="Porcentaje (%)", text="Porcentaje (%)",
            color="Rango Edad Materna", color_discrete_sequence=PALETTE
        )
        fig_em.update_layout(showlegend=False)
        fig_em = pro_bar_layout(fig_em, height=230, bottom=70)

        # Bajo peso count (para card)
        bajo_peso_count = 0
        if "peso_nacer_g" in df.columns:
            pn = pd.to_numeric(df["peso_nacer_g"], errors="coerce")
            bajo_peso_count = int((pn < 1500).sum())

        # =====================
        # Layout compacto
        # =====================
        return dbc.Container([
            kpis_row,
            html.Div(style={"height": "12px"}),

            dbc.Row([
                # Col izquierda
                dbc.Col([
                    dbc.Card(dbc.CardBody([
                        section_title("Resumen de indicadores clave"),
                        html.Div(indicators_table)
                    ]), style=card_style),

                    html.Div(style={"height": "12px"}),

                    dbc.Row([
                        dbc.Col(
                            dbc.Card(dbc.CardBody([
                                html.Div("Hospitalización promedio", style={"fontWeight": 800}),
                                html.Div(f"{dias_prom} días", style={"fontSize": "26px", "fontWeight": 900}),
                                html.Div("Promedio global de hospitalización", style={"opacity": 0.8, "fontSize": "12px"})
                            ]), style=card_style),
                            md=6
                        ),
                        dbc.Col(
                            dbc.Card(dbc.CardBody([
                                html.Div("Bajo peso al nacer (<1500 g)", style={"fontWeight": 800}),
                                html.Div(f"{fmt_int(bajo_peso_count)} infantes", style={"fontSize": "26px", "fontWeight": 900}),
                                html.Div("Conteo global categoría crítica", style={"opacity": 0.8, "fontSize": "12px"})
                            ]), style=card_style),
                            md=6
                        ),
                    ], className="g-3"),

                    html.Div(style={"height": "12px"}),

                    dbc.Row([
                        dbc.Col(dbc.Card(dbc.CardBody([
                            section_title("Demografía: distribución por edad gestacional"),
                            dcc.Graph(figure=fig_eg, config={"displayModeBar": False})
                        ]), style=card_style), md=6),

                        dbc.Col(dbc.Card(dbc.CardBody([
                            section_title("Demografía: distribución por sexo"),
                            dcc.Graph(figure=fig_sx, config={"displayModeBar": False})
                        ]), style=card_style), md=6),
                    ], className="g-3"),
                ], md=7),

                # Col derecha (3 gráficas compactas)
                dbc.Col([
                    dbc.Card(dbc.CardBody([
                        section_title("Lactancia materna exclusiva: porcentaje por momento de seguimiento"),
                        dcc.Graph(figure=fig_lact, config={"displayModeBar": False})
                    ]), style=card_style),

                    html.Div(style={"height": "12px"}),

                    dbc.Card(dbc.CardBody([
                        section_title("Complicaciones prenatales: porcentaje de casos por tipo de complicación"),
                        dcc.Graph(figure=fig_comp, config={"displayModeBar": False})
                    ]), style=card_style),

                    html.Div(style={"height": "12px"}),

                    dbc.Card(dbc.CardBody([
                        section_title("Distribución de edad materna: porcentaje por rango de edad"),
                        dcc.Graph(figure=fig_em, config={"displayModeBar": False})
                    ]), style=card_style),
                ], md=5),
            ], className="g-3"),
        ], fluid=True)

    # =========================================================
    # TAB PACIENTE
    # =========================================================
    # dropdown
    options = [{"label": str(pid), "value": str(pid)} for pid in df["id_paciente"].astype(str).head(2000)]
    # (si quieres todos: quita head(2000), pero puede ser pesado en UI)
    default_id = options[0]["value"] if options else None

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div("Selecciona un paciente", style={"fontWeight": 700, "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="patient-dd",
                    options=options,
                    value=default_id,
                    clearable=False
                ),
            ], md=6),
        ], className="g-3"),

        html.Div(style={"height": "12px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                section_title("Perfil del paciente"),
                html.Div(id="patient-profile")
            ]), style=card_style), md=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                section_title("Estado nutricional a 12 meses"),
                dcc.Graph(id="nutri-bar", config={"displayModeBar": False})
            ]), style=card_style), md=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                section_title("Uso de servicios: días de hospitalización / UCI / ventilación"),
                dcc.Graph(id="hosp-bars", config={"displayModeBar": False})
            ]), style=card_style), md=4),
        ], className="g-3"),

        html.Div(style={"height": "12px"}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                section_title("Factores de riesgo (registrados)"),
                html.Div(id="risk-factors")
            ]), style=card_style), md=4),

            dbc.Col(dbc.Card(dbc.CardBody([
                section_title("Evolución de Z-scores (peso y talla)"),
                dcc.Graph(id="zscores-line", config={"displayModeBar": False})
            ]), style=card_style), md=5),

            dbc.Col(dbc.Card(dbc.CardBody([
                section_title("Resultado estimado (demo)"),
                html.Div(id="prediction-panel")
            ]), style=card_style), md=3),
        ], className="g-3"),
    ], fluid=True)


# =========================
# Callback paciente
# =========================
@app.callback(
    Output("patient-profile", "children"),
    Output("risk-factors", "children"),
    Output("prediction-panel", "children"),
    Output("hosp-bars", "figure"),
    Output("zscores-line", "figure"),
    Output("nutri-bar", "figure"),
    Input("patient-dd", "value"),
    Input("store-patients", "data"),
)
def update_patient_view(patient_id, patients_store):
    df = pd.DataFrame(patients_store)
    row = df[df["id_paciente"].astype(str) == str(patient_id)].head(1)

    if row.empty:
        empty_fig = px.bar(pd.DataFrame({"x": [], "y": []}), x="x", y="y")
        return "Paciente no encontrado", "", "", empty_fig, empty_fig, empty_fig

    r = row.iloc[0]

    # Perfil
    profile = html.Ul([
        html.Li(f"ID: {r.get('id_paciente', '')}"),
        html.Li(f"Sede: {r.get('sede', '')}"),
        html.Li(f"Sexo: {r.get('sexo', '')}"),
        html.Li(f"Edad materna: {r.get('edad_materna_anios', '')} años"),
        html.Li(f"Edad gestacional: {r.get('edad_gestacional_semanas', '')} semanas"),
        html.Li(f"Peso nacimiento: {r.get('peso_nacer_g', '')} g"),
        html.Li(f"APGAR 5 min: {r.get('apgar_5min', '')}"),
        html.Li(f"Lactancia exclusiva 40 sem: {r.get('lactancia_exclusiva_40sem', '')}"),
    ], style={"marginBottom": 0})

    # Factores de riesgo
    risk_items = []
    risk_map = [
        ("Hipertensión materna", "hipertension_materna"),
        ("Diabetes gestacional", "diabetes_gestacional"),
        ("Infección urinaria", "infeccion_urinaria"),
        ("Amenaza parto prematuro", "amenaza_parto_prematuro"),
        ("Sangrado", "sangrado"),
        ("Embarazo múltiple", "embarazo_multiple"),
        ("Anemia", "anemia"),
        ("Enfermedad respiratoria", "enfermedad_respiratoria"),
    ]
    for name, col in risk_map:
        val = str(r.get(col, ""))
        if val.lower() in ["sí", "si", "1", "true", "verdadero", "yes"]:
            risk_items.append(html.Li(f"{name}: Sí"))
    if not risk_items:
        risk_items = [html.Li("Sin factores marcados como 'Sí' en este registro.")]

    risk_list = html.Ul(risk_items, style={"marginBottom": 0})

    # Panel predicción (demo con edad gestacional)
    label, prob = prematurity_risk_from_eg(r.get("edad_gestacional_semanas", np.nan))
    prob_txt = "N/A" if prob is None else f"{int(round(prob*100))}%"
    badge_color = "danger" if "Alto" in label else ("warning" if "Moderado" in label else "success")

    pred_panel = html.Div([
        html.Div("Probabilidad de prematurez (demo):", style={"fontWeight": 700, "marginBottom": "8px"}),
        dbc.Badge(f"{label} • {prob_txt}", color=badge_color, style={"fontSize": "14px", "padding": "10px 12px"}),
        html.Hr(style={"margin": "12px 0"}),
        html.Div("Prob. crecimiento armónico (si existe):", style={"fontWeight": 700, "marginBottom": "6px"}),
        html.Div(
            (f"{int(round(float(r.get('probabilidad_crecimiento_armonico', 0))*100))}%"
             if pd.notna(r.get("probabilidad_crecimiento_armonico", np.nan)) else "N/A"),
            style={"fontSize": "18px", "fontWeight": 900}
        ),
    ])

    # Hospitalización bars
    hosp_df = pd.DataFrame({
        "Métrica": ["Días hospital", "Días UCI", "Días ventilación"],
        "Valor": [
            pd.to_numeric(r.get("dias_hospital", 0), errors="coerce"),
            pd.to_numeric(r.get("dias_uci", 0), errors="coerce"),
            pd.to_numeric(r.get("dias_ventilacion_mecanica", 0), errors="coerce"),
        ]
    }).fillna(0)

    fig_hosp = px.bar(
        hosp_df, x="Métrica", y="Valor", text="Valor",
        color="Métrica", color_discrete_sequence=PALETTE
    )
    fig_hosp.update_layout(showlegend=False)
    fig_hosp = pro_bar_layout(fig_hosp, height=260, bottom=60)

    # Z-scores line (peso/talla)
    time_points = [
        ("Nacimiento", "zscore_peso_nacer", "zscore_talla_nacer"),
        ("40 sem", "zscore_peso_40sem", "zscore_talla_40sem"),
        ("3 m", "zscore_peso_3m", "zscore_talla_3m"),
        ("6 m", "zscore_peso_6m", "zscore_talla_6m"),
        ("9 m", "zscore_peso_9m", "zscore_talla_9m"),
        ("12 m", "zscore_peso_12m", "zscore_talla_12m"),
    ]
    z_rows = []
    for t, cp, ct in time_points:
        z_rows.append({"Momento": t, "Variable": "Z-Peso", "Z": pd.to_numeric(r.get(cp, np.nan), errors="coerce")})
        z_rows.append({"Momento": t, "Variable": "Z-Talla", "Z": pd.to_numeric(r.get(ct, np.nan), errors="coerce")})

    zdf = pd.DataFrame(z_rows).dropna(subset=["Z"])
    fig_z = px.line(zdf, x="Momento", y="Z", color="Variable", markers=True, color_discrete_sequence=PALETTE)
    fig_z.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=45, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12, color="#243B53"),
        legend_title_text=""
    )
    fig_z.update_yaxes(gridcolor="rgba(36,59,83,0.10)")
    fig_z.update_xaxes(automargin=True)

    # Estado nutricional 12m (por zscore peso 12m)
    z12 = pd.to_numeric(r.get("zscore_peso_12m", np.nan), errors="coerce")
    if pd.isna(z12):
        cats = pd.DataFrame({"Categoría": ["Sin dato"], "Conteo": [1]})
    else:
        if z12 < -2:
            cat = "Bajo peso"
        elif z12 > 2:
            cat = "Sobrepeso"
        else:
            cat = "Adecuado"
        cats = pd.DataFrame({"Categoría": ["Bajo peso", "Adecuado", "Sobrepeso"], "Conteo": [0, 0, 0]})
        cats.loc[cats["Categoría"] == cat, "Conteo"] = 1

    fig_nut = px.bar(
        cats, x="Categoría", y="Conteo", text="Conteo",
        color="Categoría", color_discrete_sequence=PALETTE
    )
    fig_nut.update_layout(showlegend=False)
    fig_nut = pro_bar_layout(fig_nut, height=260, bottom=60)

    return profile, risk_list, pred_panel, fig_hosp, fig_z, fig_nut


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(debug=True)