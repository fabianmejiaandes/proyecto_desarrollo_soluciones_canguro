import pandas as pd
import plotly.express as px

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc


# =========================
# Config
# =========================
EXCEL_PATH = "dashboard_data.xlsx"
THEME = dbc.themes.FLATLY

# Paleta corporativa (sobria)
PALETTE = ["#1F3A5F", "#2F5D8A", "#3B82A0", "#4C7A6B", "#8C6A3B", "#6B7280"]


# =========================
# Cargar datos desde Excel
# =========================
def load_data(excel_path: str) -> dict:
    xl = pd.ExcelFile(excel_path)
    data = {}
    for sh in xl.sheet_names:
        data[sh] = pd.read_excel(excel_path, sheet_name=sh)
    return data


# =========================
# Helpers UI
# =========================
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

def safe_pct(x):
    if pd.isna(x) or x == "":
        return ""
    try:
        return f"{float(x):.1f}%".replace(".", ",")
    except Exception:
        return str(x)

def fmt_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return str(x)

def pro_bar_layout(fig, height=230, bottom=70):
    """
    Ajustes pro: legible, sobrio, sin cortes de etiquetas.
    """
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

        html.Div(style={"height": "12px"}),
        dcc.Store(id="store-data"),

        html.Div(id="main-view")
    ],
    fluid=True,
    style={"backgroundColor": "#f4f7fb", "minHeight": "100vh", "paddingBottom": "34px"},
)


# =========================
# Cargar/recargar Excel
# =========================
@app.callback(
    Output("store-data", "data"),
    Output("last-load-msg", "children"),
    Input("btn-reload", "n_clicks"),
)
def reload_excel(n_clicks):
    try:
        data = load_data(EXCEL_PATH)
        payload = {k: v.to_dict("records") for k, v in data.items()}
        return payload, f"Datos cargados desde: {EXCEL_PATH}"
    except Exception as e:
        return {}, f"Error cargando Excel: {e}"


# =========================
# Render vista única (Poblacional + Demografía)
# =========================
@app.callback(
    Output("main-view", "children"),
    Input("store-data", "data"),
)
def render_main_view(data_store):

    if not data_store:
        return dbc.Alert(
            "No hay datos cargados. Verifica que 'dashboard_data.xlsx' esté en la carpeta y presiona 'Recargar datos (Excel)'.",
            color="warning"
        )

    def df(sheet_name):
        return pd.DataFrame(data_store.get(sheet_name, []))

    card_style = {"borderRadius": "16px", "boxShadow": "0 10px 24px rgba(0,0,0,0.10)"}

    # -----------------
    # KPIs
    # -----------------
    kpi_df = df("1_KPIs_Principales")
    color_map = {
        "Pacientes Totales": ("secondary", "👶"),
        "Niños Sanos": ("success", "✅"),
        "Prematuros Moderado": ("warning", "⚠️"),
        "Prematuros Alto Riesgo": ("danger", "🚨"),
    }

    cards = []
    for _, r in kpi_df.iterrows():
        indicador = str(r.get("Indicador", "")).strip()
        valor = r.get("Valor", "")
        pct = r.get("Porcentaje", "")
        col, icon = color_map.get(indicador, ("secondary", "📌"))
        subtitle = f"({safe_pct(pct)})" if pct != "" and not pd.isna(pct) else None
        cards.append(dbc.Col(kpi_card(indicador, fmt_int(valor), subtitle, col, icon), md=3))

    kpis_row = dbc.Row(cards, className="g-3")

    # -----------------
    # Tabla indicadores
    # -----------------
    ind_df = df("2_Indicadores_Clave")

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

    # -----------------
    # Lactancia (barras)
    # -----------------
    lact_df = df("7_Lactancia_General")
    fig_lact = px.bar(
        lact_df,
        x="Período",
        y="Porcentaje LME (%)",
        text="Porcentaje LME (%)",
        color="Período",
        color_discrete_sequence=PALETTE
    )
    fig_lact.update_layout(showlegend=False)
    fig_lact = pro_bar_layout(fig_lact, height=230, bottom=55)

    # -----------------
    # Complicaciones prenatales (barras)
    # -----------------
    comp_df = df("4_Complicaciones_Prenatales")
    fig_comp = px.bar(
        comp_df,
        x="Complicación",
        y="Porcentaje (%)",
        text="Porcentaje (%)",
        color="Complicación",
        color_discrete_sequence=PALETTE
    )
    fig_comp.update_layout(showlegend=False)
    fig_comp = pro_bar_layout(fig_comp, height=230, bottom=75)

    # -----------------
    # Cards inferiores (Hospitalización + Bajo peso)
    # -----------------
    hosp_stats = df("5_Hospitalizacion_Detalle")
    dias_prom = "N/A"
    try:
        dias_prom_val = hosp_stats.loc[
            hosp_stats["Estadístico"].astype(str).str.contains("Promedio", case=False, na=False),
            "Valor"
        ].iloc[0]
        dias_prom = str(dias_prom_val).replace(".", ",")
    except Exception:
        pass

    bajo_peso_df = df("6_Bajo_Peso_Nacer")
    bajo_peso_count = "N/A"
    try:
        mask = bajo_peso_df["Categoría Peso"].astype(str).str.contains("<1500", na=False)
        bajo_peso_count = int(bajo_peso_df.loc[mask, "N Pacientes"].iloc[0]) if mask.any() else int(bajo_peso_df["N Pacientes"].iloc[0])
    except Exception:
        pass

    # =========================
    # DEMOGRAFÍA
    # =========================
    eg_df = df("11a_Edad_Gestacional")
    sx_df = df("11b_Sexo")
    em_df = df("11c_Edad_Materna")

    fig_eg = px.bar(
        eg_df,
        x="Categoría EG",
        y="Porcentaje (%)",
        text="Porcentaje (%)",
        color="Categoría EG",
        color_discrete_sequence=PALETTE
    )
    fig_eg.update_layout(showlegend=False)
    fig_eg = pro_bar_layout(fig_eg, height=240, bottom=70)

    fig_sx = px.bar(
        sx_df,
        x="Sexo",
        y="Porcentaje (%)",
        text="Porcentaje (%)",
        color="Sexo",
        color_discrete_sequence=PALETTE
    )
    fig_sx.update_layout(showlegend=False)
    fig_sx = pro_bar_layout(fig_sx, height=240, bottom=60)

    fig_em = px.bar(
        em_df,
        x="Rango Edad Materna",
        y="Porcentaje (%)",
        text="Porcentaje (%)",
        color="Rango Edad Materna",
        color_discrete_sequence=PALETTE
    )
    fig_em.update_layout(showlegend=False)
    fig_em = pro_bar_layout(fig_em, height=230, bottom=75)

    # =========================
    # Layout compacto (una sola vista)
    # =========================
    return dbc.Container([
        kpis_row,
        html.Div(style={"height": "12px"}),

        dbc.Row([

            # ---- Columna izquierda (tabla + cards + 2 demografía)
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
                            html.Div(f"{bajo_peso_count} infantes", style={"fontSize": "26px", "fontWeight": 900}),
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

            # ---- Columna derecha (3 gráficas compactas y del mismo alto)
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