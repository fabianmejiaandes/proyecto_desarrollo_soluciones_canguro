# app.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc


# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "DashboardData.parquet"
THEME = dbc.themes.FLATLY

COLOR_PRIMARY = "#12355B"
COLOR_SECONDARY = "#2E86AB"
COLOR_BG = "#F4F7FB"
COLOR_TEXT = "#243B53"
COLOR_MUTED = "#52616B"
COLOR_CARD = "#FFFFFF"

RISK_COLORS = {
    "Riesgo Bajo": "#2EAD66",
    "Riesgo Moderado": "#F2B84B",
    "Riesgo Alto": "#D64545",
    "Sin clasificación": "#9CA3AF",
}

PHASES = [
    {
        "label": "Nacimiento",
        "weight": "ERN_Peso",
        "height": "ERN_Talla",
        "head": "ERN_PC",
        "z_weight": "zscorepeso0",
        "z_height": "zscoretalla0",
    },
    {
        "label": "40 semanas",
        "weight": "V218",
        "height": "V219",
        "head": "V220",
        "z_weight": "zscorepeso2",
        "z_height": "zscoretalla2",
    },
    {
        "label": "3 meses",
        "weight": "V261",
        "height": "V262",
        "head": "V263",
        "z_weight": "zscorepeso3",
        "z_height": "zscoretalla3",
    },
    {
        "label": "6 meses",
        "weight": "V304",
        "height": "V305",
        "head": "V306",
        "z_weight": "zscorepeso6",
        "z_height": "zscoretalla6",
    },
    {
        "label": "9 meses",
        "weight": "V347",
        "height": "V348",
        "head": "V349",
        "z_weight": "zscorepeso9",
        "z_height": "zscoretalla9",
    },
    {
        "label": "12 meses",
        "weight": "V389",
        "height": "V390",
        "head": "V391",
        "z_weight": "zscorepeso12",
        "z_height": "zscoretalla12",
    },
]

PHASE_ORDER = [p["label"] for p in PHASES]
PHASE_ORDER_MAP = {phase: i for i, phase in enumerate(PHASE_ORDER)}


# =====================================================
# DATA
# =====================================================

#df = pd.read_parquet(DATA_PATH)
df = pd.read_excel("Dashboard_demo.xlsx")


# =====================================================
# HELPERS
# =====================================================

def fmt_pct(x):
    if pd.isna(x):
        return "0,0%"
    return f"{float(x):.1f}%".replace(".", ",")


def fmt_num(x, dec=1):
    if pd.isna(x):
        return "Sin dato"
    return f"{float(x):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def safe_mean(data, col):
    if col not in data.columns:
        return np.nan
    return pd.to_numeric(data[col], errors="coerce").mean()


def card_style():
    return {
        "borderRadius": "18px",
        "boxShadow": "0 10px 24px rgba(0,0,0,0.08)",
        "border": "0",
        "background": COLOR_CARD,
    }


def kpi_card(title, value, subtitle="", icon="📌", color=COLOR_PRIMARY):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div(icon, style={"fontSize": "30px", "marginRight": "12px"}),
                html.Div([
                    html.Div(value, style={
                        "fontSize": "26px",
                        "fontWeight": "900",
                        "lineHeight": "1"
                    }),
                    html.Div(title, style={
                        "fontSize": "13px",
                        "fontWeight": "700"
                    }),
                    html.Div(subtitle, style={
                        "fontSize": "12px",
                        "opacity": "0.85"
                    }),
                ])
            ], style={"display": "flex", "alignItems": "center"})
        ]),
        style={
            "borderRadius": "18px",
            "background": color,
            "color": "white",
            "boxShadow": "0 10px 24px rgba(0,0,0,0.12)",
            "border": "0",
        }
    )


def graph_card(title, graph_id=None, figure=None, children=None):
    content = [
        html.H5(
            title,
            style={
                "fontWeight": "800",
                "color": COLOR_PRIMARY,
                "marginBottom": "12px"
            }
        )
    ]

    if graph_id:
        content.append(
            dcc.Loading(
                type="circle",
                children=dcc.Graph(id=graph_id, config={"displayModeBar": False})
            )
        )
    elif figure is not None:
        content.append(
            dcc.Loading(
                type="circle",
                children=dcc.Graph(figure=figure, config={"displayModeBar": False})
            )
        )
    elif children is not None:
        content.append(children)

    return dbc.Card(dbc.CardBody(content), style=card_style())


def clean_fig(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=45, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=COLOR_TEXT, size=12),
        legend_title_text="",
        title_font=dict(size=15, color=COLOR_PRIMARY),
    )
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(36,59,83,0.12)", automargin=True)
    return fig


def get_phase_summary(data):
    rows = []

    for p in PHASES:
        weight_col = p["weight"]
        height_col = p["height"]
        z_col = p["z_weight"]

        rows.append({
            "Fase": p["label"],
            "Peso promedio": safe_mean(data, weight_col),
            "Talla promedio": safe_mean(data, height_col),
            "Z-score peso promedio": safe_mean(data, z_col),
            "Pacientes con dato": data[weight_col].notna().sum()
            if weight_col in data.columns else 0,
        })

    return pd.DataFrame(rows)


def build_patient_timeline(row):
    rows = []

    for p in PHASES:
        rows.append({
            "Fase": p["label"],
            "Peso": row.get(p["weight"], np.nan),
            "Talla": row.get(p["height"], np.nan),
            "PC": row.get(p["head"], np.nan),
            "Z-score Peso": row.get(p["z_weight"], np.nan),
            "Z-score Talla": row.get(p["z_height"], np.nan),
        })

    return pd.DataFrame(rows)


def clinical_insight_by_phase(data, z_col):
    if z_col not in data.columns:
        return "No hay información suficiente para esta fase."

    z = pd.to_numeric(data[z_col], errors="coerce")
    valid = z.dropna()

    if valid.empty:
        return "No hay datos antropométricos disponibles para esta fase."

    pct_z2 = (valid < -2).mean() * 100
    pct_z15 = (valid < -1.5).mean() * 100
    mean_z = valid.mean()

    if pct_z2 >= 20:
        level = "alta proporción de pacientes con alteración nutricional"
    elif pct_z15 >= 30:
        level = "grupo importante en zona de vigilancia clínica"
    else:
        level = "comportamiento general relativamente estable"

    return (
        f"En esta fase, el Z-score promedio de peso es {fmt_num(mean_z, 2)}. "
        f"El {fmt_pct(pct_z2)} de los pacientes está por debajo de -2 DE y el "
        f"{fmt_pct(pct_z15)} por debajo de -1,5 DE. Esto sugiere {level}."
    )


def ordered_phase_values(data):
    if "fase_actual" not in data.columns:
        return []

    values = data["fase_actual"].dropna().unique().tolist()
    return sorted(values, key=lambda x: PHASE_ORDER_MAP.get(x, 999))


def get_available_values(data, col):
    if col not in data.columns:
        return []
    return sorted(data[col].dropna().unique())


# =====================================================
# APP
# =====================================================

app = Dash(
    __name__,
    external_stylesheets=[THEME],
    suppress_callback_exceptions=True
)

app.title = "KMC - Riesgo de Malnutrición"


# =====================================================
# HEADER
# =====================================================

header = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Img(
                src="/assets/logo.png",
                style={
                    "height": "95px",
                    "maxWidth": "360px",
                    "objectFit": "contain",
                    "backgroundColor": "white",
                    "borderRadius": "12px",
                    "padding": "6px",
                },
                alt="Logo institución"
            )
        ], width="auto"),

        dbc.Col([
            html.H3(
                "Dashboard Clínico de Riesgo de Malnutrición",
                style={"margin": "0", "fontWeight": "900"}
            ),
            html.Div(
                "Seguimiento longitudinal de bebés canguro",
                style={"fontSize": "14px", "opacity": 0.9}
            )
        ])
    ], align="center")
], fluid=True, style={
    "background": COLOR_PRIMARY,
    "color": "white",
    "padding": "14px 22px",
    "borderRadius": "0 0 22px 22px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.18)"
})


# =====================================================
# FILTERS
# =====================================================

filters = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Sede", style={"fontWeight": "800"}),
                dcc.Dropdown(
                    id="filter-sede",
                    options=[
                        {"label": str(s), "value": s}
                        for s in get_available_values(df, "Iden_Sede")
                    ],
                    multi=True,
                    placeholder="Todas las sedes",
                )
            ], md=2),

            dbc.Col([
                html.Label("Categoría de riesgo", style={"fontWeight": "800"}),
                dcc.Dropdown(
                    id="filter-riesgo",
                    options=[
                        {"label": "Riesgo Bajo", "value": "Riesgo Bajo"},
                        {"label": "Riesgo Moderado", "value": "Riesgo Moderado"},
                        {"label": "Riesgo Alto", "value": "Riesgo Alto"},
                        {"label": "Sin clasificación", "value": "Sin clasificación"},
                    ],
                    multi=True,
                    placeholder="Todos los riesgos",
                )
            ], md=2),

            dbc.Col([
                html.Label("Edad gestacional", style={"fontWeight": "800"}),
                dcc.RangeSlider(
                    id="filter-eg",
                    min=20,
                    max=42,
                    step=1,
                    value=[20, 42],
                    marks={i: str(i) for i in range(22, 43, 4)},
                    tooltip={"placement": "bottom", "always_visible": False},
                )
            ], md=3),

            dbc.Col([
                html.Label("Fase actual", style={"fontWeight": "800"}),
                dcc.Dropdown(
                    id="filter-fase",
                    options=[
                        {
                            "label": f"{PHASE_ORDER_MAP.get(f, 999)} - {f}",
                            "value": f
                        }
                        for f in ordered_phase_values(df)
                    ],
                    multi=True,
                    placeholder="Todas las fases",
                )
            ], md=3),

            dbc.Col([
                html.Label("Acciones", style={"fontWeight": "800"}),
                dbc.Button(
                    "🧹 Borrar filtros",
                    id="btn-clear-filters",
                    color="secondary",
                    outline=True,
                    style={"width": "100%", "marginTop": "2px"}
                )
            ], md=2),
        ], className="g-3")
    ]),
    style={
        "borderRadius": "18px",
        "boxShadow": "0 8px 20px rgba(0,0,0,0.08)",
        "border": "0",
    }
)


# =====================================================
# LAYOUT
# =====================================================

app.layout = dbc.Container([
    header,

    html.Div(style={"height": "18px"}),

    dbc.Row([
        dbc.Col([
            html.H4(
                "Monitoreo longitudinal del riesgo nutricional",
                style={"fontWeight": "900", "color": COLOR_PRIMARY}
            ),
            html.P(
                "Filtra la cohorte activa y explora el riesgo desde una perspectiva poblacional, por fase clínica e individual.",
                style={"color": COLOR_MUTED, "marginBottom": "0"}
            )
        ], md=8),

        dbc.Col([
            dbc.Badge(
                "Usuarios activos: últimos 28 meses",
                color="info",
                style={"padding": "10px", "fontSize": "13px"}
            )
        ], md=4, style={"textAlign": "right", "paddingTop": "8px"})
    ]),

    html.Div(style={"height": "12px"}),

    filters,

    html.Div(style={"height": "14px"}),

    dcc.Store(id="filtered-data"),

    dcc.Tabs(
        id="tabs",
        value="tab-pop",
        children=[
            dcc.Tab(label="📊 Vista poblacional", value="tab-pop"),
            dcc.Tab(label="🧠 Vista por fase", value="tab-phase"),
            dcc.Tab(label="👶 Vista paciente", value="tab-patient"),
        ],
        style={"fontWeight": "700"}
    ),

    html.Div(style={"height": "14px"}),

    dcc.Loading(
        type="circle",
        children=html.Div(id="tab-content")
    )

], fluid=True, style={
    "backgroundColor": COLOR_BG,
    "minHeight": "100vh",
    "paddingBottom": "34px",
})


# =====================================================
# CALLBACK BORRAR FILTROS
# =====================================================

@app.callback(
    Output("filter-sede", "value"),
    Output("filter-riesgo", "value"),
    Output("filter-eg", "value"),
    Output("filter-fase", "value"),
    Input("btn-clear-filters", "n_clicks"),
    prevent_initial_call=True
)
def clear_filters(n_clicks):
    return None, None, [20, 42], None


# =====================================================
# GLOBAL FILTER CALLBACK
# =====================================================

@app.callback(
    Output("filtered-data", "data"),
    Input("filter-sede", "value"),
    Input("filter-riesgo", "value"),
    Input("filter-eg", "value"),
    Input("filter-fase", "value"),
)
def filter_data(sede, riesgo, eg_range, fase):
    dff = df.copy()

    if sede and "Iden_Sede" in dff.columns:
        dff = dff[dff["Iden_Sede"].isin(sede)]

    if riesgo and "cat_riesgo" in dff.columns:
        dff = dff[dff["cat_riesgo"].isin(riesgo)]

    if eg_range and "edadgestaFUM" in dff.columns:
        eg = pd.to_numeric(dff["edadgestaFUM"], errors="coerce")
        dff = dff[(eg >= eg_range[0]) & (eg <= eg_range[1])]

    if fase and "fase_actual" in dff.columns:
        dff = dff[dff["fase_actual"].isin(fase)]

    return dff.to_dict("records")


# =====================================================
# RENDER TABS
# =====================================================

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("filtered-data", "data"),
)
def render_tab(tab, data):
    data = pd.DataFrame(data)

    if data.empty:
        return dbc.Alert(
            "No hay datos disponibles con los filtros seleccionados.",
            color="warning",
            style={"borderRadius": "14px"}
        )

    # =================================================
    # VISTA POBLACIONAL
    # =================================================
    if tab == "tab-pop":
        total = len(data)
        riesgo_alto = (
            (data["cat_riesgo"] == "Riesgo Alto").mean() * 100
            if "cat_riesgo" in data else 0
        )
        riesgo_prom = safe_mean(data, "predict_riesgo_pct")
        eg_prom = safe_mean(data, "edadgestaFUM")
        peso_prom = safe_mean(data, "ERN_Peso")

        risk_dist = data["cat_riesgo"].fillna("Sin clasificación").value_counts().reset_index()
        risk_dist.columns = ["Riesgo", "Pacientes"]

        fig_risk = px.bar(
            risk_dist,
            x="Riesgo",
            y="Pacientes",
            color="Riesgo",
            text="Pacientes",
            color_discrete_map=RISK_COLORS,
            title="Distribución de pacientes por categoría de riesgo",
        )
        fig_risk.update_traces(
            hovertemplate="<b>Riesgo:</b> %{x}<br><b>Pacientes:</b> %{y}<extra></extra>",
            textposition="outside"
        )
        fig_risk = clean_fig(fig_risk)

        phase_summary = get_phase_summary(data)

        fig_z = px.line(
            phase_summary,
            x="Fase",
            y="Z-score peso promedio",
            markers=True,
            title="Evolución promedio del Z-score de peso por fase",
        )
        fig_z.add_hline(
            y=-2,
            line_dash="dash",
            line_color="#D64545",
            annotation_text="Umbral -2 DE"
        )
        fig_z.update_traces(
            hovertemplate="<b>Fase:</b> %{x}<br><b>Z-score promedio:</b> %{y:.2f}<extra></extra>"
        )
        fig_z = clean_fig(fig_z)

        fig_weight = px.line(
            phase_summary,
            x="Fase",
            y="Peso promedio",
            markers=True,
            title="Evolución del peso promedio por fase",
        )
        fig_weight.update_traces(
            hovertemplate="<b>Fase:</b> %{x}<br><b>Peso promedio:</b> %{y:.1f} g<extra></extra>"
        )
        fig_weight = clean_fig(fig_weight)

        table_df = phase_summary.copy()
        table_df["Peso promedio"] = table_df["Peso promedio"].round(1)
        table_df["Talla promedio"] = table_df["Talla promedio"].round(1)
        table_df["Z-score peso promedio"] = table_df["Z-score peso promedio"].round(2)

        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    kpi_card(
                        "Pacientes activos",
                        f"{total:,}".replace(",", "."),
                        "Cohorte filtrada",
                        "👶",
                        COLOR_PRIMARY
                    ),
                    md=3
                ),
                dbc.Col(
                    kpi_card(
                        "Riesgo alto",
                        fmt_pct(riesgo_alto),
                        "Según modelo",
                        "🚨",
                        "#B42318"
                    ),
                    md=3
                ),
                dbc.Col(
                    kpi_card(
                        "Riesgo promedio",
                        fmt_pct(riesgo_prom),
                        "Probabilidad media",
                        "📈",
                        COLOR_SECONDARY
                    ),
                    md=3
                ),
                dbc.Col(
                    kpi_card(
                        "Peso nacer prom.",
                        f"{fmt_num(peso_prom, 0)} g",
                        f"EG prom: {fmt_num(eg_prom, 1)} sem",
                        "⚖️",
                        "#4C7A6B"
                    ),
                    md=3
                ),
            ], className="g-3"),

            html.Div(style={"height": "14px"}),

            dbc.Row([
                dbc.Col(graph_card("Distribución del riesgo predictivo", figure=fig_risk), md=5),
                dbc.Col(graph_card("Trayectoria poblacional del Z-score", figure=fig_z), md=7),
            ], className="g-3"),

            html.Div(style={"height": "14px"}),

            dbc.Row([
                dbc.Col(graph_card("Evolución del peso promedio", figure=fig_weight), md=7),
                dbc.Col(graph_card(
                    "Resumen por fase",
                    children=dash_table.DataTable(
                        columns=[{"name": c, "id": c} for c in table_df.columns],
                        data=table_df.to_dict("records"),
                        style_cell={
                            "fontSize": "13px",
                            "padding": "9px",
                            "textAlign": "center",
                            "fontFamily": "Arial",
                        },
                        style_header={
                            "backgroundColor": "#EEF3FB",
                            "fontWeight": "900",
                            "color": COLOR_PRIMARY,
                        },
                        style_table={"overflowX": "auto"},
                    )
                ), md=5),
            ], className="g-3"),
        ], fluid=True)

    # =================================================
    # VISTA POR FASE
    # =================================================
    if tab == "tab-phase":
        phase_cards = []

        for p in PHASES:
            label = p["label"]
            count = (
                (data["fase_actual"] == label).sum()
                if "fase_actual" in data.columns else 0
            )

            phase_cards.append(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Div(
                                label,
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "800",
                                    "color": COLOR_MUTED
                                }
                            ),
                            html.Div(
                                f"{count:,}".replace(",", "."),
                                style={
                                    "fontSize": "25px",
                                    "fontWeight": "900",
                                    "color": COLOR_PRIMARY
                                }
                            ),
                            html.Div(
                                "niños",
                                style={
                                    "fontSize": "12px",
                                    "color": COLOR_MUTED
                                }
                            )
                        ]),
                        style=card_style()
                    ),
                    md=2
                )
            )

        return dbc.Container([
            dbc.Row(phase_cards, className="g-3"),

            html.Div(style={"height": "14px"}),

            dbc.Row([
                dbc.Col(graph_card("Distribución de peso por fase", graph_id="phase-weight"), md=6),
                dbc.Col(graph_card("Distribución de Z-score peso", graph_id="phase-zscore"), md=6),
            ], className="g-3"),

            html.Div(style={"height": "14px"}),

            dbc.Row([
                dbc.Col(graph_card("Riesgo predictivo por categoría", graph_id="phase-risk"), md=6),
                dbc.Col(graph_card("Insight clínico automático", children=html.Div(id="phase-insight")), md=6),
            ], className="g-3"),
        ], fluid=True)

    # =================================================
    # VISTA PACIENTE
    # =================================================
    patient_options = [
        {"label": str(x), "value": str(x)}
        for x in data["Iden_Codigo"].dropna().astype(str).unique()
    ]

    default_patient = patient_options[0]["value"] if patient_options else None

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Selecciona paciente", style={"fontWeight": "800"}),
                dcc.Dropdown(
                    id="patient-dd",
                    options=patient_options,
                    value=default_patient,
                    clearable=False,
                )
            ], md=5),
        ]),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(graph_card("Perfil clínico del paciente", children=html.Div(id="patient-profile")), md=4),
            dbc.Col(graph_card("Evolución antropométrica combinada", graph_id="patient-combined"), md=5),
            dbc.Col(graph_card("Predicción del modelo", children=html.Div(id="patient-prediction")), md=3),
        ], className="g-3"),

        html.Div(style={"height": "14px"}),

        dbc.Row([
            dbc.Col(graph_card("Evolución de Z-scores", graph_id="patient-zscore"), md=7),
            dbc.Col(graph_card("Comparación con cohorte", graph_id="patient-cohort"), md=5),
        ], className="g-3"),
    ], fluid=True)


# =====================================================
# CALLBACK FASE
# =====================================================

@app.callback(
    Output("phase-weight", "figure"),
    Output("phase-zscore", "figure"),
    Output("phase-risk", "figure"),
    Output("phase-insight", "children"),
    Input("filter-fase", "value"),
    Input("filtered-data", "data"),
)
def update_phase_view(phase_label, data):
    data = pd.DataFrame(data)

    if isinstance(phase_label, list) and len(phase_label) > 0:
        phase_label = phase_label[0]
    elif not phase_label:
        phase_label = "6 meses"

    phase = next(p for p in PHASES if p["label"] == phase_label)

    weight_col = phase["weight"]
    z_col = phase["z_weight"]

    fig_weight = px.histogram(
        data,
        x=weight_col,
        nbins=30,
        title=f"Distribución del peso - {phase_label}",
        color_discrete_sequence=[COLOR_SECONDARY],
    )
    fig_weight.update_traces(
        hovertemplate="<b>Peso:</b> %{x} g<br><b>Pacientes:</b> %{y}<extra></extra>"
    )
    fig_weight = clean_fig(fig_weight)

    fig_z = px.histogram(
        data,
        x=z_col,
        nbins=30,
        title=f"Distribución Z-score peso - {phase_label}",
        color_discrete_sequence=["#4C7A6B"],
    )
    fig_z.add_vline(x=-2, line_dash="dash", line_color="#D64545")
    fig_z.update_traces(
        hovertemplate="<b>Z-score:</b> %{x:.2f}<br><b>Pacientes:</b> %{y}<extra></extra>"
    )
    fig_z = clean_fig(fig_z)

    risk_df = (
        data.groupby(["fase_actual", "cat_riesgo"], dropna=False)
        .size()
        .reset_index(name="Pacientes")
    )

    risk_df["cat_riesgo"] = risk_df["cat_riesgo"].fillna("Sin clasificación")

    risk_df["fase_order"] = risk_df["fase_actual"].map(PHASE_ORDER_MAP)
    risk_df = risk_df.sort_values("fase_order")

    fig_risk = px.bar(
        risk_df,
        x="fase_actual",
        y="Pacientes",
        color="cat_riesgo",
        text="Pacientes",
        barmode="group",
        color_discrete_map=RISK_COLORS,
        title="Distribución de riesgo por fase actual",
        labels={
            "fase_actual": "Fase actual",
            "cat_riesgo": "Categoría de riesgo",
            "Pacientes": "Pacientes"
        }
    )
    fig_risk.update_traces(
        textposition="outside",
        hovertemplate="<b>Fase:</b> %{x}<br><b>Pacientes:</b> %{y}<extra></extra>"
    )
    fig_risk = clean_fig(fig_risk)

    insight = clinical_insight_by_phase(data, z_col)

    insight_component = html.Div([
        html.H4(phase_label, style={"fontWeight": "900", "color": COLOR_PRIMARY}),
        html.P(insight, style={"fontSize": "15px", "lineHeight": "1.6"}),
        html.Hr(),
        html.P(
            "Este análisis permite identificar en qué momento del seguimiento aparece una señal nutricional relevante.",
            style={"color": COLOR_MUTED, "fontSize": "13px"}
        )
    ])

    return fig_weight, fig_z, fig_risk, insight_component


# =====================================================
# CALLBACK PACIENTE
# =====================================================

@app.callback(
    Output("patient-profile", "children"),
    Output("patient-combined", "figure"),
    Output("patient-prediction", "children"),
    Output("patient-zscore", "figure"),
    Output("patient-cohort", "figure"),
    Input("patient-dd", "value"),
    Input("filtered-data", "data"),
)
def update_patient_view(patient_id, data):
    data = pd.DataFrame(data)

    row_df = data[data["Iden_Codigo"].astype(str) == str(patient_id)].head(1)

    if row_df.empty:
        empty = go.Figure()
        return "Paciente no encontrado", empty, "", empty, empty

    r = row_df.iloc[0]
    timeline = build_patient_timeline(r)

    risk_label = r.get("cat_riesgo", "Sin clasificación")
    risk_pct = r.get("predict_riesgo_pct", np.nan)

    badge_color = {
        "Riesgo Alto": "danger",
        "Riesgo Moderado": "warning",
        "Riesgo Bajo": "success",
    }.get(risk_label, "secondary")

    fecha_parto = r.get("Iden_FechaParto", np.nan)
    if pd.notna(fecha_parto):
        try:
            fecha_parto = pd.to_datetime(fecha_parto).strftime("%Y-%m-%d")
        except Exception:
            fecha_parto = str(fecha_parto)
    else:
        fecha_parto = "Sin dato"

    profile_rows = [
        ("ID paciente", r.get("Iden_Codigo", "")),
        ("Sede", r.get("Iden_Sede", "")),
        ("Fecha parto", fecha_parto),
        ("Edad materna", f"{fmt_num(r.get('CP_edadmaterna'), 1)} años"),
        ("Edad gestacional", f"{fmt_num(r.get('edadgestaFUM'), 1)} semanas"),
        ("Peso al nacer", f"{fmt_num(r.get('ERN_Peso'), 0)} g"),
        ("Última fase disponible", r.get("fase_actual", "Sin dato")),
    ]

    profile = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(
                    label,
                    style={
                        "fontSize": "12px",
                        "fontWeight": "800",
                        "color": COLOR_MUTED,
                        "marginBottom": "3px"
                    }
                ),
                html.Div(
                    str(value),
                    style={
                        "fontSize": "15px",
                        "fontWeight": "800",
                        "color": COLOR_PRIMARY
                    }
                )
            ], md=6, style={
                "padding": "10px",
                "borderBottom": "1px solid #EEF3FB"
            })
            for label, value in profile_rows
        ], className="g-0")
    ])

    fig_combined = make_subplots(specs=[[{"secondary_y": True}]])

    fig_combined.add_trace(
        go.Scatter(
            x=timeline["Fase"],
            y=timeline["Peso"],
            name="Peso (g)",
            mode="lines+markers",
            hovertemplate="<b>Fase:</b> %{x}<br><b>Peso:</b> %{y} g<extra></extra>",
        ),
        secondary_y=False,
    )

    fig_combined.add_trace(
        go.Scatter(
            x=timeline["Fase"],
            y=timeline["Talla"],
            name="Talla (cm)",
            mode="lines+markers",
            hovertemplate="<b>Fase:</b> %{x}<br><b>Talla:</b> %{y} cm<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_combined.add_trace(
        go.Scatter(
            x=timeline["Fase"],
            y=timeline["PC"],
            name="PC (cm)",
            mode="lines+markers",
            hovertemplate="<b>Fase:</b> %{x}<br><b>PC:</b> %{y} cm<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_combined.update_yaxes(title_text="Peso (g)", secondary_y=False)
    fig_combined.update_yaxes(title_text="Talla / PC (cm)", secondary_y=True)
    fig_combined.update_layout(title="Peso, talla y perímetro cefálico")
    fig_combined = clean_fig(fig_combined, height=340)

    prediction = html.Div([
        html.Div(
            "Riesgo estimado de malnutrición",
            style={"fontWeight": "800", "marginBottom": "8px"}
        ),
        dbc.Badge(
            risk_label,
            color=badge_color,
            style={"fontSize": "15px", "padding": "10px 14px"}
        ),
        html.Div(style={"height": "12px"}),
        html.Div(
            fmt_pct(risk_pct),
            style={
                "fontSize": "34px",
                "fontWeight": "900",
                "color": RISK_COLORS.get(risk_label, "#333"),
            }
        ),
        html.Hr(),
        html.P(
            "La predicción corresponde al último punto disponible de seguimiento clínico.",
            style={"fontSize": "12px", "color": COLOR_MUTED}
        )
    ])

    z_long = timeline.melt(
        id_vars="Fase",
        value_vars=["Z-score Peso", "Z-score Talla"],
        var_name="Indicador",
        value_name="Z-score"
    ).dropna()

    fig_z = px.line(
        z_long,
        x="Fase",
        y="Z-score",
        color="Indicador",
        markers=True,
        title="Evolución de Z-scores del paciente",
    )
    fig_z.add_hline(
        y=-2,
        line_dash="dash",
        line_color="#D64545",
        annotation_text="Umbral -2 DE"
    )
    fig_z.update_traces(
        hovertemplate="<b>Fase:</b> %{x}<br><b>%{fullData.name}:</b> %{y:.2f}<extra></extra>"
    )
    fig_z = clean_fig(fig_z, height=330)

    fig_cohort = px.box(
        data,
        y="predict_riesgo_pct",
        points=False,
        title="Riesgo del paciente vs cohorte filtrada",
    )
    fig_cohort.add_scatter(
        x=[0],
        y=[risk_pct],
        mode="markers",
        marker=dict(size=14, color="#D64545"),
        name="Paciente",
        hovertemplate=(
            "<b>Paciente:</b> "
            + str(patient_id)
            + f"<br><b>Riesgo:</b> {fmt_pct(risk_pct)}<extra></extra>"
        ),
    )
    fig_cohort.update_yaxes(title="Riesgo (%)")
    fig_cohort = clean_fig(fig_cohort, height=330)

    return profile, fig_combined, prediction, fig_z, fig_cohort


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)