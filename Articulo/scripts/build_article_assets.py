from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "articulo_neonatologia"
CHART_DIR = OUT_DIR / "assets" / "charts"
DATA_DIR = OUT_DIR / "data"

PHASES = [
    {
        "id": "F0_Prenatal_Parto",
        "short": "F0",
        "label": "Prenatal y parto",
        "display": "F0 Prenatal y parto",
        "features": 41,
        "time": "Antecedentes maternos, embarazo y parto",
    },
    {
        "id": "F1_Nacimiento",
        "short": "F1",
        "label": "Nacimiento",
        "display": "F1 Nacimiento",
        "features": 75,
        "time": "Mediciones y condición inicial al nacer",
    },
    {
        "id": "F2_Hospitalizacion",
        "short": "F2",
        "label": "Hospitalización",
        "display": "F2 Hospitalización",
        "features": 107,
        "time": "Evolución intrahospitalaria y egreso",
    },
    {
        "id": "F3_40semanas",
        "short": "F3",
        "label": "40 semanas EC",
        "display": "F3 40 semanas EC",
        "features": 137,
        "time": "Primer punto de comparación corregida",
    },
    {
        "id": "F4_3meses",
        "short": "F4",
        "label": "3 meses EC",
        "display": "F4 3 meses EC",
        "features": 164,
        "time": "Crecimiento temprano ambulatorio",
    },
    {
        "id": "F5_6meses",
        "short": "F5",
        "label": "6 meses EC",
        "display": "F5 6 meses EC",
        "features": 183,
        "time": "Trayectoria intermedia de crecimiento",
    },
    {
        "id": "F6_9meses",
        "short": "F6",
        "label": "9 meses EC",
        "display": "F6 9 meses EC",
        "features": 198,
        "time": "Última lectura antes del desenlace a 12 meses",
    },
]

PHASE_ORDER = [phase["id"] for phase in PHASES]
PHASE_LOOKUP = {phase["id"]: phase for phase in PHASES}

OUTCOMES = {
    "Stunting": {
        "label": "Talla baja para la edad",
        "short": "Talla baja",
        "clinical": "Señal de desnutrición crónica o restricción sostenida del crecimiento lineal.",
        "color": "#D85B47",
    },
    "Bajo_peso": {
        "label": "Bajo peso para la edad",
        "short": "Bajo peso",
        "clinical": "Alerta de peso bajo respecto a la edad; puede mezclar problemas agudos y crónicos.",
        "color": "#3E6FB5",
    },
    "Wasting": {
        "label": "Desnutrición aguda",
        "short": "Aguda",
        "clinical": "Bajo peso para la talla; se interpreta como desbalance nutricional más reciente.",
        "color": "#3F8A5C",
    },
    "Mixta": {
        "label": "Desnutrición aguda sobre crónica",
        "short": "Mixta",
        "clinical": "Peso para la talla y talla para la edad por debajo de -2 DE al mismo tiempo.",
        "color": "#A1276F",
    },
}

STAGE_FEATURE_CHARTS = {
    "Stunting": "features_por_etapa_barras_stunting.svg",
    "Bajo_peso": "features_por_etapa_barras_bajo_peso.svg",
    "Wasting": "features_por_etapa_barras_wasting.svg",
    "Mixta": "features_por_etapa_barras_mixta.svg",
}

PHASE_COLORS = {
    "F0_Prenatal_Parto": "#D9A441",
    "F1_Nacimiento": "#D85B47",
    "F2_Hospitalizacion": "#3E6FB5",
    "F3_40semanas": "#3F8A5C",
    "F4_3meses": "#8B6FB5",
    "F5_6meses": "#A1276F",
    "F6_9meses": "#2F4858",
}

FEATURE_LABELS = {
    "zscoretalla9": "Puntaje Z de talla a los 9 meses de edad corregida",
    "zscoretalla6": "Puntaje Z de talla a los 6 meses de edad corregida",
    "RCIUpesoytallanacer": "Restricción de crecimiento intrauterino por peso y talla al nacer",
    "velocidad9_6mesesOMS": "Velocidad de crecimiento de 6 a 9 meses según OMS",
    "zscorepeso9": "Puntaje Z de peso a los 9 meses de edad corregida",
    "zscoretalla9cat": "Categoría de talla a los 9 meses de edad corregida",
    "simetrico": "Restricción de crecimiento intrauterino simétrica",
    "zscoretalla2": "Puntaje Z de talla a los 3 meses de edad corregida",
    "zscorepeso6": "Puntaje Z de peso a los 6 meses de edad corregida",
    "zscoretalla0": "Puntaje Z de talla al nacer",
    "velocidad6_3mesesOMS": "Velocidad de crecimiento de 3 a 6 meses según OMS",
    "zscorepeso0": "Puntaje Z de peso al nacer",
    "zscorepesotalla9": "Puntaje Z de peso para la talla a los 9 meses",
    "zscoretalla6cat": "Categoría de talla a los 6 meses",
    "gananciapesonacerpesoentradaPMC": "Ganancia de peso desde nacimiento hasta entrada al programa",
    "gananciatallanacertallaentradaPMC": "Ganancia de talla desde nacimiento hasta entrada al programa",
    "CP_TallaPadre": "Talla paterna",
    "CSP_IngresoMensual": "Ingreso mensual del hogar",
    "CP_TallaMadre": "Talla materna",
    "velocidadzscorepeso40_3m": "Cambio del puntaje Z de peso entre 40 semanas y 3 meses",
    "velocidadzscore3m_40semOMS": "Cambio del puntaje Z de talla entre 40 semanas y 3 meses",
    "zscorepesotalla6": "Puntaje Z de peso para la talla a los 6 meses",
    "gananciapesoentradapeso40sem": "Ganancia de peso desde entrada al programa hasta 40 semanas",
    "ERN_Peso": "Peso al nacer",
    "zscorepesotalla1": "Puntaje Z de peso para la talla a las 40 semanas",
    "zscorepeso1": "Puntaje Z de peso a las 40 semanas",
    "zscoretalla1": "Puntaje Z de talla a las 40 semanas",
    "zscorePC1": "Puntaje Z de perímetro cefálico a las 40 semanas",
    "zscoretalla1cat": "Categoría de talla a las 40 semanas",
    "zscorepeso1cat": "Categoría de peso a las 40 semanas",
    "zscorePC1cat": "Categoría de perímetro cefálico a las 40 semanas",
    "zscorepesotalla2": "Puntaje Z de peso para la talla a los 3 meses",
    "zscorepeso2": "Puntaje Z de peso a los 3 meses de edad corregida",
    "zscorePC2": "Puntaje Z de perímetro cefálico a los 3 meses",
    "zscoretallaOMS2": "Puntaje Z de talla OMS a los 3 meses",
    "zscorepesotalla6cat": "Categoría de peso para la talla a los 6 meses",
    "zscorePC6": "Puntaje Z de perímetro cefálico a los 6 meses",
    "zscorepesotalla9cat": "Categoría de peso para la talla a los 9 meses",
    "zscorePC9": "Puntaje Z de perímetro cefálico a los 9 meses",
    "HD_ValorMasAltoBilirubina": "Valor más alto de bilirrubina",
    "HD_UltiValorHematocrito": "Último valor de hematocrito",
    "HD_NumTrasSanguineas": "Número de transfusiones sanguíneas",
    "PA_DiasHospiMadre": "Días de hospitalización materna",
    "CP_rhMadre": "Rh materno",
    "CP_MadreDrogas": "Consumo de drogas de la madre",
    "edadgestasalPC": "Edad gestacional al salir del hospital",
    "edadsalidaPC": "Edad al salir del hospital",
    "PesosalidaPC": "Peso al salir del hospital, ajustado por percentil",
    "Gananciatallaentradatalla40sem": "Ganancia de talla desde entrada al programa hasta 40 semanas",
}

RC_PARAMS = {
    "font.family": "DejaVu Serif",
    "axes.facecolor": "#FCFCF8",
    "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "axes.edgecolor": "#D9D9D6",
    "axes.labelcolor": "#1A1A1A",
    "text.color": "#1A1A1A",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "axes.titleweight": "semibold",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.frameon": False,
}


def load_dictionary_labels() -> dict[str, str]:
    path = ROOT / "data" / "KMC-70K-diccionarioVARS-Malnutricion-PhETI-rev20250520-MAIA.xlsx"
    if not path.exists():
        return {}
    df = pd.read_excel(path, sheet_name="VARS-(KMC70k)")
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        key = row.get("NOMBRE EN LA BdeD")
        if not isinstance(key, str) or not key.strip():
            continue
        label = row.get("VAR-SHORT DESCRIPTION")
        if not isinstance(label, str) or not label.strip():
            label = row.get("VAR-LONG DESCRIPTION")
        if isinstance(label, str) and label.strip():
            mapping[key.strip()] = label.strip()
    return mapping


def display_feature(feature: str, dictionary: dict[str, str]) -> str:
    return FEATURE_LABELS.get(feature, dictionary.get(feature, feature.replace("_", " ")))


def wrap_label(text: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, format="svg", bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, dict, dict]:
    metrics = pd.read_csv(ROOT / "notebooks" / "metricas" / "metricas_por_fase.csv")
    metadata = json.loads((ROOT / "notebooks" / "modelos_pmci" / "metadata.json").read_text(encoding="utf-8"))
    readme = json.loads((ROOT / "notebooks" / "metricas" / "README.json").read_text(encoding="utf-8"))
    shap = pd.read_csv(ROOT / "notebooks" / "metricas" / "shap_importancia_global.csv")
    feature_lists = {}
    for phase in PHASE_ORDER:
        raw = json.loads(
            (ROOT / "notebooks" / "modelos_pmci" / f"features_{phase}.json").read_text(encoding="utf-8")
        )
        feature_lists[phase] = raw["features"]
    dictionary = load_dictionary_labels()
    return metrics, metadata, readme, shap, feature_lists, dictionary


def chart_auc_evolution(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(PHASE_ORDER))
    labels = [PHASE_LOOKUP[phase]["short"] for phase in PHASE_ORDER]

    for ax, outcome in zip(axes, OUTCOMES):
        sub = metrics.loc[metrics["outcome"] == outcome].set_index("fase").loc[PHASE_ORDER].reset_index()
        y = sub["AUC_cv"].to_numpy()
        color = OUTCOMES[outcome]["color"]
        ax.plot(x, y, color=color, marker="o", lw=2.6, ms=5.5)
        ax.fill_between(x, 0.5, y, color=color, alpha=0.08)
        for idx, value in enumerate(y):
            ax.text(idx, value + 0.013, f"{value:.3f}", ha="center", fontsize=8.2, color=color)
        ax.axhline(0.5, color="#B8B8B0", lw=1.0, ls="--")
        ax.set_xticks(x, labels)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", color="#ECECEA", lw=0.8)
        ax.set_title(OUTCOMES[outcome]["label"])
        ax.text(
            0.02,
            0.92,
            f"+{(y[-1] - y[0]):.3f} de F0 a F6",
            transform=ax.transAxes,
            color="#6E7B91",
            fontsize=9,
        )

    axes[0].set_ylabel("AUC en validación cruzada")
    axes[2].set_ylabel("AUC en validación cruzada")
    fig.suptitle("Evolución del AUC en las siete ventanas clínicas", y=1.02, fontsize=16)
    save_figure(fig, "auc_evolucion.svg")


def chart_metric_balance(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(PHASE_ORDER))
    labels = [PHASE_LOOKUP[phase]["short"] for phase in PHASE_ORDER]
    series = [
        ("Sensibilidad", "Sens_cv", "#3E6FB5"),
        ("Especificidad", "Spec_cv", "#3F8A5C"),
        ("F1", "F1_cv", "#D9A441"),
    ]

    for ax, outcome in zip(axes, OUTCOMES):
        sub = metrics.loc[metrics["outcome"] == outcome].set_index("fase").loc[PHASE_ORDER].reset_index()
        for label, column, color in series:
            ax.plot(x, sub[column].to_numpy(), marker="o", lw=2.1, ms=4.8, color=color, label=label)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#ECECEA", lw=0.8)
        ax.set_title(OUTCOMES[outcome]["label"])
        ax.text(
            0.02,
            0.07,
            f"Prevalencia: {sub['prevalencia'].iloc[0] * 100:.1f}%",
            transform=ax.transAxes,
            color="#6E7B91",
            fontsize=9,
        )

    axes[0].legend(loc="upper left", ncol=3, fontsize=8.4)
    axes[0].set_ylabel("Valor")
    axes[2].set_ylabel("Valor")
    fig.suptitle("Balance entre detección de casos y control de falsas alarmas", y=1.02, fontsize=16)
    save_figure(fig, "metricas_balance.svg")


def compute_top_features(shap: pd.DataFrame, feature_lists: dict, dictionary: dict[str, str]) -> pd.DataFrame:
    feature_sets = {phase: set(features) for phase, features in feature_lists.items()}
    top = shap.head(15).copy()
    labels = []
    earliest = []
    for feature in top["feature"]:
        labels.append(display_feature(feature, dictionary))
        found = next((phase for phase in PHASE_ORDER if feature in feature_sets[phase]), PHASE_ORDER[-1])
        earliest.append(found)
    top["feature_label"] = labels
    top["earliest_phase"] = earliest
    top["earliest_phase_label"] = [PHASE_LOOKUP[phase]["display"] for phase in earliest]
    return top


def chart_feature_importance(top: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 7.4), gridspec_kw={"width_ratios": [1.35, 0.95]})
    ordered = top.sort_values("mean_abs_shap", ascending=True)
    labels = [wrap_label(label, 30) for label in ordered["feature_label"]]
    colors = [PHASE_COLORS[phase] for phase in ordered["earliest_phase"]]

    ax1.barh(labels, ordered["mean_abs_shap"], color=colors, edgecolor="#FFFFFF", lw=0.8)
    ax1.set_xlabel("Importancia media absoluta SHAP")
    ax1.set_title("Variables que más movieron el riesgo de talla baja")
    ax1.grid(axis="x", color="#ECECEA", lw=0.8)

    y_positions = np.arange(len(ordered))
    for idx in range(len(PHASE_ORDER)):
        ax2.axvline(idx, color="#ECECEA", lw=0.8)
    for ypos, (_, row) in enumerate(ordered.iterrows()):
        phase_idx = PHASE_ORDER.index(row["earliest_phase"])
        ax2.hlines(ypos, 0, phase_idx, color=PHASE_COLORS[row["earliest_phase"]], lw=2, alpha=0.45)
        ax2.plot(
            phase_idx,
            ypos,
            marker="o",
            ms=8,
            color=PHASE_COLORS[row["earliest_phase"]],
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.0,
        )

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([""] * len(y_positions))
    ax2.set_xticks(range(len(PHASE_ORDER)), [PHASE_LOOKUP[phase]["short"] for phase in PHASE_ORDER])
    ax2.set_xlim(-0.3, len(PHASE_ORDER) - 0.7)
    ax2.set_title("Primera fase en la que el dato ya existe")
    fig.suptitle("Importancia global y momento clínico de disponibilidad", y=1.02, fontsize=16)
    save_figure(fig, "features_importancia_etapas.svg")


def chart_stage_feature_bars(metadata: dict, dictionary: dict[str, str]) -> None:
    model_lookup = {
        (item["outcome"], item["fase"]): item["model_file"]
        for item in metadata["modelos"]
    }

    for outcome, info in OUTCOMES.items():
        fig, axes = plt.subplots(4, 2, figsize=(15.6, 15.8))
        axes = axes.flatten()

        for idx, phase in enumerate(PHASE_ORDER):
            ax = axes[idx]
            model_file = model_lookup.get((outcome, phase))
            model_path = ROOT / "notebooks" / model_file if model_file else None

            if model_path is None or not model_path.exists():
                ax.axis("off")
                ax.set_title(PHASE_LOOKUP[phase]["display"], pad=10)
                ax.text(
                    0.5,
                    0.55,
                    "Modelo no disponible\nen artefactos locales",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="#6E7B91",
                    transform=ax.transAxes,
                )
                continue

            model = lgb.Booster(model_file=str(model_path))
            names = model.feature_name()
            gains = model.feature_importance(importance_type="gain")
            frame = (
                pd.DataFrame({"feature": names, "gain": gains})
                .loc[lambda df: df["gain"] > 0]
                .sort_values("gain", ascending=False)
                .head(6)
                .copy()
            )
            total = float(frame["gain"].sum()) or 1.0
            frame["gain_pct"] = frame["gain"] / total * 100.0
            frame["label"] = frame["feature"].map(lambda feature: display_feature(feature, dictionary))
            ordered = frame.sort_values("gain_pct", ascending=True)

            ax.barh(
                [wrap_label(label, width=24) for label in ordered["label"]],
                ordered["gain_pct"],
                color=PHASE_COLORS[phase],
                alpha=0.92,
            )
            ax.set_title(PHASE_LOOKUP[phase]["display"], pad=10)
            ax.grid(axis="x", color="#ECECEA", lw=0.8)
            ax.set_xlabel("Importancia relativa dentro del top 6 (%)")
            ax.tick_params(axis="y", labelsize=8.5)
            for ypos, value in enumerate(ordered["gain_pct"]):
                ax.text(value + 0.25, ypos, f"{value:.1f}", va="center", fontsize=8, color="#4A4A4A")

        axes[-1].axis("off")
        axes[-1].text(
            0.02,
            0.9,
            "Cómo leer estos paneles",
            fontsize=13,
            fontweight="semibold",
            transform=axes[-1].transAxes,
        )
        axes[-1].text(
            0.02,
            0.74,
            f"Cada panel muestra las seis variables que más usa el modelo de {info['short'].lower()} en una fase específica.",
            fontsize=10.5,
            color="#4A4A4A",
            transform=axes[-1].transAxes,
            wrap=True,
        )
        axes[-1].text(
            0.02,
            0.5,
            "La comparación principal es dentro de cada fase. Las barras ayudan a ver cómo cambia la lectura clínica cuando entran controles posteriores.",
            fontsize=10.5,
            color="#4A4A4A",
            transform=axes[-1].transAxes,
            wrap=True,
        )
        axes[-1].text(
            0.02,
            0.24,
            "En fases tardías suelen dominar mediciones directas de crecimiento; en fases tempranas pesan más antecedentes, nacimiento y hospitalización.",
            fontsize=10.5,
            color="#4A4A4A",
            transform=axes[-1].transAxes,
            wrap=True,
        )

        fig.suptitle(f"Variables más importantes por fase para {info['short'].lower()}", y=0.995, fontsize=16)
        fig.subplots_adjust(top=0.93, hspace=0.58, wspace=0.52)
        filename = STAGE_FEATURE_CHARTS[outcome]
        fig.savefig(CHART_DIR / filename, format="svg", bbox_inches="tight")
        if outcome == "Stunting":
            fig.savefig(CHART_DIR / "features_por_etapa_barras.svg", format="svg", bbox_inches="tight")
        plt.close(fig)


def chart_model_inventory(metadata: dict) -> None:
    rows = []
    for item in metadata["modelos"]:
        rows.append(
            {
                "outcome": item["outcome"],
                "phase": item["fase"],
                "n_pos": item["n_pos"],
                "n_samples": item["n_samples"],
                "auc_cv": item["auc_cv"],
                "rounds": item["n_rounds"],
                "exists": (ROOT / "notebooks" / item["model_file"]).exists(),
            }
        )
    frame = pd.DataFrame(rows)
    matrix = frame.pivot(index="outcome", columns="phase", values="auc_cv").loc[list(OUTCOMES), PHASE_ORDER]
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    im = ax.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(PHASE_ORDER)), [PHASE_LOOKUP[phase]["short"] for phase in PHASE_ORDER])
    ax.set_yticks(range(len(OUTCOMES)), [OUTCOMES[outcome]["short"] for outcome in OUTCOMES])

    for i, outcome in enumerate(OUTCOMES):
        for j, phase in enumerate(PHASE_ORDER):
            ax.text(j, i, f"{matrix.loc[outcome, phase]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title("Inventario de modelos por desenlace y fase")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("AUC CV")
    save_figure(fig, "inventario_modelos.svg")


def compute_missingness(feature_lists: dict, dictionary: dict[str, str]) -> dict:
    features = sorted({feature for items in feature_lists.values() for feature in items})
    path = ROOT / "notebooks" / "metricas" / "pacientes_dashboard.csv"
    df = pd.read_csv(
        path,
        usecols=lambda col: col in set(features),
        na_values=["", "#NULL!", "NULL"],
        low_memory=False,
    )

    phases = []
    for phase in PHASE_ORDER:
        cols = [feature for feature in feature_lists[phase] if feature in df.columns]
        missing_by_feature = df[cols].isna().mean()
        available_by_patient = df[cols].notna().mean(axis=1)
        phases.append(
            {
                "phase": PHASE_LOOKUP[phase]["display"],
                "short": PHASE_LOOKUP[phase]["short"],
                "features": len(cols),
                "meanMissing": round(float(missing_by_feature.mean() * 100), 1),
                "medianMissing": round(float(missing_by_feature.median() * 100), 1),
                "featuresOver50": round(float((missing_by_feature > 0.5).mean() * 100), 1),
                "medianPatientAvailable": round(float(available_by_patient.median() * 100), 1),
                "patientsWith75Available": round(float((available_by_patient >= 0.75).mean() * 100), 1),
            }
        )

    f6_cols = [feature for feature in feature_lists["F6_9meses"] if feature in df.columns]
    top_missing = df[f6_cols].isna().mean().sort_values(ascending=False).head(8)
    return {
        "rows": int(len(df)),
        "phases": phases,
        "topMissingF6": [
            {
                "feature": feature,
                "label": display_feature(feature, dictionary),
                "missing": round(float(value * 100), 1),
            }
            for feature, value in top_missing.items()
        ],
    }


def build_summary(
    metrics: pd.DataFrame,
    metadata: dict,
    readme: dict,
    feature_lists: dict,
    top_features: pd.DataFrame,
    missingness: dict,
) -> dict:
    phases = []
    for phase in PHASES:
        item = dict(phase)
        item["features"] = len(feature_lists[phase["id"]])
        phases.append(item)

    metric_method = "Validación cruzada estratificada de 5 particiones, con predicciones fuera de cada partición"

    matrix = []
    for phase in PHASE_ORDER:
        row = {"phase": PHASE_LOOKUP[phase]["display"], "short": PHASE_LOOKUP[phase]["short"]}
        for outcome in OUTCOMES:
            value = metrics.loc[(metrics["outcome"] == outcome) & (metrics["fase"] == phase), "AUC_cv"].iloc[0]
            row[outcome] = round(float(value), 3)
        matrix.append(row)

    outcome_cards = []
    for outcome, info in OUTCOMES.items():
        sub = metrics.loc[metrics["outcome"] == outcome].set_index("fase").loc[PHASE_ORDER].reset_index()
        best = sub.iloc[sub["AUC_cv"].argmax()]
        outcome_cards.append(
            {
                "id": outcome,
                "label": info["label"],
                "short": info["short"],
                "clinical": info["clinical"],
                "n": int(best["n_total"]),
                "positives": int(best["n_positivos"]),
                "prevalence": round(float(best["prevalencia"]) * 100, 1),
                "bestPhase": PHASE_LOOKUP[best["fase"]]["display"],
                "aucF0": round(float(sub["AUC_cv"].iloc[0]), 3),
                "aucF6": round(float(sub["AUC_cv"].iloc[-1]), 3),
                "gain": round(float(sub["AUC_cv"].iloc[-1] - sub["AUC_cv"].iloc[0]), 3),
                "sensF6": round(float(sub["Sens_cv"].iloc[-1]), 3),
                "specF6": round(float(sub["Spec_cv"].iloc[-1]), 3),
                "f1F6": round(float(sub["F1_cv"].iloc[-1]), 3),
            }
        )

    rounds = [int(item["n_rounds"]) for item in metadata["modelos"]]
    positives = {
        outcome: int(metrics.loc[metrics["outcome"] == outcome, "n_positivos"].iloc[0])
        for outcome in OUTCOMES
    }
    outcome_balance = []
    for outcome, info in OUTCOMES.items():
        row = metrics.loc[metrics["outcome"] == outcome].iloc[0]
        n_total = int(row["n_total"])
        n_pos = int(row["n_positivos"])
        n_neg = n_total - n_pos
        outcome_balance.append(
            {
                "outcome": info["label"],
                "nTotal": n_total,
                "nPositive": n_pos,
                "prevalence": round(float(row["prevalencia"] * 100), 1),
                "scalePosWeight": round(float(n_neg / max(n_pos, 1)), 2),
            }
        )

    phase_feature_counts = [
        {
            "phase": PHASE_LOOKUP[phase]["display"],
            "features": len(feature_lists[phase]),
        }
        for phase in PHASE_ORDER
    ]

    return {
        "project": {
            "title": "Predicción del riesgo de malnutrición a 12 meses en prematuros y recién nacidos con bajo peso",
            "records": f"{int(readme['n_pacientes']):,}".replace(",", ".") + " pacientes",
            "variables": f"{max(len(features) for features in feature_lists.values())} variables acumuladas máximas",
            "models": f"{len(metadata['modelos'])} modelos (4 desenlaces x 7 fases)",
            "population": "Niños del Programa Madre Canguro Integral de la Fundación Canguro",
            "generated": readme["generado"],
            "method": metric_method,
        },
        "phases": phases,
        "outcomes": outcome_cards,
        "metricsMatrix": matrix,
        "topFeatures": [
            {
                "feature": row["feature"],
                "label": row["feature_label"],
                "importance": round(float(row["mean_abs_shap"]), 3),
                "earliestPhase": row["earliest_phase_label"],
            }
            for _, row in top_features.iterrows()
        ],
        "stageFeatureCharts": [
            {
                "outcome": outcome,
                "label": info["label"],
                "short": info["short"],
                "image": f"assets/charts/{STAGE_FEATURE_CHARTS[outcome]}",
            }
            for outcome, info in OUTCOMES.items()
        ],
        "missingness": missingness,
        "technical": {
            "trainingDate": metadata["fecha"],
            "sourceNotebook": readme["fuente"],
            "metricMethod": metric_method,
            "modelFamily": "LightGBM, árboles de decisión potenciados por gradiente",
            "roundRange": f"{min(rounds)} a {max(rounds)} iteraciones de boosting",
            "reportedModels": len(metadata["modelos"]),
            "positiveCounts": positives,
            "phaseFeatureCounts": phase_feature_counts,
            "outcomeBalance": outcome_balance,
            "params": [
                ["objective", "binary", "Cada modelo predice si aparece o no un desenlace."],
                ["metric", "auc", "Optimiza y monitorea separación entre casos y no casos."],
                ["learning_rate", "0.05", "Hace que cada árbol corrija de forma gradual al anterior."],
                ["num_leaves", "63", "Permite capturar interacciones no lineales entre variables."],
                ["min_child_samples", "30", "Evita reglas demasiado específicas para pocos pacientes."],
                ["feature_fraction", "0.8", "Cada iteración usa una fracción de variables para reducir sobreajuste."],
                ["bagging_fraction", "0.8", "Cada iteración usa una fracción de filas para estabilizar el entrenamiento."],
                ["bagging_freq", "5", "Aplica bagging cada cinco iteraciones."],
                ["reg_alpha / reg_lambda", "0.1 / 0.1", "Penaliza modelos excesivamente complejos."],
                ["detención temprana (early_stopping)", "50", "Detiene el entrenamiento si el AUC de validación no mejora."],
                ["seed", "42", "Hace reproducible la partición y el entrenamiento."],
            ],
            "baseline": {
                "model": "Regresión logística L1 balanceada",
                "auc": 0.9104,
                "aucStd": 0.0051,
                "lgbmAuc": 0.9288,
                "gain": 0.0184,
                "selectedVariables": "161 de 200",
            },
            "artifacts": [
                ["modelos_pmci/modelo_<desenlace>_<fase>.lgb", "Modelo LightGBM entrenado por desenlace y fase."],
                ["modelos_pmci/features_<fase>.json", "Lista de variables acumuladas disponibles en cada fase."],
                ["metricas/metricas_por_fase.csv", "AUC, sensibilidad, especificidad y F1 por desenlace y fase."],
                ["metricas/shap_importancia_global.csv", "Importancia SHAP global disponible para talla baja en F6."],
                ["metricas/shap_values.csv", "Valores SHAP por paciente para explicabilidad."],
                ["metricas/pacientes_dashboard.csv", "Pacientes con desenlaces, predicciones por fase y variables para visualizaciones."],
                ["metricas/cluster_perfiles.csv", "Perfiles agregados de trayectorias de riesgo."],
            ],
        },
    }


def write_data_script(summary: dict) -> None:
    payload = "window.ARTICLE_DATA = " + json.dumps(summary, ensure_ascii=False, indent=2) + ";\n"
    (DATA_DIR / "article-data.js").write_text(payload, encoding="utf-8")


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC_PARAMS)

    metrics, metadata, readme, shap, feature_lists, dictionary = load_inputs()
    top_features = compute_top_features(shap, feature_lists, dictionary)

    chart_auc_evolution(metrics)
    chart_metric_balance(metrics)
    chart_feature_importance(top_features)
    chart_stage_feature_bars(metadata, dictionary)
    chart_model_inventory(metadata)
    missingness = compute_missingness(feature_lists, dictionary)
    summary = build_summary(metrics, metadata, readme, feature_lists, top_features, missingness)
    write_data_script(summary)


if __name__ == "__main__":
    main()
