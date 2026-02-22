"""
Modelo predictivo de malnutrición en niños prematuros y/o de bajo peso al nacer
====================================================================================
Programa Madre Canguro (KMC) - Fundación Canguro

Variable objetivo: indexnutricion12meses
    - Índice de crecimiento armónico a los 12 meses de edad corregida
    - 0 = Crecimiento NO armónico (malnutrición/falla en crecimiento)
    - 1 = Crecimiento armónico (adecuado)

Se utilizan variables prenatales, del nacimiento, hospitalización neonatal
y entrada al PMC para predecir el estado nutricional a los 12 meses.

Uso:
    # Ejecutar todos los modelos con parámetros por defecto
    python malnutrition_model.py

    # Ejecutar solo Gradient Boosting con hiperparámetros personalizados
    python malnutrition_model.py --model gradient_boosting --gb-n-estimators 300 --gb-learning-rate 0.05

    # Ejecutar Logistic Regression y Random Forest, con MLflow remoto
    python malnutrition_model.py --model logistic_regression random_forest \
        --tracking-uri http://<EC2_IP>:5000 --experiment-name produccion_v2

    # Todos los modelos, 5000 filas, sin dashboard
    python malnutrition_model.py --nrows 5000 --no-dashboard
"""

import argparse
import warnings
import logging
import sys

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    balanced_accuracy_score,
)

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================
TARGET = 'indexnutricion12meses'

MODEL_CHOICES = ['logistic_regression', 'random_forest', 'gradient_boosting', 'all']

MODEL_NAME_MAP = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
}

# Variables a EXCLUIR - Identificadores y fechas
ID_COLS = [
    'Idenfinal', 'Iden_Codigo', 'Iden_FechaParto', 'V7a', 'V7', 'V8',
    'V10', 'V10D', 'V10A', 'V10B', 'V10C', 'VAR00003', 'VAR00004',
    'VAR00005', 'VAR00006', 'V195', 'V195A', 'V215', 'V232',
    'HD_FechaEntrada', 'HD_FechaSalida', 'HD_FechaUltimaTrans',
    'ERN_FUM', 'V282', 'filter_$',
]

# Variables de seguimiento posteriores (DATA LEAKAGE)
FOLLOW_UP_PATTERNS = [
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
    'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
    'V39', 'V40', 'V41', 'V42', 'V43',
    'zscorepesotalla2', 'zscorepeso2', 'zscoretalla2', 'zscorePC2',
    'zscorepesotalla3', 'zscorepeso3', 'zscoretalla3', 'zscorePC3',
    'zscorepesotalla6', 'zscorepeso6', 'zscoretalla6', 'zscorePC6',
    'zscorepesotalla9', 'zscorepeso9', 'zscoretalla9', 'zscorePC9',
    'zscorepesotalla12', 'zscorepeso12', 'zscoretalla12', 'zscorePC12',
    'zscorepesotallaOMS2', 'zscorepesoOMS2', 'zscoretallaOMS2', 'zscorePCOMS2',
    'zscoreBMI2', 'zscoreBMI12', 'BMI2', 'BMI12',
    'velocidad',
    'tallametro12m', 'pesokilo12m', 'BMI12', 'zscoreBMI12',
    'zscorepesotallaOMS1',
]

OUTCOME_COLS = [
    TARGET, 'Indexnutricion40sem', 'indexnutricion12meses',
    'tallametro12m', 'pesokilo12m', 'BMInacermas2DE',
    'edaddestete', 'Diasoxigenoambulatorio', 'PesosalidaPC', 'edadsalidaPC',
    'edadgestasalPC', 'gestaentrada', 'gestaentradacat', 'gestasalsindecimales',
    'ROP', 'ROPcirugia', 'ROPciego',
    'resptometria', 'audiometria', 'oftalmologia',
    'formulaversusLME3m', 'mixtaversusLME3m',
    'nutmadre', 'Nutpadre',
    'SGAprema', 'LBWI', 'preterm',
    'MUERTE1ANO', 'mortalidadhasta40sem', 'mortalidad40sem12meses',
    'Desercionreal12meses', 'desercioncat', 'motivosimplificadosdesercion',
    'underweight12m', 'Overweightorobesity12m',
    'examenneurodurante12meses', 'examenneuropsico12meses',
    'rehosp40a12meses', 'rehosp40',
    'DIASTOT08', 'DIASTOT09', 'DIASTOT10', 'DIASTOT11', 'DIASTOT12',
    'REHOSP08', 'REHOSP09', 'REHOSP10', 'REHOSP11', 'REHOSP12',
    'CONSULT08', 'CONSULT09', 'CONSULT10', 'CONSULT11', 'CONSULT12',
    'NEURO40', 'ali40', 'ali3m', 'ali6m', 'ali9m', 'ali12m',
    'vino40', 'vino3m', 'vino6m', 'vino9m', 'vino12m',
    'algoLM40sem', 'algoLM3meses', 'algoLM6meses',
    'algoLA40', 'algoLA3m', 'algoLA6m', 'algoLA9m', 'algoLA12m',
    'LME40', 'LME3m', 'LME6m',
    'infanib3m', 'infanib6m', 'infanib9m', 'infanib12m',
    'rsm6m', 'rsm12m', 'CD6', 'CD12',
    'IQ6cat', 'IQ12cat',
    'riesgoPC12m',
    'vacunas40semBCG', 'vacunas40semHepB',
    'vacunas12mBCG', 'vacunas12mMMR', 'vacunas12mneumo',
    'vacunas12mrota', 'vacunas12mpenta', 'vacunas12mHepB',
    'indexvacuna12msinMMR', 'indexvacuna12m',
    'PrimaryLast',
    'oftalmopato', 'optometriapato', 'audiometriapato',
    'gananciapesoentradapeso40sem', 'gananciapesoentradapeso40semhosp',
    'Gananciatallaentradatalla40sem', 'Gananciatallaentradatalla40semhosp',
    'velocidadzscorepeso40_3m', 'velocidadzscore3m_40semOMS',
    'velocidad12_9mesesOMS', 'velocidad9_6mesesOMS', 'velocidad6_3mesesOMS',
    'zscoreBMI12cat', 'zscorepesotalla12cat', 'zscorepeso12cat',
    'zscoretalla12cat', 'zscorePC12cat',
    'zscorepesotalla2cat', 'zscorepeso2cat', 'zscoretalla2cat', 'zscorePC2cat',
    'zscorepesoOMS2cat', 'zscoretallaOMS2cat', 'zscorePCOMS2cat',
    'zscorepesotalla3cat', 'zscorepeso3cat', 'zscoretalla3cat', 'zscorePC3cat',
    'zscorepesotalla6cat', 'zscorepeso6cat', 'zscoretalla6cat', 'zscorePC6cat',
    'zscorepesotalla9cat', 'zscorepeso9cat', 'zscoretalla9cat', 'zscorePC9cat',
]


# ============================================================================
# ARGUMENTOS CLI
# ============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Modelo predictivo de malnutrición - Programa Madre Canguro (MLflow)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python malnutrition_model.py --model all
  python malnutrition_model.py --model gradient_boosting --gb-n-estimators 300
  python malnutrition_model.py --model logistic_regression random_forest \\
      --tracking-uri http://ec2-ip:5000 --experiment-name prod_v2
        """,
    )

    # --- Selección de modelo ---
    parser.add_argument(
        '--model', nargs='+', default=['all'], choices=MODEL_CHOICES,
        help='Modelo(s) a entrenar. Opciones: logistic_regression, random_forest, '
             'gradient_boosting, all (default: all)',
    )

    # --- Datos ---
    data = parser.add_argument_group('Datos')
    data.add_argument(
        '--data-path', type=str, default=None,
        help='Ruta al archivo Excel de datos. Default: data/KMC-70k-...xlsx',
    )
    data.add_argument(
        '--dict-path', type=str, default=None,
        help='Ruta al diccionario de variables. Default: data/KMC-70K-diccionario...xlsx',
    )
    data.add_argument('--nrows', type=int, default=10000, help='Filas a leer del dataset (default: 10000)')
    data.add_argument('--test-size', type=float, default=0.2, help='Proporción test split (default: 0.2)')
    data.add_argument('--cv-folds', type=int, default=5, help='Folds para cross-validation (default: 5)')
    data.add_argument('--random-state', type=int, default=42, help='Semilla aleatoria (default: 42)')
    data.add_argument(
        '--missing-threshold', type=float, default=0.70,
        help='Umbral de missing values para eliminar columnas (default: 0.70)',
    )
    data.add_argument(
        '--correlation-threshold', type=float, default=0.95,
        help='Umbral de correlación para eliminar features colineales (default: 0.95)',
    )

    # --- MLflow ---
    mf = parser.add_argument_group('MLflow')
    mf.add_argument(
        '--tracking-uri', type=str, default='mlruns',
        help='MLflow tracking URI. Ej: http://<EC2_IP>:5000 o ruta local (default: mlruns)',
    )
    mf.add_argument(
        '--experiment-name', type=str, default='malnutricion-kmc',
        help='Nombre del experimento en MLflow (default: malnutricion-kmc)',
    )
    mf.add_argument(
        '--run-name', type=str, default=None,
        help='Nombre del run en MLflow. Si no se especifica se genera automáticamente.',
    )
    mf.add_argument(
        '--register-model', action='store_true',
        help='Registrar el mejor modelo en el Model Registry de MLflow',
    )
    mf.add_argument(
        '--registered-model-name', type=str, default='malnutricion-kmc-best',
        help='Nombre para el modelo registrado (default: malnutricion-kmc-best)',
    )

    # --- Hiperparámetros Logistic Regression ---
    lr = parser.add_argument_group('Logistic Regression')
    lr.add_argument('--lr-C', type=float, default=1.0, help='Regularización inversa C (default: 1.0)')
    lr.add_argument('--lr-max-iter', type=int, default=1000, help='Iteraciones máximas (default: 1000)')
    lr.add_argument(
        '--lr-solver', type=str, default='lbfgs',
        choices=['lbfgs', 'liblinear', 'saga', 'newton-cg'],
        help='Solver (default: lbfgs)',
    )

    # --- Hiperparámetros Random Forest ---
    rf = parser.add_argument_group('Random Forest')
    rf.add_argument('--rf-n-estimators', type=int, default=300, help='Número de árboles (default: 300)')
    rf.add_argument('--rf-max-depth', type=int, default=15, help='Profundidad máxima (default: 15)')
    rf.add_argument('--rf-min-samples-split', type=int, default=10, help='Min samples split (default: 10)')
    rf.add_argument('--rf-min-samples-leaf', type=int, default=5, help='Min samples leaf (default: 5)')

    # --- Hiperparámetros Gradient Boosting ---
    gb = parser.add_argument_group('Gradient Boosting')
    gb.add_argument('--gb-n-estimators', type=int, default=200, help='Número de estimadores (default: 200)')
    gb.add_argument('--gb-max-depth', type=int, default=5, help='Profundidad máxima (default: 5)')
    gb.add_argument('--gb-learning-rate', type=float, default=0.1, help='Learning rate (default: 0.1)')
    gb.add_argument('--gb-subsample', type=float, default=0.8, help='Subsample ratio (default: 0.8)')
    gb.add_argument('--gb-min-samples-split', type=int, default=10, help='Min samples split (default: 10)')
    gb.add_argument('--gb-min-samples-leaf', type=int, default=5, help='Min samples leaf (default: 5)')

    # --- Output ---
    out = parser.add_argument_group('Output')
    out.add_argument('--output-dir', type=str, default=None, help='Directorio de salida (default: output/)')
    out.add_argument('--no-dashboard', action='store_true', help='Omitir generación del Excel dashboard')
    out.add_argument('--no-plots', action='store_true', help='Omitir generación de gráficos EDA/resultados')

    return parser.parse_args(argv)


# ============================================================================
# CARGA DE DATOS
# ============================================================================
def load_data(data_path, dict_path, nrows):
    logger.info("=" * 80)
    logger.info("1. CARGA DE DATOS")
    logger.info("=" * 80)

    logger.info(f"Leyendo datos desde: {data_path}")
    df_raw = pd.read_excel(data_path, nrows=nrows)
    logger.info(f"Dimensiones originales: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

    dict_df = pd.read_excel(dict_path, sheet_name='VARS-(KMC70k)')
    dict_df = dict_df[['NOMBRE EN LA BdeD', 'ID-VAR', 'VAR-SHORT DESCRIPTION',
                        'VAR-TYPE-prim', 'VAR-MISSING-VALUE']].dropna(subset=['NOMBRE EN LA BdeD'])
    logger.info(f"Variables en diccionario: {len(dict_df)}")

    return df_raw, dict_df


# ============================================================================
# SELECCIÓN DE FEATURES
# ============================================================================
def select_features(df_raw):
    logger.info("=" * 80)
    logger.info("2. DEFINICIÓN DE VARIABLE OBJETIVO Y SELECCIÓN DE FEATURES")
    logger.info("=" * 80)

    cols_to_exclude = set(ID_COLS + OUTCOME_COLS)

    for col in df_raw.columns:
        if col.startswith('V') and col[1:].replace('_', '').replace('A', '').replace('B', '').replace('C', '').replace('D', '').replace('E', '').replace('F', '').replace('G', '').replace('H', '').replace('I', '').replace('J', '').replace('K', '').replace('L', '').isdigit():
            try:
                num = int(''.join(c for c in col[1:] if c.isdigit()))
                if num >= 218:
                    cols_to_exclude.add(col)
            except ValueError:
                pass
        for pat in FOLLOW_UP_PATTERNS:
            if pat in col:
                cols_to_exclude.add(col)

    feature_cols = [c for c in df_raw.columns if c not in cols_to_exclude and c != TARGET]
    logger.info(f"Columnas totales: {len(df_raw.columns)}")
    logger.info(f"Columnas excluidas: {len(cols_to_exclude)}")
    logger.info(f"Features seleccionadas: {len(feature_cols)}")
    return feature_cols


# ============================================================================
# PREPROCESAMIENTO
# ============================================================================
def preprocess(df_raw, feature_cols, dict_df, missing_threshold=0.70):
    logger.info("=" * 80)
    logger.info("3-4. PREPROCESAMIENTO")
    logger.info("=" * 80)

    df = df_raw[feature_cols + [TARGET]].copy()

    # Convertir #NULL! a NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace('#NULL!', np.nan)

    # Limpiar variable objetivo
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    n_before = len(df)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    n_after = len(df)
    logger.info(f"Filas eliminadas por target faltante: {n_before - n_after}")
    logger.info(f"Filas restantes: {n_after}")
    logger.info(f"Proporción clase 1 (armónico): {df[TARGET].mean():.3f}")
    logger.info(f"Proporción clase 0 (malnutrición): {1 - df[TARGET].mean():.3f}")

    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    # Reemplazar missing values codificados
    missing_map = {}
    for _, row in dict_df.iterrows():
        varname = row['NOMBRE EN LA BdeD']
        miss_val = row['VAR-MISSING-VALUE']
        if pd.notna(miss_val) and varname in X.columns:
            try:
                missing_map[varname] = float(miss_val)
            except (ValueError, TypeError):
                pass

    for col in X.columns:
        if col in missing_map:
            X[col] = X[col].replace(missing_map[col], np.nan)
        X[col] = X[col].replace(-1, np.nan)
        X[col] = X[col].replace(-1.0, np.nan)

    # Convertir columnas object a numéricas cuando sea posible
    for col in X.columns:
        if X[col].dtype == object:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except Exception:
                pass

    # Eliminar fechas
    date_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
    X = X.drop(columns=date_cols, errors='ignore')

    # Eliminar columnas con alto % de missing
    missing_pct = X.isnull().mean()
    high_missing = missing_pct[missing_pct > missing_threshold].index.tolist()
    logger.info(f"Columnas con >{missing_threshold*100:.0f}% missing: {len(high_missing)}")
    X = X.drop(columns=high_missing)

    # Eliminar varianza cero
    numeric_now = X.select_dtypes(include=[np.number]).columns
    zero_var_cols = [c for c in numeric_now if X[c].nunique(dropna=True) <= 1]
    logger.info(f"Columnas con varianza cero: {len(zero_var_cols)}")
    X = X.drop(columns=zero_var_cols)

    logger.info(f"Dimensiones después de limpieza: {X.shape}")
    return X, y


# ============================================================================
# EDA (GRÁFICOS)
# ============================================================================
def run_eda(X, y, output_dir):
    logger.info("=" * 80)
    logger.info("5. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    logger.info("=" * 80)

    # 5.1 Distribución del target
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    y.value_counts().plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
    ax.set_title('Distribución de la Variable Objetivo\n(Índice Nutrición 12 meses EC)')
    ax.set_xlabel('Clase (0=Malnutrición, 1=Armónico)')
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(['0 - No armónico', '1 - Armónico'], rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_01_target_distribution.png', dpi=150)
    plt.close()

    # 5.2 Missing values
    missing_pct_final = X.isnull().mean().sort_values(ascending=False)
    top_missing = missing_pct_final[missing_pct_final > 0.05]
    fig, ax = plt.subplots(figsize=(12, max(6, len(top_missing) * 0.3)))
    top_missing.plot(kind='barh', ax=ax, color='#3498db')
    ax.set_title('Variables con >5% de Valores Faltantes')
    ax.set_xlabel('Proporción de valores faltantes')
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_02_missing_values.png', dpi=150)
    plt.close()

    # 5.3 Correlación con target
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    correlations = pd.DataFrame()
    for col in numeric_features:
        valid_mask = X[col].notna() & y.notna()
        if valid_mask.sum() > 30:
            correlations.loc[col, 'correlation'] = X.loc[valid_mask, col].corr(y[valid_mask])
    correlations = correlations.dropna()
    correlations['abs_corr'] = correlations['correlation'].abs()
    correlations = correlations.sort_values('abs_corr', ascending=False)

    top_corr = correlations.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in top_corr['correlation']]
    ax.barh(range(len(top_corr)), top_corr['correlation'], color=colors)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index, fontsize=8)
    ax.set_xlabel('Correlación con índice nutrición 12m')
    ax.set_title('Top 25 Variables Correlacionadas con el Target')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'eda_03_correlations.png', dpi=150)
    plt.close()

    # 5.4 Distribución de variables clave
    key_vars = ['ERN_Peso', 'ERN_Talla', 'ERN_PC', 'CP_edadmaterna',
                'HD_TotalDiasHospital', 'ERN_Ballard']
    key_vars = [v for v in key_vars if v in X.columns]
    if key_vars:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, var in enumerate(key_vars):
            if i < len(axes):
                for label, color in [(0, '#e74c3c'), (1, '#2ecc71')]:
                    mask = y == label
                    data = X.loc[mask, var].dropna()
                    if len(data) > 0:
                        axes[i].hist(data, bins=30, alpha=0.5, label=f'Clase {label}',
                                     color=color, density=True)
                axes[i].set_title(var, fontsize=10)
                axes[i].legend(fontsize=8)
        for j in range(len(key_vars), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Distribución de Variables Clave por Clase', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'eda_04_key_distributions.png', dpi=150)
        plt.close()

    logger.info("Gráficos EDA generados")


# ============================================================================
# INGENIERÍA DE FEATURES
# ============================================================================
def feature_engineering(X, correlation_threshold=0.95):
    logger.info("=" * 80)
    logger.info("6. INGENIERÍA DE FEATURES Y PREPROCESAMIENTO FINAL")
    logger.info("=" * 80)

    object_cols_final = X.select_dtypes(include=['object']).columns.tolist()

    cat_to_onehot = []
    cat_to_drop = []
    for col in object_cols_final:
        n_unique = X[col].nunique(dropna=True)
        if n_unique <= 30:
            cat_to_onehot.append(col)
        else:
            cat_to_drop.append(col)

    logger.info(f"One-Hot Encoding: {len(cat_to_onehot)} variables")
    logger.info(f"Eliminadas (alta cardinalidad): {len(cat_to_drop)} variables")
    X = X.drop(columns=cat_to_drop)

    if cat_to_onehot:
        X = pd.get_dummies(X, columns=cat_to_onehot, drop_first=True, dummy_na=False)

    # Imputación con mediana
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Eliminar features con alta colinealidad
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    logger.info(f"Features eliminados por colinealidad (>{correlation_threshold}): {len(high_corr_features)}")
    X = X.drop(columns=high_corr_features)

    logger.info(f"Dimensiones finales: {X.shape}")
    return X


# ============================================================================
# CONSTRUCCIÓN DE MODELOS
# ============================================================================
def build_models(args, selected_models):
    models = {}
    class_weight = 'balanced'

    if 'logistic_regression' in selected_models:
        models['Logistic Regression'] = LogisticRegression(
            C=args.lr_C,
            max_iter=args.lr_max_iter,
            solver=args.lr_solver,
            class_weight=class_weight,
            random_state=args.random_state,
        )

    if 'random_forest' in selected_models:
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight=class_weight,
            random_state=args.random_state,
            n_jobs=-1,
        )

    if 'gradient_boosting' in selected_models:
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=args.gb_n_estimators,
            max_depth=args.gb_max_depth,
            learning_rate=args.gb_learning_rate,
            subsample=args.gb_subsample,
            min_samples_split=args.gb_min_samples_split,
            min_samples_leaf=args.gb_min_samples_leaf,
            random_state=args.random_state,
        )

    return models


def get_model_params(args, model_key):
    """Retorna dict de hiperparámetros por modelo para logging en MLflow."""
    if model_key == 'logistic_regression':
        return {
            'C': args.lr_C,
            'max_iter': args.lr_max_iter,
            'solver': args.lr_solver,
        }
    elif model_key == 'random_forest':
        return {
            'n_estimators': args.rf_n_estimators,
            'max_depth': args.rf_max_depth,
            'min_samples_split': args.rf_min_samples_split,
            'min_samples_leaf': args.rf_min_samples_leaf,
        }
    elif model_key == 'gradient_boosting':
        return {
            'n_estimators': args.gb_n_estimators,
            'max_depth': args.gb_max_depth,
            'learning_rate': args.gb_learning_rate,
            'subsample': args.gb_subsample,
            'min_samples_split': args.gb_min_samples_split,
            'min_samples_leaf': args.gb_min_samples_leaf,
        }
    return {}


# ============================================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================
def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, cv):
    logger.info(f"--- {name} ---")

    # Escalar para Logistic Regression
    if 'Logistic' in name:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_te = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    else:
        X_tr, X_te = X_train, X_test

    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1')
    logger.info(f"  CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Balanced Accuracy: {bal_acc:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {auc:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Malnutrición', 'Armónico'])}")

    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_score': f1,
        'roc_auc': auc,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
    }


# ============================================================================
# GRÁFICOS DE RESULTADOS
# ============================================================================
def plot_results(results, y_test, X_train, output_dir):
    logger.info("Generando gráficos de resultados...")

    # Matrices de confusión
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for i, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Malnutrición', 'Armónico'],
                    yticklabels=['Malnutrición', 'Armónico'])
        axes[i].set_title(f'{name}\nF1={r["f1_score"]:.3f}, AUC={r["roc_auc"]:.3f}')
        axes[i].set_ylabel('Real')
        axes[i].set_xlabel('Predicho')
    plt.suptitle('Matrices de Confusión', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'results_01_confusion_matrices.png', dpi=150)
    plt.close()

    # Curvas ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        ax.plot(fpr, tpr, label=f'{name} (AUC={r["roc_auc"]:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title('Curvas ROC - Comparación de Modelos')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'results_02_roc_curves.png', dpi=150)
    plt.close()

    # Curvas Precision-Recall
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y_test, r['y_proba'])
        ax.plot(recall, precision, label=f'{name}', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Curvas Precision-Recall')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_dir / 'results_03_precision_recall.png', dpi=150)
    plt.close()

    # Feature importance (para modelos basados en árboles)
    for name, r in results.items():
        if hasattr(r['model'], 'feature_importances_'):
            fi = pd.DataFrame({
                'feature': X_train.columns,
                'importance': r['model'].feature_importances_,
            }).sort_values('importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 10))
            top_n = min(30, len(fi))
            top_fi = fi.head(top_n)
            ax.barh(range(top_n), top_fi['importance'], color='#3498db')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_fi['feature'], fontsize=8)
            ax.set_xlabel('Importancia')
            ax.set_title(f'Top {top_n} Features ({name})')
            ax.invert_yaxis()
            plt.tight_layout()
            safe_name = name.lower().replace(' ', '_')
            plt.savefig(output_dir / f'results_04_feature_importance_{safe_name}.png', dpi=150)
            plt.close()

            fi.to_csv(output_dir / f'feature_importance_{safe_name}.csv', index=False)

    # Tabla comparativa
    comparison = pd.DataFrame({
        name: {
            'Accuracy': r['accuracy'],
            'Balanced Acc': r['balanced_accuracy'],
            'F1-Score': r['f1_score'],
            'ROC-AUC': r['roc_auc'],
            'CV F1 (mean)': r['cv_f1_mean'],
            'CV F1 (std)': r['cv_f1_std'],
        }
        for name, r in results.items()
    }).round(4)
    comparison.to_csv(output_dir / 'model_comparison.csv')
    logger.info("Gráficos de resultados generados")


# ============================================================================
# DASHBOARD EXCEL
# ============================================================================
def generate_dashboard(df_raw, results, X_train, output_dir, data_path):
    logger.info("=" * 80)
    logger.info("12. GENERACIÓN DE EXCEL PARA DASHBOARD")
    logger.info("=" * 80)

    df_dash = df_raw.copy()
    for col in df_dash.columns:
        if df_dash[col].dtype == object:
            df_dash[col] = df_dash[col].replace('#NULL!', np.nan)
            try:
                df_dash[col] = pd.to_numeric(df_dash[col], errors='ignore')
            except Exception:
                pass

    for col in df_dash.columns:
        try:
            df_dash[col] = df_dash[col].replace(-1, np.nan)
            df_dash[col] = df_dash[col].replace(-1.0, np.nan)
        except Exception:
            pass

    # --- Variables auxiliares ---
    df_dash['ERN_Peso_num'] = pd.to_numeric(df_dash['ERN_Peso'], errors='coerce')
    df_dash['CP_edadmaterna_num'] = pd.to_numeric(df_dash['CP_edadmaterna'], errors='coerce')
    df_dash['ERN_Ballard_num'] = pd.to_numeric(df_dash['ERN_Ballard'], errors='coerce')
    df_dash['HD_TotalDias_num'] = pd.to_numeric(df_dash['HD_TotalDiasHospital'], errors='coerce')
    df_dash['LME40_num'] = pd.to_numeric(df_dash.get('LME40', pd.Series(dtype=float)), errors='coerce')
    df_dash['LME3m_num'] = pd.to_numeric(df_dash.get('LME3m', pd.Series(dtype=float)), errors='coerce')
    df_dash['LME6m_num'] = pd.to_numeric(df_dash.get('LME6m', pd.Series(dtype=float)), errors='coerce')
    df_dash['preterm_num'] = pd.to_numeric(df_dash.get('preterm', pd.Series(dtype=float)), errors='coerce')
    df_dash['PESO1500G_num'] = pd.to_numeric(df_dash.get('PESO1500G', pd.Series(dtype=float)), errors='coerce')
    df_dash['toxemia_num'] = pd.to_numeric(df_dash.get('toxemia', pd.Series(dtype=float)), errors='coerce')
    df_dash['SA_InfUrinaria_num'] = pd.to_numeric(df_dash.get('CP_SA_InfUrinaria', pd.Series(dtype=float)), errors='coerce')
    df_dash['SA_Anemia_num'] = pd.to_numeric(df_dash.get('CP_SA_Anemia', pd.Series(dtype=float)), errors='coerce')
    df_dash['SA_EnfResp_num'] = pd.to_numeric(df_dash.get('CP_SA_EnfRespiratoria', pd.Series(dtype=float)), errors='coerce')
    df_dash['SA_APP_num'] = pd.to_numeric(df_dash.get('CP_SA_APP', pd.Series(dtype=float)), errors='coerce')
    df_dash['SA_Sangrado_num'] = pd.to_numeric(df_dash.get('CP_SA_Sangrado', pd.Series(dtype=float)), errors='coerce')
    df_dash['indexnut12_num'] = pd.to_numeric(df_dash.get('indexnutricion12meses', pd.Series(dtype=float)), errors='coerce')
    df_dash['indexnut40_num'] = pd.to_numeric(df_dash.get('Indexnutricion40sem', pd.Series(dtype=float)), errors='coerce')
    df_dash['muerte_num'] = pd.to_numeric(df_dash.get('MUERTE1ANO', pd.Series(dtype=float)), errors='coerce')
    df_dash['embarazo_mult_num'] = pd.to_numeric(df_dash.get('Iden_embarazoMultiple', pd.Series(dtype=float)), errors='coerce')
    df_dash['ERN_Sexo_num'] = pd.to_numeric(df_dash.get('ERN_Sexo', pd.Series(dtype=float)), errors='coerce')
    df_dash['menosde31sem_num'] = pd.to_numeric(df_dash.get('menosde31sem', pd.Series(dtype=float)), errors='coerce')

    df_dash['hipertension_materna'] = df_dash['toxemia_num'].fillna(0).astype(int)
    df_dash['diabetes_gestacional'] = pd.to_numeric(
        df_dash.get('HD_C_Hipoglicemia', pd.Series(dtype=float)), errors='coerce'
    ).apply(lambda x: 1 if x == 1 else 0)
    df_dash['infeccion_urinaria'] = df_dash['SA_InfUrinaria_num'].fillna(0).astype(int)
    df_dash['edad_extrema'] = df_dash['CP_edadmaterna_num'].apply(
        lambda x: 1 if pd.notna(x) and (x < 20 or x > 35) else 0
    )
    df_dash['peso_menor_1500'] = df_dash['PESO1500G_num'].fillna(0).astype(int)
    df_dash['eg_menor_37'] = df_dash['preterm_num'].fillna(0).astype(int)
    df_dash['lactancia_exclusiva'] = df_dash['LME40_num'].fillna(0).astype(int)

    def clasificar_riesgo(row):
        score = 0
        if row.get('peso_menor_1500', 0) == 1:
            score += 3
        if row.get('menosde31sem_num', 0) == 1:
            score += 3
        if row.get('hipertension_materna', 0) == 1:
            score += 1
        if row.get('infeccion_urinaria', 0) == 1:
            score += 1
        if row.get('edad_extrema', 0) == 1:
            score += 1
        if pd.notna(row.get('ERN_Peso_num')) and row['ERN_Peso_num'] < 1000:
            score += 2
        if row.get('diabetes_gestacional', 0) == 1:
            score += 1
        if score >= 4:
            return 'Alto'
        elif score >= 2:
            return 'Moderado'
        else:
            return 'Bajo'

    df_dash['nivel_riesgo'] = df_dash.apply(clasificar_riesgo, axis=1)
    df_dash['nino_sano'] = (
        (df_dash['indexnut12_num'] == 1) |
        ((df_dash['indexnut12_num'].isna()) & (df_dash['indexnut40_num'] == 1))
    ).astype(int)

    total_pacientes = len(df_dash)
    ninos_sanos = df_dash['nino_sano'].sum()
    prematuros_moderado = (df_dash['nivel_riesgo'] == 'Moderado').sum()
    prematuros_alto = (df_dash['nivel_riesgo'] == 'Alto').sum()

    # --- SHEET 1: KPIs ---
    kpis = pd.DataFrame({
        'Indicador': ['Pacientes Totales', 'Niños Sanos', 'Prematuros Riesgo Moderado', 'Prematuros Alto Riesgo'],
        'Valor': [total_pacientes, ninos_sanos, prematuros_moderado, prematuros_alto],
        'Porcentaje': [100.0, round(ninos_sanos / total_pacientes * 100, 1),
                       round(prematuros_moderado / total_pacientes * 100, 1),
                       round(prematuros_alto / total_pacientes * 100, 1)],
    })

    # --- SHEET 2: Indicadores ---
    indicadores = {
        'Hipertensión Materna': df_dash['hipertension_materna'],
        'Diabetes Gestacional': df_dash['diabetes_gestacional'],
        'Infección Urinaria': df_dash['infeccion_urinaria'],
        'Edad Materna (<20 o >35 años)': df_dash['edad_extrema'],
        'Peso al Nacer <1500g': df_dash['peso_menor_1500'],
        'Edad Gestacional <37 semanas': df_dash['eg_menor_37'],
    }
    indicadores_rows = []
    for nombre, serie in indicadores.items():
        total_validos = serie.notna().sum()
        positivos = (serie == 1).sum()
        pct = round(positivos / total_validos * 100, 1) if total_validos > 0 else 0
        riesgo = 'Alto' if pct > 25 else ('Moderado' if pct > 15 else 'Bajo')
        indicadores_rows.append({
            'Indicador': nombre, 'Porcentaje (%)': pct,
            'Total Afectados': positivos, 'Total Evaluados': total_validos, 'Nivel Riesgo': riesgo,
        })
    dias_hosp = df_dash['HD_TotalDias_num'].dropna()
    indicadores_rows.append({
        'Indicador': 'Días Promedio de Hosp.', 'Porcentaje (%)': round(dias_hosp.mean(), 1),
        'Total Afectados': round(dias_hosp.mean(), 1), 'Total Evaluados': len(dias_hosp),
        'Nivel Riesgo': 'Alto' if dias_hosp.mean() > 14 else 'Moderado',
    })
    df_indicadores = pd.DataFrame(indicadores_rows)

    # --- SHEET 3: Lactancia ---
    condiciones_lact = {
        'Hipertensión': 'hipertension_materna', 'Diabetes Gestacional': 'diabetes_gestacional',
        'Infección Urinaria': 'infeccion_urinaria', 'Edad Extrema': 'edad_extrema',
    }
    lactancia_rows = []
    for cond_nombre, cond_col in condiciones_lact.items():
        for lm_nombre, lm_col in [('40 semanas', 'LME40_num'), ('3 meses', 'LME3m_num'), ('6 meses', 'LME6m_num')]:
            mask_cond = df_dash[cond_col] == 1
            lm_series = df_dash.loc[mask_cond, lm_col].dropna()
            n_con = len(lm_series)
            pct_con = round(lm_series.mean() * 100, 1) if n_con > 0 else 0
            mask_sin = df_dash[cond_col] == 0
            lm_sin = df_dash.loc[mask_sin, lm_col].dropna()
            n_sin = len(lm_sin)
            pct_sin = round(lm_sin.mean() * 100, 1) if n_sin > 0 else 0
            lactancia_rows.append({
                'Condición': cond_nombre, 'Período Lactancia': lm_nombre,
                'LME Con Condición (%)': pct_con, 'N Con Condición': n_con,
                'LME Sin Condición (%)': pct_sin, 'N Sin Condición': n_sin,
            })
    df_lactancia = pd.DataFrame(lactancia_rows)

    # --- SHEET 4: Complicaciones ---
    complicaciones = {
        'Hipertensión/Preeclampsia': 'hipertension_materna',
        'Diabetes Gestacional': 'diabetes_gestacional',
        'Infección Urinaria': 'infeccion_urinaria',
        'Anemia': 'SA_Anemia_num', 'Amenaza Parto Prematuro': 'SA_APP_num',
        'Sangrado': 'SA_Sangrado_num', 'Enfermedad Respiratoria': 'SA_EnfResp_num',
        'Edad Extrema Materna': 'edad_extrema', 'Embarazo Múltiple': 'embarazo_mult_num',
    }
    compl_rows = []
    for nombre, col in complicaciones.items():
        serie = df_dash[col].dropna() if col in df_dash.columns else pd.Series(dtype=float)
        total = len(serie)
        positivos = (serie == 1).sum()
        pct = round(positivos / total * 100, 1) if total > 0 else 0
        compl_rows.append({'Complicación': nombre, 'Pacientes Afectados': positivos,
                           'Total Evaluados': total, 'Porcentaje (%)': pct})
    df_complicaciones = pd.DataFrame(compl_rows).sort_values('Porcentaje (%)', ascending=False)

    # --- SHEET 5: Hospitalización ---
    dias = df_dash['HD_TotalDias_num'].dropna()
    hosp_stats = pd.DataFrame({
        'Estadístico': [
            'Promedio Días Hospitalización', 'Mediana Días Hospitalización',
            'Desviación Estándar', 'Mínimo', 'Máximo',
            'Percentil 25', 'Percentil 75', 'Total Pacientes Evaluados',
            'Pacientes con >7 días', '% Pacientes con >7 días',
            'Pacientes con >14 días', '% Pacientes con >14 días',
            'Pacientes con >30 días', '% Pacientes con >30 días',
        ],
        'Valor': [
            round(dias.mean(), 1), round(dias.median(), 1), round(dias.std(), 1),
            dias.min(), dias.max(), round(dias.quantile(0.25), 1), round(dias.quantile(0.75), 1),
            len(dias), (dias > 7).sum(), round((dias > 7).mean() * 100, 1),
            (dias > 14).sum(), round((dias > 14).mean() * 100, 1),
            (dias > 30).sum(), round((dias > 30).mean() * 100, 1),
        ],
    })
    hosp_riesgo_rows = []
    for riesgo in ['Bajo', 'Moderado', 'Alto']:
        mask = df_dash['nivel_riesgo'] == riesgo
        d = df_dash.loc[mask, 'HD_TotalDias_num'].dropna()
        hosp_riesgo_rows.append({
            'Nivel Riesgo': riesgo, 'N Pacientes': len(d),
            'Promedio Días': round(d.mean(), 1) if len(d) > 0 else 0,
            'Mediana Días': round(d.median(), 1) if len(d) > 0 else 0,
            'Max Días': d.max() if len(d) > 0 else 0,
        })
    df_hosp_riesgo = pd.DataFrame(hosp_riesgo_rows)

    # --- SHEET 6: Bajo Peso ---
    peso = df_dash['ERN_Peso_num'].dropna()
    peso_cats = pd.DataFrame({
        'Categoría Peso': ['Extremo Bajo Peso (<1000g)', 'Muy Bajo Peso (1000-1499g)',
                           'Bajo Peso (1500-2499g)', 'Peso Normal (≥2500g)'],
        'N Pacientes': [(peso < 1000).sum(), ((peso >= 1000) & (peso < 1500)).sum(),
                        ((peso >= 1500) & (peso < 2500)).sum(), (peso >= 2500).sum()],
        'Porcentaje (%)': [round((peso < 1000).mean() * 100, 1),
                           round(((peso >= 1000) & (peso < 1500)).mean() * 100, 1),
                           round(((peso >= 1500) & (peso < 2500)).mean() * 100, 1),
                           round((peso >= 2500).mean() * 100, 1)],
        'Peso Promedio (g)': [
            round(peso[peso < 1000].mean(), 0) if (peso < 1000).sum() > 0 else 0,
            round(peso[(peso >= 1000) & (peso < 1500)].mean(), 0),
            round(peso[(peso >= 1500) & (peso < 2500)].mean(), 0),
            round(peso[peso >= 2500].mean(), 0) if (peso >= 2500).sum() > 0 else 0,
        ],
    })

    # --- SHEET 7: Lactancia General ---
    lact_general = pd.DataFrame({
        'Período': ['40 semanas EG', '3 meses EC', '6 meses EC'],
        'Madres con LME': [df_dash['LME40_num'].dropna().sum(),
                           df_dash['LME3m_num'].dropna().sum(),
                           df_dash['LME6m_num'].dropna().sum()],
        'Total Evaluadas': [df_dash['LME40_num'].dropna().shape[0],
                            df_dash['LME3m_num'].dropna().shape[0],
                            df_dash['LME6m_num'].dropna().shape[0]],
        'Porcentaje LME (%)': [round(df_dash['LME40_num'].dropna().mean() * 100, 1),
                                round(df_dash['LME3m_num'].dropna().mean() * 100, 1),
                                round(df_dash['LME6m_num'].dropna().mean() * 100, 1)],
    })

    # --- SHEET 8: Distribución Riesgo ---
    riesgo_dist = df_dash['nivel_riesgo'].value_counts().reset_index()
    riesgo_dist.columns = ['Nivel Riesgo', 'N Pacientes']
    riesgo_dist['Porcentaje (%)'] = round(riesgo_dist['N Pacientes'] / total_pacientes * 100, 1)
    orden_riesgo = {'Bajo': 0, 'Moderado': 1, 'Alto': 2}
    riesgo_dist['orden'] = riesgo_dist['Nivel Riesgo'].map(orden_riesgo)
    riesgo_dist = riesgo_dist.sort_values('orden').drop(columns='orden')

    riesgo_nut_rows = []
    for riesgo in ['Bajo', 'Moderado', 'Alto']:
        mask = df_dash['nivel_riesgo'] == riesgo
        nut12 = df_dash.loc[mask, 'indexnut12_num'].dropna()
        n_total = mask.sum()
        n_eval = len(nut12)
        n_armonico = (nut12 == 1).sum()
        n_malnut = (nut12 == 0).sum()
        riesgo_nut_rows.append({
            'Nivel Riesgo': riesgo, 'N Total': n_total,
            'N Evaluados Nutrición 12m': n_eval, 'Crecimiento Armónico': n_armonico,
            'Malnutrición': n_malnut,
            '% Armónico': round(n_armonico / n_eval * 100, 1) if n_eval > 0 else 0,
            '% Malnutrición': round(n_malnut / n_eval * 100, 1) if n_eval > 0 else 0,
        })
    df_riesgo_nutricion = pd.DataFrame(riesgo_nut_rows)

    # --- SHEET 9: Resultados ML ---
    model_results = pd.DataFrame({
        'Modelo': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Balanced Accuracy': [r['balanced_accuracy'] for r in results.values()],
        'F1-Score': [r['f1_score'] for r in results.values()],
        'ROC-AUC': [r['roc_auc'] for r in results.values()],
        'CV F1 Mean': [r['cv_f1_mean'] for r in results.values()],
        'CV F1 Std': [r['cv_f1_std'] for r in results.values()],
        'Mejor': ['Sí' if name == max(results, key=lambda k: results[k]['f1_score']) else 'No'
                  for name in results.keys()],
    }).round(4)

    # --- SHEET 10: Feature Importance ---
    # Usar el primer modelo que tenga feature_importances_
    fi_top = pd.DataFrame({'Variable': [], 'Importancia': [], 'Importancia (%)': []})
    for name, r in results.items():
        if hasattr(r['model'], 'feature_importances_'):
            fi = pd.DataFrame({
                'Variable': X_train.columns,
                'Importancia': r['model'].feature_importances_,
            }).sort_values('Importancia', ascending=False).head(30)
            fi['Importancia (%)'] = (fi['Importancia'] * 100).round(2)
            fi_top = fi
            break

    # --- SHEET 11: Demografía ---
    eg = df_dash['ERN_Ballard_num'].dropna()
    eg_cats = pd.DataFrame({
        'Categoría EG': ['Extremo Prematuro (<28 sem)', 'Muy Prematuro (28-31 sem)',
                         'Moderado Prematuro (32-33 sem)', 'Tardío Prematuro (34-36 sem)',
                         'A Término (≥37 sem)'],
        'N Pacientes': [(eg < 28).sum(), ((eg >= 28) & (eg < 32)).sum(),
                        ((eg >= 32) & (eg < 34)).sum(), ((eg >= 34) & (eg < 37)).sum(),
                        (eg >= 37).sum()],
        'Porcentaje (%)': [round((eg < 28).mean() * 100, 1),
                           round(((eg >= 28) & (eg < 32)).mean() * 100, 1),
                           round(((eg >= 32) & (eg < 34)).mean() * 100, 1),
                           round(((eg >= 34) & (eg < 37)).mean() * 100, 1),
                           round((eg >= 37).mean() * 100, 1)],
        'EG Promedio (sem)': [
            round(eg[eg < 28].mean(), 1) if (eg < 28).sum() > 0 else 0,
            round(eg[(eg >= 28) & (eg < 32)].mean(), 1),
            round(eg[(eg >= 32) & (eg < 34)].mean(), 1),
            round(eg[(eg >= 34) & (eg < 37)].mean(), 1),
            round(eg[eg >= 37].mean(), 1) if (eg >= 37).sum() > 0 else 0,
        ],
    })

    sexo = df_dash['ERN_Sexo_num'].dropna()
    sexo_dist = pd.DataFrame({
        'Sexo': ['Masculino (1)', 'Femenino (2)'],
        'N Pacientes': [(sexo == 1).sum(), (sexo == 2).sum()],
        'Porcentaje (%)': [round((sexo == 1).mean() * 100, 1), round((sexo == 2).mean() * 100, 1)],
    })

    edad = df_dash['CP_edadmaterna_num'].dropna()
    edad_cats = pd.DataFrame({
        'Rango Edad Materna': ['<20 años', '20-25 años', '26-30 años', '31-35 años', '>35 años'],
        'N Pacientes': [(edad < 20).sum(), ((edad >= 20) & (edad <= 25)).sum(),
                        ((edad >= 26) & (edad <= 30)).sum(), ((edad >= 31) & (edad <= 35)).sum(),
                        (edad > 35).sum()],
        'Porcentaje (%)': [round((edad < 20).mean() * 100, 1),
                           round(((edad >= 20) & (edad <= 25)).mean() * 100, 1),
                           round(((edad >= 26) & (edad <= 30)).mean() * 100, 1),
                           round(((edad >= 31) & (edad <= 35)).mean() * 100, 1),
                           round((edad > 35).mean() * 100, 1)],
    })

    # --- Escribir Excel ---
    excel_path = output_dir / 'dashboard_data.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        kpis.to_excel(writer, sheet_name='1_KPIs_Principales', index=False)
        df_indicadores.to_excel(writer, sheet_name='2_Indicadores_Clave', index=False)
        df_lactancia.to_excel(writer, sheet_name='3_Lactancia_x_Condicion', index=False)
        df_complicaciones.to_excel(writer, sheet_name='4_Complicaciones_Prenatales', index=False)
        hosp_stats.to_excel(writer, sheet_name='5_Hospitalizacion_Detalle', index=False)
        df_hosp_riesgo.to_excel(writer, sheet_name='5b_Hosp_x_Riesgo', index=False)
        peso_cats.to_excel(writer, sheet_name='6_Bajo_Peso_Nacer', index=False)
        lact_general.to_excel(writer, sheet_name='7_Lactancia_General', index=False)
        riesgo_dist.to_excel(writer, sheet_name='8_Distribucion_Riesgo', index=False)
        df_riesgo_nutricion.to_excel(writer, sheet_name='8b_Riesgo_x_Nutricion', index=False)
        model_results.to_excel(writer, sheet_name='9_Resultados_Modelo', index=False)
        fi_top.to_excel(writer, sheet_name='10_Feature_Importance', index=False)
        eg_cats.to_excel(writer, sheet_name='11a_Edad_Gestacional', index=False)
        sexo_dist.to_excel(writer, sheet_name='11b_Sexo', index=False)
        edad_cats.to_excel(writer, sheet_name='11c_Edad_Materna', index=False)

    logger.info(f"Excel dashboard generado: {excel_path}")
    return excel_path


# ============================================================================
# MAIN
# ============================================================================
def main(argv=None):
    args = parse_args(argv)

    # --- Resolver modelos seleccionados ---
    if 'all' in args.model:
        selected_models = ['logistic_regression', 'random_forest', 'gradient_boosting']
    else:
        selected_models = list(set(args.model))

    logger.info(f"Modelos seleccionados: {selected_models}")

    # --- Resolver rutas ---
    base_dir = Path(__file__).parent
    data_path = Path(args.data_path) if args.data_path else base_dir / "data" / "KMC-70k-93-2024-Malnutricion-conVel-DATA-SPSS-20250322.xlsx"
    dict_path = Path(args.dict_path) if args.dict_path else base_dir / "data" / "KMC-70K-diccionarioVARS-Malnutricion-PhETI-rev20250520-MAIA.xlsx"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- MLflow setup ---
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    logger.info(f"MLflow tracking URI: {args.tracking_uri}")
    logger.info(f"MLflow experiment: {args.experiment_name}")

    # --- Carga y preprocesamiento ---
    df_raw, dict_df = load_data(data_path, dict_path, args.nrows)
    feature_cols = select_features(df_raw)
    X, y = preprocess(df_raw, feature_cols, dict_df, missing_threshold=args.missing_threshold)

    if not args.no_plots:
        run_eda(X, y, output_dir)

    X = feature_engineering(X, correlation_threshold=args.correlation_threshold)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    # --- Construir y entrenar modelos ---
    models = build_models(args, selected_models)
    results = {}

    # Run padre que agrupa todos los modelos de esta ejecución
    with mlflow.start_run(run_name=args.run_name or "malnutricion_pipeline") as parent_run:
        # Log parámetros globales en el run padre
        mlflow.log_params({
            'nrows': args.nrows,
            'test_size': args.test_size,
            'cv_folds': args.cv_folds,
            'random_state': args.random_state,
            'missing_threshold': args.missing_threshold,
            'correlation_threshold': args.correlation_threshold,
            'n_features': X.shape[1],
            'n_samples_train': X_train.shape[0],
            'n_samples_test': X_test.shape[0],
            'target': TARGET,
            'models_selected': ','.join(selected_models),
        })

        for model_key in selected_models:
            display_name = MODEL_NAME_MAP[model_key]
            model = models[display_name]

            # Run hijo por cada modelo
            with mlflow.start_run(run_name=display_name, nested=True) as child_run:
                # Log hiperparámetros del modelo
                model_params = get_model_params(args, model_key)
                mlflow.log_params(model_params)
                mlflow.log_param('model_type', model_key)

                # Entrenar y evaluar
                result = train_and_evaluate(
                    display_name, model, X_train, X_test, y_train, y_test, cv,
                )
                results[display_name] = result

                # Log métricas
                mlflow.log_metrics({
                    'accuracy': result['accuracy'],
                    'balanced_accuracy': result['balanced_accuracy'],
                    'f1_score': result['f1_score'],
                    'roc_auc': result['roc_auc'],
                    'cv_f1_mean': result['cv_f1_mean'],
                    'cv_f1_std': result['cv_f1_std'],
                })

                # Log modelo sklearn
                mlflow.sklearn.log_model(
                    result['model'],
                    artifact_path=f"model_{model_key}",
                    input_example=X_test.iloc[:1],
                )

                # Log feature importance como artefacto si aplica
                if hasattr(result['model'], 'feature_importances_'):
                    fi = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': result['model'].feature_importances_,
                    }).sort_values('importance', ascending=False)
                    fi_path = output_dir / f'feature_importance_{model_key}.csv'
                    fi.to_csv(fi_path, index=False)
                    mlflow.log_artifact(str(fi_path))

        # --- Gráficos de resultados ---
        if not args.no_plots:
            plot_results(results, y_test, X_train, output_dir)
            # Log gráficos como artefactos en el run padre
            for png in output_dir.glob('*.png'):
                mlflow.log_artifact(str(png))

        # --- Mejor modelo ---
        best_name = max(results, key=lambda k: results[k]['f1_score'])
        best = results[best_name]
        logger.info(f"Mejor modelo: {best_name} (F1={best['f1_score']:.4f}, AUC={best['roc_auc']:.4f})")

        mlflow.log_params({
            'best_model': best_name,
            'best_f1': round(best['f1_score'], 4),
            'best_auc': round(best['roc_auc'], 4),
        })

        # Log comparación
        comparison = pd.DataFrame({
            name: {
                'Accuracy': r['accuracy'], 'Balanced Acc': r['balanced_accuracy'],
                'F1-Score': r['f1_score'], 'ROC-AUC': r['roc_auc'],
                'CV F1 (mean)': r['cv_f1_mean'], 'CV F1 (std)': r['cv_f1_std'],
            }
            for name, r in results.items()
        }).round(4)
        comp_path = output_dir / 'model_comparison.csv'
        comparison.to_csv(comp_path)
        mlflow.log_artifact(str(comp_path))

        # --- Registrar mejor modelo en Model Registry ---
        if args.register_model:
            best_model_key = [k for k, v in MODEL_NAME_MAP.items() if v == best_name][0]
            model_uri = f"runs:/{parent_run.info.run_id}/model_{best_model_key}"
            mlflow.register_model(model_uri, args.registered_model_name)
            logger.info(f"Modelo registrado como '{args.registered_model_name}'")

        # --- Dashboard ---
        if not args.no_dashboard:
            excel_path = generate_dashboard(df_raw, results, X_train, output_dir, data_path)
            mlflow.log_artifact(str(excel_path))

    # --- Resumen final ---
    logger.info("=" * 80)
    logger.info("RESUMEN FINAL")
    logger.info("=" * 80)
    for name, r in results.items():
        marker = " <-- MEJOR" if name == best_name else ""
        logger.info(f"  {name}{marker}: F1={r['f1_score']:.4f}, AUC={r['roc_auc']:.4f}, "
                     f"Acc={r['accuracy']:.4f}, CV-F1={r['cv_f1_mean']:.4f}+/-{r['cv_f1_std']:.4f}")

    logger.info("ANÁLISIS COMPLETADO")
    return results


if __name__ == '__main__':
    main()
