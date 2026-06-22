"""
train_final.py
==============
Final stres tespiti modeli - Mezuniyet Projesi v3
Yazar: HCI-ML Project

Phase 3 iyilestirmeleri:
  1. features_5min.csv kullaniliyor (sifir-aktivite filtrelenmis + 5-dk agregat)
  2. 20 feature (5 yeni engineered feature eklendi)
  3. GroupKFold temporal split (temporal leakage onleme)
  4. RandomForest + GradientBoosting karsilastirmasi
  5. Per-user normalizasyon
  6. 3-sinif sadele$tirme
  7. class_weight='balanced'
  8. balanced_accuracy_score + classification_report
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.multioutput import MultiOutputClassifier
import joblib

# -----------------------------------------------
# FEATURE SETI (20 feature: 15 base + 5 engineered)
# -----------------------------------------------
ALL_FEATURES = [
    # Keyboard
    'typing_rate_kps',
    'inter_key_latency_mean_ms',
    'inter_key_latency_std_ms',
    'key_dwell_time_mean_ms',
    'key_dwell_time_std_ms',
    'backspace_count',
    'long_pause_count',
    'special_key_count',
    # Mouse
    'mouse_move_count',
    'mouse_avg_speed_px_s',
    'mouse_std_speed_px_s',
    'mouse_idle_time_ms',
    'mouse_click_count',
    'mouse_path_length_px',
    # Context
    'window_switch_count',
    # Engineered (2nd-order)
    'activity_ratio',
    'mouse_speed_cv',
    'typing_burst_count',
    'idle_to_active_ratio',
    'click_per_move_ratio',
    
    # ---- NATIVE EXP3A ROLLING DELTA FEATURES ----
    'typing_rate_kps_delta', 'typing_rate_kps_rmean3', 'typing_rate_kps_rstd3',
    'inter_key_latency_mean_ms_delta', 'inter_key_latency_mean_ms_rmean3', 'inter_key_latency_mean_ms_rstd3',
    'key_dwell_time_mean_ms_delta', 'key_dwell_time_mean_ms_rmean3', 'key_dwell_time_mean_ms_rstd3',
    'mouse_avg_speed_px_s_delta', 'mouse_avg_speed_px_s_rmean3', 'mouse_avg_speed_px_s_rstd3',
    'mouse_idle_time_ms_delta', 'mouse_idle_time_ms_rmean3', 'mouse_idle_time_ms_rstd3',
    'activity_ratio_delta', 'activity_ratio_rmean3', 'activity_ratio_rstd3'
]

# -----------------------------------------------
# 2-SINIF MAPPING (BINARY)
# -----------------------------------------------
LABEL_MAPS = {
    'stress_val': {
        'F_Great':     'Not_Stressed',
        'F_Good':      'Not_Stressed',
        'Neutral':     'Not_Stressed',
        'S_Stressed':  'Stressed',
        'V_Stressed':  'Stressed',
    },
    'fatigue_val': {
        'No':          'Not_Fatigued',
        'Low':         'Not_Fatigued',
        'Avg':         'Fatigued',
        'Below_Avg':   'Fatigued',
        'Above_Avg':   'Fatigued',
        'V_High':      'Fatigued',
    },
    'energy_val': {
        'V_Energetic': 'High_Energy',
        'S_Energetic': 'High_Energy',
        'Neutral':     'Low_Energy',
        'S_Low_Energy':'Low_Energy',
        'V_Low_Energy':'Low_Energy',
    },
    'pleasant_val': {
        'V_Pleasant':   'Pleasant',
        'S_Pleasant':   'Pleasant',
        'Neutral':      'Unpleasant',
        'S_Unpleasant': 'Unpleasant',
        'V_Unpleasant': 'Unpleasant',
    },
}

TARGET_COLS = list(LABEL_MAPS.keys())
MAPPED_COLS = [f"{c}_3cls" for c in TARGET_COLS]


def apply_label_mapping(df):
    """5-sinifli etiketleri 3-sinifa donustur."""
    for col, mapping in LABEL_MAPS.items():
        new_col = f"{col}_3cls"
        df[new_col] = df[col].map(mapping)
        unmapped = df[df[new_col].isna() & df[col].notna()][col].unique()
        if len(unmapped) > 0:
            print(f"  [WARN] {col} eslestirilemeyen: {unmapped}")
    return df


def per_user_normalize(df, feature_cols):
    """Per-user z-score normalizasyonu."""
    df = df.copy()
    for col in feature_cols:
        df[col] = df[col].astype('float64')
    for user in df['user_id'].unique():
        mask = df['user_id'] == user
        for col in feature_cols:
            col_data = df.loc[mask, col]
            mu = col_data.mean()
            sigma = col_data.std()
            if sigma > 0:
                df.loc[mask, col] = (col_data - mu) / sigma
            else:
                df.loc[mask, col] = 0.0
    return df


def create_temporal_groups(df):
    """
    Label degisimlerini kullanarak temporal grup ID'leri olustur.
    Ayni label'a sahip ardisik satirlar ayni gruba atanir.
    GroupKFold icin kullanilir -> ayni zaman segmenti train+test'e ayrilmaz.
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'Window'])
    
    # Kullanici gecisleri + label degisimi = yeni segment
    user_change = (df['user_id'] != df['user_id'].shift())
    label_change = (df['stress_val_3cls'] != df['stress_val_3cls'].shift())
    df['segment_id'] = (user_change | label_change).cumsum()
    
    return df


def build_rf_pipeline():
    """RandomForest pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])


def build_gbm_pipeline():
    """GradientBoosting pipeline."""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        ))
    ])


def evaluate_target(X, y_series, groups, target_name, n_splits=5):
    """StratifiedGroupKFold CV ile tek bir target degerlendirme."""
    # Az ornekli siniflari filtrele
    counts = y_series.value_counts()
    valid_classes = counts[counts >= n_splits].index
    valid_mask = y_series.isin(valid_classes)
    X_f = X[valid_mask].copy()
    y_f = y_series[valid_mask].copy()
    g_f = groups[valid_mask].copy()

    results = {}
    
    for name, pipe_fn in [("RandomForest", build_rf_pipeline), ("GradientBoosting", build_gbm_pipeline)]:
        pipe = pipe_fn()
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        try:
            y_pred = cross_val_predict(pipe, X_f, y_f, cv=cv, groups=g_f, n_jobs=-1)
        except Exception as e:
            # Fallback: eger GroupKFold basarisiz olursa normal StratifiedKFold dene
            from sklearn.model_selection import StratifiedKFold
            print(f"    [WARN] GroupKFold failed for {name}, falling back to StratifiedKFold: {e}")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            y_pred = cross_val_predict(pipe, X_f, y_f, cv=cv, n_jobs=-1)

        acc = accuracy_score(y_f, y_pred)
        bal_acc = balanced_accuracy_score(y_f, y_pred)
        report = classification_report(y_f, y_pred, zero_division=0)

        print(f"\n  --- {name} ---")
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"\n{report}")
        results[name] = (acc, bal_acc)
    
    return results


def print_feature_importances(X, y_series, feature_names):
    pipe = build_rf_pipeline()
    pipe.fit(X, y_series)
    imps = pipe.named_steps['model'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Imp': imps})
    feat_imp = feat_imp.sort_values('Imp', ascending=False)
    print("\n  Feature Importances:")
    for _, row in feat_imp.iterrows():
        bar = '#' * int(row['Imp'] * 100)
        print(f"    {row['Feature']:>25}: {row['Imp']:.4f}  {bar}")


def train_and_save(X, y_df, model_name, feature_names, use_gbm=False):
    """Tum data uzerinde fit et ve kaydet."""
    from sklearn.multioutput import MultiOutputClassifier
    if use_gbm:
        # GBM does not support class_weight, train each target separately
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', MultiOutputClassifier(
                GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    min_samples_leaf=5, subsample=0.8, random_state=42
                )
            ))
        ])
    else:
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=300, max_depth=None, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
            ))
        ])
    
    pipe.fit(X, y_df)
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump({'pipeline': pipe, 'features': feature_names}, path)
    print(f"\n  [OK] Model kaydedildi: {path}")
    
    class_info = {col: list(pipe.named_steps['model'].estimators_[i].classes_)
                  for i, col in enumerate(y_df.columns)}
    class_path = os.path.join(model_dir, f"{model_name}_classes.pkl")
    joblib.dump(class_info, class_path)
    print(f"  [OK] Sinif isimleri: {class_path}")
    return pipe


def main():
    # Always prefer the native unfiltered v4 aggregated file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agg_path = os.path.join(base_dir, "features_v4.csv")
    raw_path = os.path.join(base_dir, "features.csv")
    
    if os.path.exists(agg_path):
        data_path = agg_path
        print("[OK] Native Exp3a features found. Using features_v4.csv")
    elif os.path.exists(raw_path):
        data_path = raw_path
        print("[WARN] features_v4.csv not found, falling back to features.csv")
    else:
        print("HATA: No feature file found. Run preprocess_fs_v1.py first.")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=TARGET_COLS)
    print(f"[STATS] Toplam ornek: {len(df)}")

    # 3-sinif etiket mapping
    df = apply_label_mapping(df)
    
    # Kullanilabilir featurelari belirle
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing features (skipped): {missing}")
    print(f"  Feature count: {len(available_features)}")

    # Per-user normalizasyon
    df_norm = per_user_normalize(df, available_features)
    X_all = df_norm[available_features].copy()

    # Temporal groups olustur
    df = create_temporal_groups(df)
    groups = df['segment_id']

    # Sinif dagilimlari
    print("\n[STATS] 3-SINIF DAGILIMI:")
    for mc, oc in zip(MAPPED_COLS, TARGET_COLS):
        if mc in df.columns:
            print(f"  {oc:>15} >> {dict(df[mc].value_counts())}")

    # -----------------------------------------------
    # EGITIM: Tum veri seti
    # -----------------------------------------------
    print(f"\n\n{'#'*60}")
    print(f"  EGITIM: TUM VERI SETI ({len(df)} ornek)")
    print(f"  Temporal segments: {df['segment_id'].nunique()}")
    print(f"{'#'*60}")

    best_results = {}
    for mapped_col, orig_col in zip(MAPPED_COLS, TARGET_COLS):
        y = df[mapped_col].dropna()
        X_clean = X_all.loc[y.index]
        g_clean = groups.loc[y.index]
        
        print(f"\n{'='*55}")
        print(f"  TARGET: {orig_col} (3-sinif)")
        print(f"  Sinif Dagilimi: {dict(y.value_counts())}")
        print(f"{'='*55}")
        
        results = evaluate_target(X_clean, y, g_clean, f"{orig_col} (3-sinif)")
        
        # En iyi modeli sec
        best_model = max(results.items(), key=lambda x: x[1][1])  # by balanced_acc
        best_results[orig_col] = (best_model[0], best_model[1][0], best_model[1][1])
        print(f"  >>> En iyi: {best_model[0]} (bal_acc={best_model[1][1]:.4f})")

    # Feature importances
    stress_y = df['stress_val_3cls'].dropna()
    print_feature_importances(X_all.loc[stress_y.index], stress_y, available_features)

    # -----------------------------------------------
    # FINAL MODELLERI KAYDET (RF + GBM)
    # -----------------------------------------------
    y_multi = df[MAPPED_COLS].dropna()
    X_multi = X_all.loc[y_multi.index]
    
    train_and_save(X_multi, y_multi, "final_model_rf", available_features, use_gbm=False)
    train_and_save(X_multi, y_multi, "final_model_gbm", available_features, use_gbm=True)

    # -----------------------------------------------
    # OZET
    # -----------------------------------------------
    print(f"\n\n{'='*60}")
    print("  SONUC OZETI")
    print(f"{'='*60}")
    for target, (model, acc, bal_acc) in best_results.items():
        print(f"  {target:>15}: {model:>20} | acc={acc:.4f} | bal_acc={bal_acc:.4f}")
    
    print("\n[OK] TUM EGITIMLER TAMAMLANDI!")
    print("     RF + GBM modelleri /models klasorune kaydedildi.")


if __name__ == "__main__":
    main()
