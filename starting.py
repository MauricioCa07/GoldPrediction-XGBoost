import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, confusion_matrix
)


HORIZON    = 4      # The number of candlestick that were going take in to account for each prediction
THRESHOLD  = 0.05   # The minimal percentage that were consider as a valid operation


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def add_features(df, prefix=''):
    """
    add  technical features that are calculated from the dataset
    """
    c = df['Close']
    p = prefix

    out = pd.DataFrame(index=df.index) 


    for n in [1, 2, 3, 5, 10, 20]:
        out[f'{p}ret_{n}'] = c.pct_change(n) * 100

    # Vela 
    out[f'{p}range_pct']      = (df['High'] - df['Low']) / c * 100
    out[f'{p}body_pct']       = (c - df['Open']) / c * 100
    out[f'{p}upper_wick_pct'] = (df['High'] - df[['Open','Close']].max(axis=1)) / c * 100
    out[f'{p}lower_wick_pct'] = (df[['Open','Close']].min(axis=1) - df['Low']) / c * 100

    
    for w in [10, 20, 50, 100]:
        sma = c.rolling(w).mean()
        ema = c.ewm(span=w, adjust=False).mean()
        out[f'{p}dist_sma_{w}'] = (c - sma) / sma * 100
        out[f'{p}dist_ema_{w}'] = (c - ema) / ema * 100

     
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    out[f'{p}cross_10_20'] = (sma10 - sma20) / sma20 * 100
    out[f'{p}cross_20_50'] = (sma20 - sma50) / sma50 * 100

  
    out[f'{p}rsi_7']  = calc_rsi(c, 7)
    out[f'{p}rsi_14'] = calc_rsi(c, 14)
    out[f'{p}rsi_28'] = calc_rsi(c, 28)



    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    out[f'{p}macd']      = macd / c * 100
    out[f'{p}macd_sig']  = sig  / c * 100
    out[f'{p}macd_hist'] = (macd - sig) / c * 100

  
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    out[f'{p}bb_width']    = (bb_up - bb_dn) / bb_mid * 100
    out[f'{p}bb_position'] = (c - bb_dn) / (bb_up - bb_dn).replace(0, np.nan)

   
    pc  = c.shift(1)
    tr  = np.maximum(df['High']-df['Low'],
          np.maximum(abs(df['High']-pc), abs(df['Low']-pc)))
    out[f'{p}atr_14'] = tr.rolling(14).mean() / c * 100
    out[f'{p}atr_20'] = tr.rolling(20).mean() / c * 100


    r1 = c.pct_change(1) * 100
    out[f'{p}vol_10']   = r1.rolling(10).std()
    out[f'{p}vol_20']   = r1.rolling(20).std()
    out[f'{p}vol_50']   = r1.rolling(50).std()
    out[f'{p}vol_ratio'] = out[f'{p}vol_10'] / out[f'{p}vol_50'].replace(0, np.nan)

  
    vol_sma = df['Volume'].rolling(20).mean()
    out[f'{p}vol_rel']      = df['Volume'] / vol_sma.replace(0, np.nan)
    out[f'{p}vol_rel_lag1'] = out[f'{p}vol_rel'].shift(1)
    out[f'{p}vol_rel_lag3'] = out[f'{p}vol_rel'].shift(3)


    low14  = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    stoch  = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    out[f'{p}stoch_k'] = stoch
    out[f'{p}stoch_d'] = stoch.rolling(3).mean()


    out[f'{p}mom_10'] = c.pct_change(10) * 100
    out[f'{p}mom_20'] = c.pct_change(20) * 100

    return out


def load_and_prepare(path):
    """
    This function read the datasets and 
    set the correct date format for the dataframe
    """
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df



"""

    Load data stage

"""

df_1h = load_and_prepare('XAU_1h_data.csv')
df_4h = load_and_prepare('XAU_4h_data.csv')
df_1d = load_and_prepare('XAU_1d_data.csv')
print("The dataframes were loaded correctly")



feat_1h = add_features(df_1h, prefix='1h_')
feat_4h = add_features(df_4h, prefix='4h_')
feat_1d = add_features(df_1d, prefix='1d_')
print("The features were added correctly ")


# We take the previous data set and add it to the dataset with the 
# feautures
feat_1h['Date'] = df_1h['Date']
feat_4h['Date'] = df_4h['Date']
feat_1d['Date'] = df_1d['Date']


# I delete the "noise" that is in the dataset 
# for this i doesnt take in to account 
# the movements where the change is lower that 0.05

df_1h['future_return'] = df_1h['Close'].pct_change(HORIZON).shift(-HORIZON) * 100
df_1h['target'] = np.where(
    df_1h['future_return'] >  THRESHOLD, 1,
    np.where(
    df_1h['future_return'] < -THRESHOLD, 0,
    np.nan)  
)

feat_1h['future_return'] = df_1h['future_return']
feat_1h['target']        = df_1h['target']



print("Mergeando timeframes...")

feat_1h = feat_1h.sort_values('Date')
feat_4h = feat_4h.sort_values('Date')
feat_1d = feat_1d.sort_values('Date')

# merge_asof: para cada fila de 1h, toma el último valor de 4h/1d
# que sea ANTERIOR a esa fecha (sin filtrar el futuro)
df_merged = pd.merge_asof(feat_1h, feat_4h, on='Date', direction='backward')
df_merged = pd.merge_asof(df_merged, feat_1d, on='Date', direction='backward')
print(df_merged)

# Features temporales (sobre la fecha de la vela 1h)
df_merged['hour']      = df_merged['Date'].dt.hour
df_merged['dayofweek'] = df_merged['Date'].dt.dayofweek
df_merged['month']     = df_merged['Date'].dt.month

# Eliminar filas sin target (movimientos pequeños) y NaN de rolling
df_merged.dropna(inplace=True)
df_merged.reset_index(drop=True, inplace=True)



# =============================================================================
# 6. DEFINIR FEATURES Y TARGET
# =============================================================================

drop_cols = ['Date', 'target', 'future_return']
feature_cols = [c for c in df_merged.columns if c not in drop_cols]

X = df_merged[feature_cols].values
y = df_merged['target'].values.astype(int)

# =============================================================================
# 7. SPLIT CRONOLÓGICO 80/20
# =============================================================================

split_idx = int(len(X) * 0.9)
x_train, x_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(x_train):,} muestras  →  hasta {df_merged['Date'].iloc[split_idx-1]}")
print(f"Test:  {len(x_test):,} muestras  →  desde {df_merged['Date'].iloc[split_idx]}")

# Balanceo de clases
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos = n_neg / n_pos
print(f"\nBalanceo de clases en train → scale_pos_weight: {scale_pos:.3f}")

# =============================================================================
# 8. ENTRENAMIENTO
# =============================================================================

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=3000,
    alpha=5,
    reg_lambda=5,
    colsample_bytree=0.7,
    subsample=0.8,
    min_child_weight=10,
    scale_pos_weight=scale_pos,
    early_stopping_rounds=75,
    n_jobs=-1,
    random_state=123,
)

print("\nEntrenando...")
model.fit(
    x_train, y_train,
    eval_set=[(x_train, y_train), (x_test, y_test)],
    verbose=200,
)

print(f"\nMejor iteración: {model.best_iteration}")

# =============================================================================
# 9. EVALUACIÓN
# =============================================================================

y_pred_proba = model.predict_proba(x_test)[:, 1]
y_pred       = (y_pred_proba >= 0.5).astype(int)

auc      = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)

print("\n" + "="*55)
print("RESULTADOS FINALES")
print("="*55)
print(f"AUC-ROC:   {auc:.4f}   (>0.53 tiene valor real en trading)")
print(f"Accuracy:  {accuracy*100:.2f}%")
print()
print("Matriz de confusión:")
print(f"              Pred Baja   Pred Sube")
print(f"  Real Baja      {cm[0][0]:>6}      {cm[0][1]:>6}")
print(f"  Real Sube      {cm[1][0]:>6}      {cm[1][1]:>6}")
print()
print(classification_report(y_test, y_pred, target_names=['Baja','Sube']))
print("="*55)

# =============================================================================
# 10. ANÁLISIS POR UMBRAL DE CONFIANZA
#     Solo operar cuando el modelo está muy seguro
# =============================================================================

print("\nPrecisión según umbral de confianza del modelo:")
print(f"{'Umbral':>8} {'Señales':>10} {'% del total':>12} {'Accuracy':>10}")
print("-" * 45)

for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask  = (y_pred_proba >= thresh) | (y_pred_proba <= 1 - thresh)
    n_sig = mask.sum()
    if n_sig == 0:
        continue
    pred_t  = (y_pred_proba[mask] >= thresh).astype(int)
    # para señales de baja, comparar con 0
    pred_th = np.where(y_pred_proba[mask] >= thresh, 1, 0)
    acc_t   = accuracy_score(y_test[mask], pred_th)
    print(f"{thresh:>8.2f} {n_sig:>10,} {n_sig/len(y_test)*100:>11.1f}%"
          f" {acc_t*100:>9.2f}%")

# =============================================================================
# 11. IMPORTANCIA DE FEATURES (top 25)
# =============================================================================

importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)

print(f"\nTop 25 features más importantes:")
print(importance.head(25).to_string())

print("\n--- Features por timeframe ---")
for pref in ['1h_', '4h_', '1d_']:
    cols = [c for c in feature_cols if c.startswith(pref)]
    imp  = importance[cols].sum()
    print(f"  {pref}  →  importancia total: {imp:.4f}  ({imp*100:.1f}%)")