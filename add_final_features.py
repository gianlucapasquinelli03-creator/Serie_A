import pandas as pd
import numpy as np

# Ignora warning
import warnings
warnings.filterwarnings('ignore')

print("--- CREAZIONE FEATURE FINALI (Value Ratio & xG Relative) ---")

# 1. CARICA IL DATASET PRECEDENTE
try:
    df = pd.read_csv('data/dataset_completo_xgboost.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # --- FIX: RICALCOLIAMO LA STAGIONE (Season_Year) CHE MANCAVA ---
    print("   🛠️  Rigenero colonna Season_Year...")
    df['Season_Year'] = df['date'].apply(lambda x: x.year if x.month > 7 else x.year - 1)
    # ----------------------------------------------------------------
    
    print(f"✅ Dataset caricato: {len(df)} righe.")
except FileNotFoundError:
    print("❌ Errore: Esegui prima lo script precedente per creare 'dataset_completo_xgboost.csv'")
    exit()
except KeyError as e:
    print(f"❌ Errore Chiave: {e} - Controlla che le colonne nel CSV siano corrette.")
    exit()

# ==============================================================================
# FEATURE 1: VALUE STRENGTH (Rapporto Squadra vs Avversario)
# ==============================================================================
print("1. Calcolo Value Ratio (Squadra vs Avversario)...")

# Per calcolare questo, dobbiamo incrociare la partita con se stessa.
# Ogni partita ha un ID 'game' univoco (es. "2023-08-20 Inter-Monza").
# Nel dataset hai due righe per ogni game: una per l'Inter, una per il Monza.

# Creiamo una copia del dataset che contiene solo le info dell'AVVERSARIO
df_opponents = df[['game', 'team', 'Starting_XI_Value']].copy()
df_opponents.rename(columns={
    'team': 'opponent_name', 
    'Starting_XI_Value': 'Opponent_Value'
}, inplace=True)

# Uniamo il dataset originale con quello degli avversari
# La chiave è 'game', ma dobbiamo assicurarci di non unire la squadra con se stessa
# Trucco: Uniamo su 'game' e poi filtriamo dove team != opponent_name
df_merged = df.merge(df_opponents, on='game', how='left')

# Filtriamo via le righe dove la squadra si è unita con se stessa (Inter vs Inter)
df_final = df_merged[df_merged['team'] != df_merged['opponent_name']].copy()

# Rimuoviamo duplicati creati dal merge (tieni la riga corretta)
# A volte il merge crea righe doppie se i dati sono sporchi, puliamo per sicurezza
df_final = df_final.drop_duplicates(subset=['game', 'team'])

# --- CALCOLO MATEMATICO ---
# Ratio: Se > 1 la mia squadra vale più dell'avversario
# Aggiungiamo un piccolo valore (+1) per evitare divisioni per zero se i dati mancano
df_final['Value_Ratio_vs_Opponent'] = df_final['Starting_XI_Value'] / (df_final['Opponent_Value'] + 1)

# ==============================================================================
# FEATURE 2: xG RELATIVE (Forma rispetto al campionato)
# ==============================================================================
print("2. Calcolo xG Relative (Media Mobile vs Campionato)...")

# ATTENZIONE: Per un modello predittivo, non possiamo usare l'xG della partita di OGGI
# per predire la partita di OGGI. Dobbiamo usare la media delle ULTIME 5 PARTITE.

# 1. Calcoliamo la media mobile degli xG (Forma recente)
df_final = df_final.sort_values(['team', 'date'])
# Calcola la media degli xG delle ultime 5 partite (shiftata di 1 per non includere oggi)
df_final['xG_Rolling_Mean'] = df_final.groupby('team')['xG'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)

# 2. Calcoliamo la media e deviazione standard del CAMPIONATO per quella stagione
# Raggruppiamo per Stagione (Season_Year)
# Nota: L'xG medio del campionato cambia anno per anno
league_stats = df_final.groupby('Season_Year')['xG'].agg(['mean', 'std']).reset_index()
league_stats.rename(columns={'mean': 'League_xG_Mean', 'std': 'League_xG_Std'}, inplace=True)

# Uniamo queste statistiche al dataset
df_final = df_final.merge(league_stats, on='Season_Year', how='left')

# 3. Z-Score (La tua feature "xG Relative")
# Formula: (Mio xG Medio - Media Campionato) / Deviazione Standard Campionato
# Interpretazione: 
# +1.0 = Attacco decisamente sopra la media
# 0.0 = Attacco nella media
# -1.0 = Attacco scarso
df_final['xG_Relative_Form'] = (df_final['xG_Rolling_Mean'] - df_final['League_xG_Mean']) / df_final['League_xG_Std']

# ==============================================================================
# PULIZIA E SALVATAGGIO
# ==============================================================================
# Riempiamo i NaN (es. prime partite della stagione dove non c'è media mobile) con 0
df_final['xG_Relative_Form'] = df_final['xG_Relative_Form'].fillna(0)

# Selezioniamo le colonne finali per XGBoost
cols_to_keep = [
    'date', 'game', 'team', 'opponent', 'result', 
    'Starting_XI_Value', 'Opponent_Value',        # Dati grezzi (per controllo)
    'Lineup_Strength_Ratio',                      # Feature 1: Assenze (Inter oggi vs Inter solita)
    'Value_Ratio_vs_Opponent',                    # Feature 2: Forza (Inter vs Empoli)
    'xG_Relative_Form'                            # Feature 3: Attacco (Inter vs Media Serie A)
]

# Filtra solo colonne esistenti
output_df = df_final[[c for c in cols_to_keep if c in df_final.columns]]

print("-" * 30)
print("✅ CALCOLO COMPLETATO")
print(output_df.tail()) # Mostra le ultime righe
print("-" * 30)

output_df.to_csv('data/dataset_xgboost_ready.csv', index=False)
print("📁 File salvato come: data/dataset_xgboost_ready.csv")