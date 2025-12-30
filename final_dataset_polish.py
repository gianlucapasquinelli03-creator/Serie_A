import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("--- RAFFINAMENTO FINALE DATASET ---")

# 1. CARICAMENTO DATASET
try:
    # Carichiamo il dataset che hai appena creato
    df = pd.read_csv('data/dataset_xgboost_ready.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Carichiamo anche il calendario per sapere con certezza chi è in casa
    schedule = pd.read_csv('data/fbref_schedule.csv')
    # Puliamo il calendario per avere solo game e home_team
    # Nota: Assumiamo che 'home_team' sia il nome della squadra in casa
    schedule = schedule[['game', 'home_team']]
    
    # Carichiamo le stats originali per recuperare xGA (Difesa) se manca
    stats_raw = pd.read_csv('data/fbref_match_stats.csv')
    if 'xGA' in stats_raw.columns:
        # Se c'è xGA lo usiamo, altrimenti useremo i Gol Subiti o l'xG avversario dopo
        # Uniamo xGA al dataset principale
        # Attenzione: stats_raw potrebbe non avere 'game', usiamo la logica di prima se serve
        if 'game' in stats_raw.columns:
            df = df.merge(stats_raw[['game', 'team', 'xGA']], on=['game', 'team'], how='left')
    
    print(f"✅ Dati caricati. Righe iniziali: {len(df)}")

except FileNotFoundError:
    print("❌ Errore: Mancano i file. Assicurati di aver eseguito gli script precedenti.")
    exit()

# 2. FEATURE: CASA / TRASFERTA (Is_Home)
print("1. Assegnazione Casa/Trasferta...")
# Uniamo col calendario per vedere chi è la home_team
df = df.merge(schedule, on='game', how='left')

# Standardizziamo i nomi anche qui per confronto sicuro
from thefuzz import process # Usiamo la logica semplice se i nomi sono puliti, ma meglio essere sicuri
# Per semplicità, assumiamo che se il nome nel dataset contiene il nome home_team, è in casa
# Oppure, logica diretta:
df['is_home'] = df.apply(lambda x: 1 if str(x['team']) in str(x['home_team']) else 0, axis=1)
# Rimuoviamo la colonna di appoggio
df.drop(columns=['home_team'], inplace=True)

# 3. FEATURE: GIORNATA (Matchweek)
print("2. Calcolo Giornata (Matchweek)...")
df = df.sort_values(['team', 'date'])
# Conta progressiva delle partite per squadra in ogni stagione
# Recuperiamo Season_Year se manca
if 'Season_Year' not in df.columns:
    df['Season_Year'] = df['date'].apply(lambda x: x.year if x.month > 7 else x.year - 1)

df['matchweek'] = df.groupby(['Season_Year', 'team']).cumcount() + 1

# 4. FEATURE DIFENSIVA (xGA Rolling Relative)
print("3. Calcolo Feature Difensiva (xGA Form)...")
# Se xGA non c'è (dipende dai dati scaricati), usiamo una logica di fallback o skip
if 'xGA' in df.columns:
    # Media mobile xGA ultime 5 partite
    df['xGA_Rolling'] = df.groupby('team')['xGA'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Statistiche Campionato per xGA
    league_xga = df.groupby('Season_Year')['xGA'].agg(['mean', 'std']).reset_index()
    league_xga.rename(columns={'mean': 'L_xGA_Mean', 'std': 'L_xGA_Std'}, inplace=True)
    df = df.merge(league_xga, on='Season_Year', how='left')
    
    # Z-Score Difesa (Più è basso, meglio è la difesa)
    df['Defense_Form_Relative'] = (df['xGA_Rolling'] - df['L_xGA_Mean']) / df['L_xGA_Std']
    df['Defense_Form_Relative'] = df['Defense_Form_Relative'].fillna(0)
else:
    print("⚠️ Attenzione: Colonna 'xGA' non trovata. Salto feature difensiva avanzata.")
    df['Defense_Form_Relative'] = 0

# 5. RECUPERO DATI AVVERSARIO (Simmetria)
print("4. Recupero statistiche Avversario...")

# Creiamo un dataset "ombra" con le info che vogliamo dell'avversario
cols_to_clone = ['game', 'team', 'Lineup_Strength_Ratio', 'xG_Relative_Form', 'Defense_Form_Relative']
df_opp = df[cols_to_clone].copy()
df_opp.rename(columns={
    'team': 'opponent_name',
    'Lineup_Strength_Ratio': 'Opponent_Lineup_Ratio',
    'xG_Relative_Form': 'Opponent_Attack_Form',
    'Defense_Form_Relative': 'Opponent_Defense_Form'
}, inplace=True)

# Uniamo al dataset originale
df_final = df.merge(df_opp, left_on=['game', 'opponent'], right_on=['game', 'opponent_name'], how='left')

# 6. PULIZIA FINALE
print("5. Pulizia e Salvataggio...")
df_final = df_final.drop_duplicates(subset=['game', 'team'])

# Gestione NaN (Le prime giornate avranno NaN sulle medie mobili)
# Sostituiamo con 0 (media del campionato)
cols_nan = ['xG_Relative_Form', 'Defense_Form_Relative', 'Opponent_Attack_Form', 'Opponent_Defense_Form']
df_final[cols_nan] = df_final[cols_nan].fillna(0)

# Lista colonne finale ordinata
final_cols = [
    'date', 'matchweek', 'is_home',         # Info contesto
    'team', 'opponent', 'result',           # Info partita
    'Starting_XI_Value', 'Opponent_Value',  # Info soldi (grezzi)
    'Value_Ratio_vs_Opponent',              # Feature 1: Chi è più ricco
    'Lineup_Strength_Ratio',                # Feature 2: Assenze MIE
    'Opponent_Lineup_Ratio',                # Feature 3: Assenze AVVERSARIO
    'xG_Relative_Form',                     # Feature 4: Mio Attacco
    'Defense_Form_Relative',                # Feature 5: Mia Difesa
    'Opponent_Attack_Form',                 # Feature 6: Attacco Avversario
    'Opponent_Defense_Form'                 # Feature 7: Difesa Avversario
]

# Filtra solo colonne esistenti (per sicurezza)
final_cols = [c for c in final_cols if c in df_final.columns]
df_ready = df_final[final_cols]

print("-" * 30)
print(df_ready.head())
print("-" * 30)

df_ready.to_csv('data/dataset_ultimate.csv', index=False)
print("🚀 File salvato come: data/dataset_ultimate.csv")