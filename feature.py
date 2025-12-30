import pandas as pd
from thefuzz import process
import warnings

# Ignoriamo i warning per pulizia
warnings.filterwarnings('ignore')

print("--- AVVIO INTEGRAZIONE FBREF & TRANSFERMARKT ---")

# ==============================================================================
# 1. CARICAMENTO DATI FBREF
# ==============================================================================
print("1. Caricamento dati FBref...")
try:
    df_lineups = pd.read_csv('data/fbref_lineups.csv') 
    df_stats = pd.read_csv('data/fbref_match_stats.csv') 

    # --- CORREZIONE FONDAMENTALE: CREAZIONE DELLA DATA ---
    print("   🛠️  Estraggo la data dalla colonna 'game'...")
    
    # Se la colonna 'game' esiste, estraiamo i primi 10 caratteri (YYYY-MM-DD)
    if 'game' in df_lineups.columns:
        df_lineups['date'] = pd.to_datetime(df_lineups['game'].str[:10])
    else:
        print("❌ ERRORE: Nel file fbref_lineups.csv manca la colonna 'game'!")
        exit()
        
    # Facciamo lo stesso per le statistiche
    if 'game' in df_stats.columns:
         df_stats['date'] = pd.to_datetime(df_stats['game'].str[:10])

except FileNotFoundError:
    print("❌ ERRORE: Non trovo i file fbref nella cartella 'data/'.")
    exit()

# ==============================================================================
# 2. CARICAMENTO DATI KAGGLE (SOLDI)
# ==============================================================================
print("2. Caricamento dati Valori (Kaggle)...")
try:
    # Carichiamo solo quello che serve
    df_k_players = pd.read_csv('data/players.csv', usecols=['player_id', 'name', 'last_season'])
    df_k_vals = pd.read_csv('data/player_valuations.csv')
    
    # Convertiamo la data in formato data
    df_k_vals['date'] = pd.to_datetime(df_k_vals['date'])
    
except FileNotFoundError:
    print("❌ ERRORE: Non trovo i file Kaggle (players.csv, player_valuations.csv).")
    exit()

# ==============================================================================
# 3. CREAZIONE DIZIONARIO NOMI (FUZZY MATCHING)
# ==============================================================================
print("3. Creazione mappa nomi (questo potrebbe richiedere 1-2 minuti)...")

# Ottimizzazione: Filtriamo solo giocatori recenti di Kaggle per velocizzare
recent_players = df_k_players[df_k_players['last_season'] >= 2017]
kaggle_names = recent_players['name'].unique()

# Nomi da FBref
fbref_names = df_lineups['player'].dropna().unique()

print(f"   Devo mappare {len(fbref_names)} giocatori...")

name_mapping = {}
# Creiamo dizionario per match esatto veloce (lowercase)
kaggle_names_clean = {str(n).lower(): n for n in kaggle_names}

for name in fbref_names:
    n_str = str(name)
    n_clean = n_str.lower()
    
    # 1. Match Esatto
    if n_clean in kaggle_names_clean:
        name_mapping[name] = kaggle_names_clean[n_clean]
    else:
        # 2. Match Fuzzy (Simile)
        match, score = process.extractOne(n_str, kaggle_names)
        if score >= 85: 
            name_mapping[name] = match
        else:
            name_mapping[name] = None 

print(f"   Mappati {len([k for k,v in name_mapping.items() if v])} su {len(fbref_names)} giocatori.")

# ==============================================================================
# 4. APPLICAZIONE VALORI E CALCOLO
# ==============================================================================
print("4. Calcolo valore formazioni...")

# Uniamo i nomi corretti
df_lineups['kaggle_name'] = df_lineups['player'].map(name_mapping)
df_lineups_matched = df_lineups.dropna(subset=['kaggle_name'])

# Recuperiamo l'ID giocatore Kaggle
df_lineups_matched = df_lineups_matched.merge(recent_players[['name', 'player_id']], left_on='kaggle_name', right_on='name', how='left')

# Ordiniamo per il merge asof
df_k_vals = df_k_vals.sort_values('date')
df_lineups_matched = df_lineups_matched.sort_values('date')

# Colleghiamo il valore (Soldi) alla partita (Data)
df_valued = pd.merge_asof(
    df_lineups_matched,
    df_k_vals[['player_id', 'date', 'market_value_in_eur']],
    on='date',
    by='player_id',
    direction='backward'
)

# Filtriamo solo i TITOLARI (is_starter = True)
starters = df_valued[df_valued['is_starter'] == True]

# --- QUI C'ERA L'ERRORE PRIMA: USIAMO 'game' INVECE DI 'game_id' ---
lineup_values = starters.groupby(['game', 'team'])['market_value_in_eur'].sum().reset_index()
lineup_values.rename(columns={'market_value_in_eur': 'Starting_XI_Value'}, inplace=True)

# ==============================================================================
# 5. MERGE FINALE E FEATURE ENGINEERING
# ==============================================================================
print("5. Creazione dataset finale...")

# Uniamo usando 'game' e 'team'
final_df = df_stats.merge(lineup_values, on=['game', 'team'], how='left')

# Creiamo l'anno della stagione
final_df['Season_Year'] = final_df['date'].apply(lambda x: x.year if x.month > 7 else x.year - 1)

# Calcoliamo la mediana stagionale ("Valore Solito")
final_df['Typical_XI_Value'] = final_df.groupby(['team', 'Season_Year'])['Starting_XI_Value'].transform('median')

# Feature: Ratio (Valore Oggi / Valore Solito)
final_df['Lineup_Strength_Ratio'] = final_df['Starting_XI_Value'] / final_df['Typical_XI_Value']

# Pulizia finale
final_df = final_df.dropna(subset=['Starting_XI_Value']) # Rimuove righe senza valori
cols_to_keep = ['date', 'game', 'team', 'opponent', 'result', 'xG', 'Starting_XI_Value', 'Lineup_Strength_Ratio']
final_output = final_df[[c for c in cols_to_keep if c in final_df.columns]]

print("-" * 30)
print(f"✅ FATTO! Dataset creato con {len(final_output)} partite.")
print(final_output.head())

final_output.to_csv('data/dataset_completo_xgboost.csv', index=False)