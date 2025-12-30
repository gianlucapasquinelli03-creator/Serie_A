import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("--- AGGIUNTA QUOTE (ODDS) AL DATASET ---")

def standardize_names(name):
    """
    Standardizza i nomi per far combaciare Odds e Dataset.
    Aggiungi qui eventuali eccezioni se vedi che perdi troppe righe.
    """
    if pd.isna(name): return name
    name = str(name).lower().strip()
    
    mapping = {
        'inter': 'inter', 'internazionale': 'inter', 'fc internazionale': 'inter', 'inter milan': 'inter',
        'milan': 'milan', 'ac milan': 'milan',
        'juventus': 'juventus', 'juve': 'juventus',
        'roma': 'roma', 'as roma': 'roma',
        'napoli': 'napoli', 'ssc napoli': 'napoli',
        'lazio': 'lazio', 'ss lazio': 'lazio',
        'atalanta': 'atalanta',
        'fiorentina': 'fiorentina',
        'torino': 'torino',
        'bologna': 'bologna',
        'verona': 'verona', 'hellas verona': 'verona',
        'lecce': 'lecce',
        'udinese': 'udinese',
        'sassuolo': 'sassuolo',
        'monza': 'monza',
        'empoli': 'empoli',
        'salernitana': 'salernitana',
        'frosinone': 'frosinone',
        'genoa': 'genoa',
        'cagliari': 'cagliari',
        'spal': 'spal',
        'crotone': 'crotone',
        'benevento': 'benevento',
        'spezia': 'spezia',
        'sampdoria': 'sampdoria',
        'venezia': 'venezia',
        'parma': 'parma',
        'brescia': 'brescia',
        'manchester city': 'man city', 'man city': 'man city',
        'manchester united': 'man utd', 'man utd': 'man utd'
    }
    return mapping.get(name, name)

# 1. CARICAMENTO DATI
try:
    # Il tuo dataset principale
    df = pd.read_csv('data/dataset_ultimate.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Il file delle quote
    odds = pd.read_csv('data/odds_history.csv')
    odds['date'] = pd.to_datetime(odds['date'])
    
    print(f"✅ File caricati.\nDataset righe: {len(df)}\nOdds righe: {len(odds)}")
    
except FileNotFoundError:
    print("❌ Errore: Mancano i file (dataset_ultimate.csv o odds_history.csv).")
    exit()

# 2. SELEZIONE COLONNE QUOTE (Bet365 o Media)
# Cerchiamo le colonne B365H (Home), B365D (Draw), B365A (Away)
# Se non ci sono, cerchiamo AvgH, AvgD, AvgA
print("🔍 Ricerca colonne quote migliori...")

cols_b365 = ['B365H', 'B365D', 'B365A']
cols_avg = ['AvgH', 'AvgD', 'AvgA']
cols_bw = ['BW H', 'BW D', 'BW A'] # A volte soccerdata le chiama così

selected_cols = []
source_label = ""

# Controllo quali colonne esistono nel file odds
odds_columns = odds.columns.tolist()

if all(c in odds_columns for c in cols_b365):
    selected_cols = cols_b365
    source_label = "Bet365"
elif all(c in odds_columns for c in cols_avg):
    selected_cols = cols_avg
    source_label = "Media"
else:
    # Fallback brutale: prende le prime 3 colonne numeriche dopo le date/nomi
    # (Da adattare se il tuo file odds è strano)
    print("⚠️ Attenzione: Colonne standard non trovate. Provo a cercare colonne simili...")
    # Qui dovresti controllare il tuo CSV. Per ora proviamo a vedere se esistono colonne 'H', 'D', 'A' generiche
    possible_h = [c for c in odds_columns if c.endswith('H') and 'AH' not in c] # Esclude Asian Handicap
    if possible_h:
        base = possible_h[0][:-1] # Prende il prefisso (es 'B365')
        selected_cols = [base+'H', base+'D', base+'A']
        source_label = f"Generico ({base})"
    else:
        print("❌ Impossibile trovare colonne quote (H/D/A). Controlla il CSV odds_history!")
        exit()

print(f"   Usando quote fonte: {source_label} -> {selected_cols}")

# 3. TRASFORMAZIONE QUOTE: DA MATCH-CENTRIC A TEAM-CENTRIC
print("🔄 Riorganizzazione quote per squadra...")

# Standardizziamo i nomi nel file odds
odds['home_team_std'] = odds['home_team'].apply(standardize_names)
odds['away_team_std'] = odds['away_team'].apply(standardize_names)

# Creiamo due dataset temporanei: uno per la squadra di casa, uno per l'ospite

# --- LATO CASA ---
odds_home_view = odds[['date', 'home_team_std', selected_cols[0], selected_cols[1], selected_cols[2]]].copy()
odds_home_view.columns = ['date', 'team_key', 'Odds_Win', 'Odds_Draw', 'Odds_Lose'] 
# Per chi gioca in casa: Win=H, Draw=D, Lose=A

# --- LATO TRASFERTA ---
odds_away_view = odds[['date', 'away_team_std', selected_cols[2], selected_cols[1], selected_cols[0]]].copy()
odds_away_view.columns = ['date', 'team_key', 'Odds_Win', 'Odds_Draw', 'Odds_Lose']
# Per chi gioca fuori: Win=A, Draw=D, Lose=H (Nota l'inversione H e A!)

# Uniamo tutto in un unico "listone" di quote per squadra
odds_long = pd.concat([odds_home_view, odds_away_view], ignore_index=True)

# Rimuoviamo eventuali duplicati (se ci sono quote doppie per errore)
odds_long = odds_long.drop_duplicates(subset=['date', 'team_key'])

# 4. MERGE FINALE
print("🔗 Unione al dataset principale...")

# Standardizziamo i nomi nel dataset principale per il match
df['team_key'] = df['team'].apply(standardize_names)

# Uniamo su Data e Squadra
# Usa 'left' per non perdere partite se mancano le quote (avranno NaN)
df_final = df.merge(odds_long, on=['date', 'team_key'], how='left')

# Pulizia
df_final.drop(columns=['team_key'], inplace=True)

# 5. CONTROLLO E SALVATAGGIO
missing_odds = df_final['Odds_Win'].isna().sum()
total_rows = len(df_final)
print(f"📊 Report Merge: Quote trovate per {total_rows - missing_odds} righe su {total_rows}.")

if missing_odds > 0:
    print(f"⚠️ Mancano quote per {missing_odds} righe. Potrebbe essere colpa dei nomi squadre diversi.")
    print("   Esempio squadre senza quote:")
    print(df_final[df_final['Odds_Win'].isna()]['team'].unique()[:10])

# Riempiamo i NaN (opzionale, per XGBoost meglio lasciare NaN o mettere -1, ma per ora lasciamo così)
# df_final = df_final.dropna(subset=['Odds_Win']) # Scommenta se vuoi cancellare chi non ha quote

df_final.to_csv('data/dataset_con_quote.csv', index=False)
print("🚀 SALVATO: 'data/dataset_con_quote.csv'")
print(df_final[['date', 'team', 'opponent', 'Odds_Win', 'Odds_Draw', 'Odds_Lose']].head())