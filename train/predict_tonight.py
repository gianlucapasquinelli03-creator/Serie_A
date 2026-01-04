import pandas as pd
import xgboost as xgb
import numpy as np
import joblib

# --- CONFIGURAZIONE: INSERISCI QUI LE PARTITE DI STASERA ---
# Formato: ("Squadra_Casa", "Squadra_Ospite")
matches_tonight = [
    ("Atalanta", "Roma")    # Esempio
    # Aggiungi qui le altre...
]
# -----------------------------------------------------------

print("--- 🔮 PREVISIONI LIVE SERIE A ---")

# 1. CARICAMENTO MODELLO E DATI STORICI
try:
    # Carichiamo il modello salvato
    model = joblib.load('train/modello_serie_a.pkl')
    
    # Carichiamo lo storico per recuperare la "Forma" attuale delle squadre
    # Usiamo il file finale che contiene già Attack_Form, Defense_Form, ecc.
    df_history = pd.read_csv("data/dataset_train_final_3.csv")
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history = df_history.sort_values('date') # Ordiniamo per data
    
    print("✅ Modello e Storico caricati.")
except Exception as e:
    print(f"❌ Errore: {e}")
    print("Assicurati di aver salvato il modello ('modello_serie_a.json') e di avere il dataset.")
    exit()

# Funzione per estrarre l'ultima forma nota di una squadra
def get_latest_stats(team_name, df):
    # Cerchiamo le partite dove la squadra ha giocato (Casa o Fuori)
    last_match = df[(df['Home_Team'] == team_name) | (df['Away_Team'] == team_name)].tail(1)
    
    if last_match.empty:
        return None
    
    row = last_match.iloc[0]
    stats = {}
    
    if row['Home_Team'] == team_name:
        stats['Value'] = row['Home_Value']
        stats['Lineup_Ratio'] = row['Home_Lineup_Ratio']
        stats['Attack_Form'] = row['Home_Attack_Form']
        stats['Defense_Form'] = row['Home_Defense_Form']
    else: # Away Team
        stats['Value'] = row['Away_Value']
        stats['Lineup_Ratio'] = row['Away_Lineup_Ratio']
        stats['Attack_Form'] = row['Away_Attack_Form']
        stats['Defense_Form'] = row['Away_Defense_Form']
        
    return stats

# 2. GENERAZIONE PREVISIONI
results = []

print(f"\nAnalisi di {len(matches_tonight)} partite...\n")

for home, away in matches_tonight:
    # Recupera le statistiche più recenti
    stats_home = get_latest_stats(home, df_history)
    stats_away = get_latest_stats(away, df_history)
    
    if not stats_home or not stats_away:
        print(f"⚠️ Dati mancanti per {home} o {away}. Salto la partita.")
        continue
        
    # Costruzione della riga di Input (Feature Engineering al volo)
    # Calcoliamo le feature "relative" come fatto nel training
    
    # 1. Value Ratio (Valore Casa / Valore Ospite)
    # Aggiungiamo 1 per evitare divisioni per zero
    val_ratio = stats_home['Value'] / (stats_away['Value'] + 1)

    # 2. Calcolo le NUOVE feature (Attacco vs Difesa avversaria)
    # È fondamentale calcolarle perché il modello le aspetta!
    home_att_vs_def = stats_home['Attack_Form'] - stats_away['Defense_Form']
    away_att_vs_def = stats_away['Attack_Form'] - stats_home['Defense_Form']
    
    # Creiamo il DataFrame con le colonne ESATTE usate nel training
    # L'ordine deve essere lo stesso di 'features_to_drop' escluse nel notebook
    input_data = pd.DataFrame([{
        'Value_Ratio_vs_Opponent': val_ratio,
        'Home_Value': stats_home['Value'],
        'Away_Value': stats_away['Value'],
        'Home_Lineup_Ratio': stats_home['Lineup_Ratio'],
        'Away_Lineup_Ratio': stats_away['Lineup_Ratio'],
        'Home_Attack_Form': stats_home['Attack_Form'],
        'Home_Defense_Form': stats_home['Defense_Form'],
        'Away_Attack_Form': stats_away['Attack_Form'],
        'Away_Defense_Form': stats_away['Defense_Form'],
        'Home_Attack_vs_Def': home_att_vs_def,
        'Away_Attack_vs_Def': away_att_vs_def
    }])
    
    # Predizione
    probs = model.predict_proba(input_data)[0] # [Prob_1, Prob_X, Prob_2]
    
    # Formattazione Output
    print(f"⚽ {home} vs {away}")
    print(f"   📊 Probabilità: 1 [{probs[0]:.0%}] - X [{probs[1]:.0%}] - 2 [{probs[2]:.0%}]")
    