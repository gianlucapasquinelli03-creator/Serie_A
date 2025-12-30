import pandas as pd

print("--- PULIZIA E PREPARAZIONE FINALE (1X2) ---")

# 1. CARICA IL DATASET CON QUOTE (quello FIXED)
try:
    # Usa il file generato dallo script di debug/fix
    df = pd.read_csv('data/dataset_con_quote_FIXED.csv')
    print(f"1. Righe Totali Iniziali: {len(df)}")
except FileNotFoundError:
    print("❌ Errore: Manca 'data/dataset_con_quote_FIXED.csv'")
    exit()

# 2. RIMUOVI PARTITE SENZA QUOTE
# Se manca Odds_1, mancano tutte.
initial_count = len(df)
df_clean = df.dropna(subset=['Odds_1']).copy()
removed = initial_count - len(df_clean)

print(f"2. Pulizia Quote: Rimosse {removed} righe senza quote.")
print(f"   Righe valide rimaste: {len(df_clean)}")

# 3. TRASFORMAZIONE IN RIGA SINGOLA (MATCH-CENTRIC)
# Teniamo solo le righe 'is_home' == 1.
# Poiché abbiamo le righe doppie, ogni partita ha una riga Home e una Away.
# Prendendo solo Home, abbiamo una riga per partita con tutti i dati.

df_single = df_clean[df_clean['is_home'] == 1].copy()

print(f"3. Trasformazione: Da {len(df_clean)} righe doppie a {len(df_single)} partite uniche.")

# 4. RINOMINA COLONNE (Per chiarezza Input Casa vs Fuori)
rename_map = {
    'team': 'Home_Team',
    'opponent': 'Away_Team',
    'Starting_XI_Value': 'Home_Value',
    'Opponent_Value': 'Away_Value',
    'Lineup_Strength_Ratio': 'Home_Lineup_Ratio',
    'Opponent_Lineup_Ratio': 'Away_Lineup_Ratio',
    'xG_Relative_Form': 'Home_Attack_Form',
    'Defense_Form_Relative': 'Home_Defense_Form',
    'Opponent_Attack_Form': 'Away_Attack_Form',
    'Opponent_Defense_Form': 'Away_Defense_Form',
    # Le quote su riga Home sono già corrette (Odds_1 = Vittoria Casa)
}

df_single.rename(columns=rename_map, inplace=True)

# 5. CREAZIONE TARGET (0, 1, 2)
# Convertiamo W/D/L in 0/1/2
target_map = {'W': 0, 'D': 1, 'L': 2}
df_single['Target'] = df_single['result'].map(target_map)

# 6. SELEZIONE FINALE COLONNE
cols_order = [
    'date', 'Home_Team', 'Away_Team', 'Target',          # Info Base
    'Odds_1', 'Odds_X', 'Odds_2',                        # Quote
    'Value_Ratio_vs_Opponent',                           # Feature Regina
    'Home_Value', 'Away_Value',                          # Dati valore
    'Home_Lineup_Ratio', 'Away_Lineup_Ratio',            # Dati assenze
    'Home_Attack_Form', 'Home_Defense_Form',             # Dati forma Casa
    'Away_Attack_Form', 'Away_Defense_Form'              # Dati forma Trasferta
]

# Filtriamo solo le colonne che esistono davvero
final_cols = [c for c in cols_order if c in df_single.columns]
df_final = df_single[final_cols]

# 7. SALVATAGGIO
print("-" * 30)
print(df_final.head())
print("-" * 30)

df_final.to_csv('data/dataset_train_final.csv', index=False)
print("🚀 TUTTO PRONTO! File salvato: 'data/dataset_train_final.csv'")