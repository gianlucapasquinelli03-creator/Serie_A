import pandas as pd

# 1. Carica il dataset completo (quello con le quote)
try:
    df = pd.read_csv('data/dataset_con_quote_FIXED.csv')
    print(f"Righe totali (Doppie): {len(df)}")
except FileNotFoundError:
    print("❌ Esegui prima lo script delle quote!")
    exit()

# 2. FILTRO: TENIAMO SOLO LA PROSPETTIVA DELLA SQUADRA DI CASA
# Se is_home == 1, quella riga contiene già TUTTO: i dati della casa e i dati dell'avversario (ospite)
df_single = df[df['is_home'] == 1].copy()

# 3. RINOMINA COLONNE (Per non confondersi)
# Ora "team" diventa "Home_Team" e "opponent" diventa "Away_Team"
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
    # Quote
    'Odds_Win': 'Odds_1',   # La quota vittoria per chi è in casa è l'1
    'Odds_Draw': 'Odds_X',
    'Odds_Lose': 'Odds_2'   # La quota sconfitta per chi è in casa è il 2
}

df_single.rename(columns=rename_map, inplace=True)

# 4. CREAZIONE TARGET (0, 1, 2)
# XGBoost vuole numeri interi che partono da 0
# 0 = Vittoria Casa (1)
# 1 = Pareggio (X)
# 2 = Vittoria Ospite (2)

target_map = {
    'W': 0,  # Home ha Vinto
    'D': 1,  # Home ha Pareggiato
    'L': 2   # Home ha Perso (quindi Ospite ha vinto)
}

df_single['Target'] = df_single['result'].map(target_map)

# Pulizia: Rimuoviamo colonne inutili ora
cols_to_drop = ['is_home', 'matchweek', 'result'] # matchweek la teniamo se vuoi, is_home è sempre 1
df_single.drop(columns=['is_home', 'result'], inplace=True)

# 5. ORDINE COLONNE (Per pulizia visiva)
cols_order = [
    'date', 'Home_Team', 'Away_Team', 'Target',
    'Odds_1', 'Odds_X', 'Odds_2',
    'Value_Ratio_vs_Opponent',
    'Home_Value', 'Away_Value',
    'Home_Lineup_Ratio', 'Away_Lineup_Ratio',
    'Home_Attack_Form', 'Home_Defense_Form',
    'Away_Attack_Form', 'Away_Defense_Form'
]

# Selezioniamo solo le colonne esistenti
final_cols = [c for c in cols_order if c in df_single.columns]
df_final = df_single[final_cols]

print(f"Righe finali (Singole): {len(df_final)}")
print("-" * 30)
print("Esempio Target (0=1, 1=X, 2=2):")
print(df_final[['date', 'Home_Team', 'Away_Team', 'Target', 'Odds_1']].head())

# Salva
df_final.to_csv('data/dataset_1x2_ready.csv', index=False)
print("🚀 SALVATO: 'data/dataset_1x2_ready.csv'")