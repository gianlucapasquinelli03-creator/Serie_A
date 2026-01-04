import pandas as pd
import warnings

warnings.filterwarnings('ignore')

print("--- DEBUG E RIPARAZIONE QUOTE ---")

# 1. CARICAMENTO
try:
    df = pd.read_csv('data/dataset_ultimate_3.csv') # Il tuo dataset senza quote
    odds = pd.read_csv('data/odds_history.csv')   # Il file delle quote
    print(f"✅ File caricati.\nDataset: {len(df)} righe\nOdds: {len(odds)} righe")
except FileNotFoundError:
    print("❌ Errore: File non trovati.")
    exit()

# 2. CONTROLLO NOMI COLONNE (Il problema potrebbe essere qui)
print("\n🔍 --- NOMI COLONNE NEL FILE ODDS ---")
print(odds.columns.tolist())

# Cerchiamo colonne che sembrano quote (H, D, A)
potential_cols = [c for c in odds.columns if c.endswith('H') and len(c) < 7]
print(f"👉 Colonne che sembrano quote Casa: {potential_cols}")

if not potential_cols:
    print("❌ NON TROVO COLONNE QUOTE! Controlla il CSV.")
    exit()

# Scegliamo le migliori (Priorità: B365 -> Avg -> Prima che trova)
if 'B365H' in odds.columns:
    cols = ['B365H', 'B365D', 'B365A']
    print("✅ Uso quote Bet365")
elif 'AvgH' in odds.columns:
    cols = ['AvgH', 'AvgD', 'AvgA']
    print("✅ Uso quote Medie (Avg)")
else:
    # Prende la prima che ha trovato (es. PSH)
    prefix = potential_cols[0][:-1] # Toglie l'ultima lettera H
    cols = [prefix+'H', prefix+'D', prefix+'A']
    print(f"⚠️ Non trovo Bet365, uso: {cols}")

# 3. FIX DATE (Il problema più probabile)
print("\n🛠️ --- RIPARAZIONE DATE ---")
print(f"Esempio Data Dataset (Prima): {df['date'].iloc[0]}")
print(f"Esempio Data Odds (Prima):    {odds['date'].iloc[0]}")

# Trasformiamo tutto in Datetime
df['date'] = pd.to_datetime(df['date'])
odds['date'] = pd.to_datetime(odds['date'])

# !!! TRUCCO FONDAMENTALE !!!
# Rimuoviamo l'orario (normalize) così 20:45 diventa 00:00 e combacia
df['date_norm'] = df['date'].dt.normalize()
odds['date_norm'] = odds['date'].dt.normalize()

print("✅ Date normalizzate (orario rimosso).")

# 4. STANDARDIZZAZIONE NOMI
def clean_name(name):
    if pd.isna(name): return ""
    name = str(name).lower().strip()
    replacements = {
        'inter': 'inter', 'internazionale': 'inter',
        'milan': 'milan', 'ac milan': 'milan',
        'juventus': 'juventus', 'juve': 'juventus',
        'manchester city': 'man city', 'man city': 'man city',
        'manchester united': 'man utd', 'man utd': 'man utd',
        'verona': 'verona', 'hellas verona': 'verona'
    }
    return replacements.get(name, name)

df['team_key'] = df['team'].apply(clean_name)
odds['home_key'] = odds['home_team'].apply(clean_name)
odds['away_key'] = odds['away_team'].apply(clean_name)

# 5. PREPARAZIONE QUOTE (Match-Centric -> Team-Centric)
# Creiamo un dizionario lookup: (Data, Squadra) -> [1, X, 2]
odds_lookup = {}

print("🔄 Indicizzazione quote...")
count = 0
for _, row in odds.iterrows():
    d = row['date_norm']
    h_team = row['home_key']
    a_team = row['away_key']
    
    # Quote 1X2 dal CSV
    q1 = row[cols[0]]
    qX = row[cols[1]]
    q2 = row[cols[2]]
    
    # Per la squadra di CASA: Win=1, Draw=X, Lose=2
    odds_lookup[(d, h_team)] = {'Odds_1': q1, 'Odds_X': qX, 'Odds_2': q2}
    
    # Per la squadra OSPITE: Win=2, Draw=X, Lose=1 (INVERSIONE!)
    # Nota: Qui Odds_1 significa "Quota che IO vinca" (che è il 2 della schedina)
    odds_lookup[(d, a_team)] = {'Odds_1': q2, 'Odds_X': qX, 'Odds_2': q1}
    count += 1

print(f"   Indicizzate {len(odds_lookup)} coppie (Data, Squadra).")

# 6. APPLICAZIONE AL DATASET
print("🔗 Unione al dataset...")

def get_odds(row):
    key = (row['date_norm'], row['team_key'])
    if key in odds_lookup:
        return pd.Series(odds_lookup[key])
    else:
        return pd.Series([None, None, None], index=['Odds_1', 'Odds_X', 'Odds_2'])

# Applica la funzione riga per riga
odds_cols = df.apply(get_odds, axis=1)
df_final = pd.concat([df, odds_cols], axis=1)

# Pulizia
df_final.drop(columns=['date_norm', 'team_key'], inplace=True)

# 7. VERIFICA FINALE
missing = df_final['Odds_1'].isna().sum()
print("\n📊 --- RISULTATO ---")
print(f"Totale Righe: {len(df_final)}")
print(f"Righe con Quote: {len(df_final) - missing}")
print(f"Righe SENZA Quote: {missing}")

if missing < len(df_final):
    print("✅ SUCCESSO! Le quote sono state inserite.")
    df_final.to_csv('data/dataset_con_quote_FIXED_3.csv', index=False)
    print("📁 Salvato in: 'data/dataset_con_quote_FIXED.csv'")
else:
    print("❌ ANCORA TUTTO VUOTO. Il problema è nei nomi delle squadre o le date non coincidono per niente.")
    print("Esempio Chiave Dataset:", (df['date_norm'].iloc[0], df['team_key'].iloc[0]))
    print("Esempio Chiave Odds (primi 3):", list(odds_lookup.keys())[:3])