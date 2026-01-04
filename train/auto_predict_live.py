import pandas as pd
import requests
from bs4 import BeautifulSoup
from thefuzz import process
import joblib
import warnings
import datetime

warnings.filterwarnings('ignore')

print("--- 🤖 PREDIZIONE LIVE INTERATTIVA ---")

# --- CONFIGURAZIONE ---
MATCHES_TONIGHT = [
    ("Sevilla", "Levante"),
    # Aggiungi qui altre partite...
]

FILE_PLAYERS = 'data/players.csv'
FILE_VALUATIONS = 'data/player_valuations.csv'
FILE_HISTORY = 'data/dataset_train_final_3.csv'
MODEL_PATH = 'train/modello_serie_a.pkl' # O 'train/modello_serie_a.pkl'

# ==============================================================================
# 1. FUNZIONI DI SCRAPING (Con gestione errore migliorata)
# ==============================================================================
def get_probable_lineups(home_team, away_team):
    url = "https://www.fantacalcio.it/probabili-formazioni-serie-a"
    print(f"🌍 Cerco formazioni su Fantacalcio.it...")
    
    found_players = {home_team: [], away_team: []}
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Logica di ricerca generica per trovare i team
        match_divs = soup.find_all('div', class_='match-preview')
        
        for div in match_divs:
            text = div.get_text().lower()
            if home_team.lower() in text and away_team.lower() in text:
                print(f"   ✅ Partita trovata nel sito!")
                # Tenta di estrarre i nomi dai link giocatori
                player_links = div.find_all('a', class_='player-name')
                names = [p.get_text().strip() for p in player_links]
                
                if len(names) >= 22:
                    found_players[home_team] = names[:11]
                    found_players[away_team] = names[11:22]
                    return found_players
                break
    except Exception:
        pass # Se fallisce, restituisce liste vuote e gestiamo dopo
        
    return found_players

# ==============================================================================
# 2. CALCOLO VALORE (Con Fuzzy Matching)
# ==============================================================================
def calculate_lineup_value(player_names, df_vals, df_players):
    total_value = 0
    mapped_count = 0
    
    # Valutazioni più recenti
    latest_date = df_vals['date'].max()
    df_recent_vals = df_vals[df_vals['date'] == latest_date]
    kaggle_names = df_players['name'].unique()
    
    print(f"   ...Calcolo valore su {len(player_names)} giocatori...")
    
    for p_name in player_names:
        # Fuzzy Match per trovare il nome nel database Kaggle
        match, score = process.extractOne(p_name, kaggle_names)
        
        if score > 80:
            p_id = df_players[df_players['name'] == match]['player_id'].values[0]
            val_row = df_recent_vals[df_recent_vals['player_id'] == p_id]
            if not val_row.empty:
                val = val_row['market_value_in_eur'].values[0]
                total_value += val
                mapped_count += 1
    
    return total_value

# ==============================================================================
# 3. CARICAMENTO DATI
# ==============================================================================
print("📂 Carico dati...")
try:
    df_p = pd.read_csv(FILE_PLAYERS)
    df_v = pd.read_csv(FILE_VALUATIONS)
    df_v['date'] = pd.to_datetime(df_v['date'])
    df_hist = pd.read_csv(FILE_HISTORY)
    
    # Calcolo valore tipico storico (Mediana ultima stagione)
    # Serve come denominatore per il Lineup_Ratio
    vals = pd.concat([
        df_hist[['Home_Team', 'Home_Value']].rename(columns={'Home_Team': 'Team', 'Home_Value': 'Value'}),
        df_hist[['Away_Team', 'Away_Value']].rename(columns={'Away_Team': 'Team', 'Away_Value': 'Value'})
    ])
    typical_values = vals.groupby('Team')['Value'].median().to_dict()
    
    model = joblib.load(MODEL_PATH)
    print("✅ Ready.")

except Exception as e:
    print(f"❌ Errore file: {e}")
    exit()

# ==============================================================================
# 4. LOOP PARTITE CON INPUT MANUALE
# ==============================================================================
print("\n" + "="*50)
for home, away in MATCHES_TONIGHT:
    print(f"⚽ {home.upper()} vs {away.upper()}")
    
    # 1. TENTATIVO AUTOMATICO
    lineups = get_probable_lineups(home, away)
    
    # 2. INPUT MANUALE SE FALLISCE
    # Gestione CASA
    if not lineups[home]:
        print(f"   ⚠️  Non ho trovato la formazione per {home}.")
        choice = input(f"   Vuoi inserire i nomi manualmente? (s/n) [n usa storico]: ").strip().lower()
        if choice == 's':
            raw_input = input(f"   📝 Incolla i titolari {home} (separati da virgola): ")
            lineups[home] = [x.strip() for x in raw_input.split(',') if x.strip()]
    
    # Gestione OSPITE
    if not lineups[away]:
        print(f"   ⚠️  Non ho trovato la formazione per {away}.")
        choice = input(f"   Vuoi inserire i nomi manualmente? (s/n) [n usa storico]: ").strip().lower()
        if choice == 's':
            raw_input = input(f"   📝 Incolla i titolari {away} (separati da virgola): ")
            lineups[away] = [x.strip() for x in raw_input.split(',') if x.strip()]

    # 3. CALCOLO VALORI
    # Se la lista è ancora vuota, usa il valore storico (Fallback)
    val_home = calculate_lineup_value(lineups[home], df_v, df_p) if lineups[home] else typical_values.get(home, 100_000_000)
    val_away = calculate_lineup_value(lineups[away], df_v, df_p) if lineups[away] else typical_values.get(away, 100_000_000)
    
    # 4. RECUPERO FORMA
    last_h = df_hist[(df_hist['Home_Team']==home) | (df_hist['Away_Team']==home)].iloc[-1]
    last_a = df_hist[(df_hist['Home_Team']==away) | (df_hist['Away_Team']==away)].iloc[-1]
    
    # Logica per prendere stats corrette (Attack/Defense)
    h_att = last_h['Home_Attack_Form'] if last_h['Home_Team'] == home else last_h['Away_Attack_Form']
    h_def = last_h['Home_Defense_Form'] if last_h['Home_Team'] == home else last_h['Away_Defense_Form']
    a_att = last_a['Home_Attack_Form'] if last_a['Home_Team'] == away else last_a['Away_Attack_Form']
    a_def = last_a['Home_Defense_Form'] if last_a['Home_Team'] == away else last_a['Away_Defense_Form']

    # 5. PREDIZIONE
    input_row = pd.DataFrame([{
        'Value_Ratio_vs_Opponent': val_home / (val_away + 1),
        'Home_Value': val_home,
        'Away_Value': val_away,
        'Home_Lineup_Ratio': val_home / typical_values.get(home, val_home),
        'Away_Lineup_Ratio': val_away / typical_values.get(away, val_away),
        'Home_Attack_Form': h_att,
        'Home_Defense_Form': h_def,
        'Away_Attack_Form': a_att,
        'Away_Defense_Form': a_def,
        'Home_Attack_vs_Def': h_att - a_def,
        'Away_Attack_vs_Def': a_att - h_def
    }])
    
    probs = model.predict_proba(input_row)[0]
    
    print(f"\n   📊 1: {probs[0]:.0%} | X: {probs[1]:.0%} | 2: {probs[2]:.0%}")