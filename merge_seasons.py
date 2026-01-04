import pandas as pd
import os

print("--- UNIONE DATI STORICI E NUOVA STAGIONE ---")

# 1. Uniamo i LINEUPS
try:
    old_lineups = pd.read_csv('data/fbref_lineups.csv')
    new_lineups = pd.read_csv('data/fbref_lineups_2526.csv')
    
    # Concateniamo (mettiamo il nuovo sotto il vecchio)
    full_lineups = pd.concat([old_lineups, new_lineups], ignore_index=True)
    
    # Rimuoviamo eventuali duplicati (se per sbaglio c'erano partite sovrapposte)
    full_lineups.drop_duplicates(subset=['game', 'player'], inplace=True)
    
    # Sovrascriviamo il file originale
    full_lineups.to_csv('data/fbref_lineups.csv', index=False)
    print(f"✅ Lineups aggiornati! Totale righe: {len(full_lineups)}")
    
    # Pulizia: cancelliamo il file temporaneo
    os.remove('data/fbref_lineups_2526.csv')

except FileNotFoundError:
    print("⚠️ Non ho trovato i file lineups (vecchi o nuovi). Controlla i nomi.")

# 2. Uniamo le STATISTICHE (Match Stats)
try:
    old_stats = pd.read_csv('data/fbref_match_stats.csv')
    new_stats = pd.read_csv('data/fbref_match_stats_2526.csv')
    
    full_stats = pd.concat([old_stats, new_stats], ignore_index=True)
    full_stats.drop_duplicates(subset=['game', 'team'], inplace=True)
    
    full_stats.to_csv('data/fbref_match_stats.csv', index=False)
    print(f"✅ Stats aggiornate! Totale righe: {len(full_stats)}")
    
    os.remove('data/fbref_match_stats_2526.csv')

except FileNotFoundError:
    print("⚠️ Non ho trovato i file stats (vecchi o nuovi).")

# 3. Uniamo gli SCHEDULE (Calendario e Risultati)
try:
    print("3. Unione Schedule (Calendario)...")
    old_schedule = pd.read_csv('data/fbref_schedule.csv')
    
    # Assicurati che il nome del file sia corretto (quello che hai scaricato per la 25/26)
    # Se il tuo file si chiama diversamente (es. senza .csv), correggi qui sotto:
    new_schedule = pd.read_csv('data/fbref_schedule_2526.csv') 
    
    # Concateniamo
    full_schedule = pd.concat([old_schedule, new_schedule], ignore_index=True)
    
    # Rimuoviamo duplicati basandoci sull'ID della partita (game_id) o data+squadre
    if 'game_id' in full_schedule.columns:
        full_schedule.drop_duplicates(subset=['game_id'], inplace=True)
    else:
        # Fallback se manca game_id: usiamo data e squadre
        full_schedule.drop_duplicates(subset=['date', 'home_team', 'away_team'], inplace=True)
    
    # Ordiniamo per data (così le nuove vanno in fondo)
    full_schedule = full_schedule.sort_values('date')
    
    # Sovrascriviamo il Master File
    full_schedule.to_csv('data/fbref_schedule.csv', index=False)
    print(f"✅ Schedule aggiornato! Totale partite: {len(full_schedule)}")

except FileNotFoundError:
    print("⚠️ ERRORE: Non trovo 'data/fbref_schedule.csv' o il file nuovo 'data/fbref_schedule_2526.csv'. Controlla i nomi!")