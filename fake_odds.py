import pandas as pd

print("--- AGGIUNTA QUOTE FITTIZIE (PER BYPASSARE MERGE_ODDS) ---")

try:
    # 1. Carica il file uscito da add_final_features
    df = pd.read_csv('data/dataset_xgboost_ready_3.csv')
    print(f"✅ Caricato dataset: {len(df)} righe.")
    
    # 2. Aggiungi le colonne quote con valore neutro (1.0)
    # Questo serve solo perché prepare_final_dataset se le aspetta
    df['Odds_1'] = 1.0
    df['Odds_X'] = 1.0
    df['Odds_2'] = 1.0
    
    # 3. Salva con il nome che di solito usa merge_odds
    # (Così prepare_final_dataset lo troverà pronto)
    output_name = 'data/dataset_con_quote_3.csv'
    df.to_csv(output_name, index=False)
    
    print(f"✅ File salvato come: {output_name}")
    print("Ora puoi lanciare prepare_final_dataset.py senza errori!")

except FileNotFoundError:
    print("❌ Errore: Non trovo 'data/dataset_xgboost_ready.csv'. Hai lanciato add_final_features.py?")