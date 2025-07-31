# NomiAI

Sistema di analisi finanziaria basato su AI.

## 📦 Installazione

1. **Installa le dipendenze da PyPI**

   Assicurati di avere Python 3 installato. Poi esegui:

   ```bash
   pip3 install -r requirements.txt
   ```

2. **Imposta le chiavi API**

   * Esporta la chiave API di Google (per Gemini):

     ```bash
     export GOOGLE_API_KEY="la_tua_chiave_google"
     ```

   * Sostituisci `{FMP_API_KEY}` nel codice con la tua chiave API di [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/).

3. **Esegui il programma**

   ```bash
   python3 agent.py
   ```
