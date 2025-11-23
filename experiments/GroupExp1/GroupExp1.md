# Experiment 1

### Problem Definition:
**Target**

Vorhersage, ob der Preis von BTC/USD in den nächsten t = [5, 10, 15, 30, 60, 120, 240] Minuten steigt oder fällt in einem Zeitraum vom 01.01.2018 bis 15.11.2025
Dazu wird der Trend berechnet durch:
- lineare Regression des zukünftigen Preisfensters
- Normalisierung des Slopes durch den Durchschnittspreis
- binäre Label:
1 = Trend nach oben,
0 = Trend nach unten

Damit handelt es sich um ein Short-Term Crypto Trend Prediction Problem.

**Input Features**

Wir verzichten bewusst auf unnötig viele Indikatoren und konzentrieren uns auf wenige, erklärbare Features. Wenn wir uns sicherer fühlen, erweitern wir es natürlich.

Preisbasierte Features
- Normalisierte Close-Preise
- Log-Returns über 1 Minute

Trend-Features
- Normalisiertes exponential moving average (EMA) von t=[5, 10, 15, 20, 30, 60, 120, 240] minuten
- EMA-Differenz (Trendrichtung)
- Slope von EMA (Trendschärfe)

Volumen-Feature
- Normalisiertes Volumen

### Procedure Overview:

- Sammeln von 1-Minute OHLCV-Daten der Kryptowährungen BTC/USD und ETH/USD über die Alpaca Crypto API für den Zeitraum 01.01.2021 – 15.11.2025.
- Berechnung der Features: normalisierte Close-Preise, Log-Returns, EMAs (t = 5, 10, 15, 20, 30, 60, 120, 240 Minuten), EMA-Differenzen (z. B. EMA30 – EMA10), Slope von EMA10 und normalisiertes Handelsvolumen.
- Erstellung der Zielvariable für verschiedene Zeitfenster t ∈ {5, 15, 30, 60 Minuten}, die angibt, ob der Kurs steigt (1) oder fällt (0).
- Training eines neuronalen Netzwerks auf Basis dieser Features zur Vorhersage der kurzfristigen Trendrichtung (binäre Klassifikation) und Evaluation mittels zeitbasierter Train-/Validation-/Test-Splits.
- Optionales Backtesting der Modellvorhersagen in einer simplen Trading-Strategie: Long-Positionen bei positiven Trendvorhersagen eröffnen und für die jeweilige Dauer t halten.

Data Acquisition

Für die Datenerhebung verwenden wir ausschließlich programmatisch abrufbare historische Marktdaten. Unser Ziel ist es, eine konsistente, lückenfreie 1-Minuten-Zeitreihe für BTC/USD und ETH/USD aufzubauen, die anschließend als Basis für Feature Engineering und Modelltraining dient.

Abruf der historischen 1-Minute-OHLCV-Daten über die Alpaca Crypto API für den Zeitraum 01.01.2018 – 15.11.2025.

Nutzung der Endpunkte für historische Bars mit den Parametern:
symbol = {BTC/USD, ETH/USD}, timeframe = 1Min, start, end, limit-Pagination, UTC-Zeitzone.

Die Daten werden iterativ in Blöcken geladen, geprüft (monotone Zeitstempel, Duplikate, fehlende Minuten) und anschließend als raw CSV sowie zusätzlich als Parquet-Dateien gespeichert, um effizientes späteres Processing zu ermöglichen.

Speicherung erfolgt in einer klar getrennten Ordnerstruktur: raw (unverändert), staged (bereinigt) und processed (Feature-Versionen).

Zusätzlich verwenden wir Logging, um API-Requests, Pagination und mögliche Rate-Limit-Retries vollständig reproduzierbar zu dokumentieren.

Die Rohdaten enthalten open, high, low, close, volume sowie Zeitstempel. Ein Beispielausschnitt der Originaldaten wurde im Projektprotokoll festgehalten.

Durch diesen strukturierten Download-Prozess stellen wir sicher, dass die Datengrundlage vollständig, konsistent und für alle weiteren Schritte im Machine-Learning-Prozess verlässlich nutzbar ist.