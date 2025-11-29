# Experiment 1

### Problem Definition:
**Target**

Vorhersage, ob der Preis von BTC/USD in den nächsten t = [5, 10, 15, 30, 60, 120, 240] Minuten steigt oder fällt in einem Zeitraum vom 01.01.2021 bis 15.11.2025
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

## Procedure Overview:

- Sammeln von 1-Minute OHLCV-Daten der Kryptowährungen BTC/USD und ETH/USD über die Alpaca Crypto API für den Zeitraum 01.01.2021 – 15.11.2025.
- Berechnung der Features: normalisierte Close-Preise, Log-Returns, EMAs (t = 5, 10, 15, 20, 30, 60, 120, 240 Minuten), EMA-Differenzen (z. B. EMA30 – EMA10), Slope von EMA und normalisiertes Handelsvolumen.
- Erstellung der Zielvariable für verschiedene Zeitfenster t ∈ {5, 15, 30, 60 Minuten}, die angibt, ob der Kurs steigt (1) oder fällt (0).
- Training eines neuronalen Netzwerks auf Basis dieser Features zur Vorhersage der kurzfristigen Trendrichtung (binäre Klassifikation) und Evaluation mittels zeitbasierter Train-/Validation-/Test-Splits.
- Optionales Backtesting der Modellvorhersagen in einer simplen Trading-Strategie: Long-Positionen bei positiven Trendvorhersagen eröffnen und für die jeweilige Dauer t halten.

---

### Data Acquisition

Extrahiert Rohdaten für BTC/USD und ETH/USD von der Alpaca Crypto API.

**Script**

[scripts/01_data_acquisition/crypto_data_acquisition.py](scripts/01_data_acquisition/crypto_data_acquisition.py)

Ruft 1-Minuten Daten von 2021-01-01 bis 2025-11-15 ab jeweils für BTC/USD und ETH/USD und speichert sie als .csv sowie .parquet Dateien in:
[data](data)


Beispiel für ETH/USD Daten:

<img src="images/01_ETHUSD_1m_raw.png" alt="drawing" width="800"/>

Beispiel für BTC/USD Daten:

<img src="images/01_BTCUSD_1m_raw.png" alt="drawing" width="800"/>

---

### Data Understanding
Visualisiert die Kursentwicklung und das Handelsvolumen von BTC und ETH und untersucht erste Eigenschaften wie auch Gemeinsamkeiten der Daten.

**Script**  

[02_crypto_data_understanding.py](scripts/02_data_understanding/02_crypto_data_understanding.py)

**Plots**  

![02_BTC_Zeitreihe.png](images/02_BTC_Zeitreihe.png)
![02_ETH_Zeitreihe.png](images/02_ETH_Zeitreihe.png)


![02_BTC_Volumen.png](images/02_BTC_Volumen.png)
![02_ETH_Volumen.png](images/02_ETH_Volumen.png)

**Erste Erkenntnisse**

- Die Close-Preise von BTC und ETH zeigen typische kurzfristige Schwankungen im Minutenbereich, wobei BTC tendenziell höhere Preisniveaus aufweist.
- Das Handelsvolumen variiert stark über die Zeit und zeigt Spitzen zu bestimmten Handelszeiten.
- BTC und ETH weisen teilweise ähnliche Bewegungsmuster auf, was auf eine gewisse Korrelation im Marktverhalten hinweist.

---

### Pre-Split Preparation


