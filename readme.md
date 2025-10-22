# Muscle Fibers Analysis

Progetto per l'elaborazione e analisi di immagini al microscopio di fibre muscolari.

## Descrizione

Questo progetto fornisce strumenti per l'analisi automatizzata di immagini microscopiche di fibre muscolari. Gli script permettono di:

- Processare immagini da microscopio
- Segmentare e identificare singole fibre muscolari
- Estrarre parametri morfometrici (area, diametro, perimetro)
- Classificare tipi di fibre
- Generare statistiche e visualizzazioni

## Struttura del Progetto

muscle-fibers-analysis/ ├── scripts/ # Script di analisi ├── data/ # Directory per immagini di input ├── output/ # Risultati delle analisi ├── requirements.txt # Dipendenze Python └── README.md # Documentazione


## Requisiti

- Python 3.8 o superiore
- Librerie specificate in `requirements.txt`

## Installazione

```bash
# Clona il repository
git clone https://github.com/francescauccheddu-bit/muscle-fibers-analysis.git
cd muscle-fibers-analysis

# Crea un ambiente virtuale (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt