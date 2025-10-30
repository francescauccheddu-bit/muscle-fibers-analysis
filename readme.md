# Muscle Fibers Analysis

Progetto per l'elaborazione e analisi di immagini al microscopio di fibre muscolari.

## Descrizione

Questo progetto fornisce strumenti per l'analisi automatizzata di immagini microscopiche di fibre muscolari. Gli script permettono di:

- **Segmentare immagini fluorescenza laminina** (nuovo!)
- Processare immagini da microscopio
- Identificare cicli chiusi nelle fibre muscolari
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
```

## Script Disponibili

### 1. Segmentazione Automatica Fluorescenza Laminina

**Script**: `scripts/segment_laminina.py`

Crea maschere automatiche da immagini di fluorescenza della laminina, replicando il processo manuale di Photoshop/ImageJ. Risolve il problema di QuantiMus che fallisce su immagini a mosaico con intensità variabile.

**Metodi disponibili**:
- **Adaptive Threshold**: gestisce intensità variabile locale
- **CLAHE + Otsu**: equalizzazione contrasto + soglia automatica
- **Morphological Gradient**: rilevamento bordi/contorni

**Uso**:
```bash
# Testa tutti i metodi (raccomandato)
python scripts/segment_laminina.py \
  --input data/laminina_originale.png \
  --reference data/laminina_maschera_reference.png \
  --output output_segmentation

# Oppure testa un solo metodo
python scripts/segment_laminina.py \
  --input data/laminina_originale.png \
  --reference data/laminina_maschera_reference.png \
  --output output_segmentation \
  --method adaptive
```

**Output**:
- `mask_*.png`: Maschere generate con i vari metodi
- `comparison_*.png`: Visualizzazioni comparative con metriche (IoU, Dice, Precision, Recall, F1)

**Metriche di similarità**:
- **IoU/Dice**: quanto la maschera predetta è simile al riferimento (1.0 = perfetto)
- **Precision**: percentuale di pixel predetti corretti
- **Recall**: percentuale di pixel del riferimento catturati
- **F1-score**: media armonica di precision e recall

### 2. Analisi Contorni e Cicli Chiusi

**Script**: `scripts/analyze_contours.py`

Identifica cicli chiusi nelle maschere binarie usando skeletonization e binary fill holes.

**Uso**:
```bash
python scripts/analyze_contours.py \
  --input data/Maschera.png \
  --output output \
  --min-cycle-area 1000 \
  --closing-size 3
```

**Output**:
- Skeleton sottile e ispessito con puntini rossi sui centroidi
- Maschera originale con bordi cicli e centroidi in rosso
- Istogramma distribuzione aree cicli