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

### 🚀 Pipeline Completa (RACCOMANDATO)

**Script**: `scripts/analyze_laminina_complete.py`

**Pipeline all-in-one** che fa tutto automaticamente:
1. Segmentazione da fluorescenza (adaptive threshold)
2. Morphological closing (K15)
3. Identificazione cicli e calcolo statistiche
4. Generazione visualizzazioni e CSV

**Uso rapido**:
```bash
python scripts/analyze_laminina_complete.py \
  --input data/laminina_originale.png \
  --output output_final \
  --kernel-size 15 \
  --min-fiber-area 1000 \
  --dot-radius 5
```

**Output**:
```
output_final/
  ├── laminina_with_centroids.png        # Fluorescenza + pallini rossi
  ├── skeleton_with_centroids.png        # Skeleton + pallini rossi
  ├── mask_with_cycles_and_centroids.png # Maschera + bordi cicli + pallini
  ├── area_distribution.png              # Istogramma aree fibre
  ├── fibers_statistics.csv              # Dati per ogni fibra
  ├── summary_statistics.csv             # Statistiche sommarie
  └── metadata.json                      # Metadati analisi
```

**CSV Output**:
- `fibers_statistics.csv`: fiber_id, area_px, centroid_x, centroid_y, perimeter_px, equiv_diameter_px, circularity, aspect_ratio, bbox_*
- `summary_statistics.csv`: n_fibers, mean_area_px, median_area_px, std_area_px, quartili, coverage, ecc.

**Parametri principali**:
- `--kernel-size 15`: dimensione closing (raccomandato 15)
- `--min-fiber-area 1000`: area minima fibra (filtra rumore)
- `--dot-radius 5`: raggio pallini (3-10)

---

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

### 2. Chiusura Intelligente Cicli (Multi-Threshold)

**Script**: `scripts/close_cycles_multithreshold.py`

**NOVITÀ!** Chiude i cicli aperti usando informazioni di intensità dall'immagine fluorescenza originale. Usa multiple soglie per identificare connessioni plausibili basate sull'intensità.

**Pipeline completa STEP 1 + STEP 2**:
```bash
# STEP 1: Crea maschera iniziale
python scripts/segment_laminina.py \
  --input data/laminina_originale.png \
  --reference data/laminina_maschera_reference.png \
  --output output_segmentation \
  --method adaptive

# STEP 2: Chiudi cicli intelligentemente
python scripts/close_cycles_multithreshold.py \
  --mask output_segmentation/mask_adaptive.png \
  --fluorescence data/laminina_originale.png \
  --output output_closed \
  --n-thresholds 5
```

**Come funziona**:
1. Identifica gap regions (cicli non chiusi)
2. Analizza intensità fluorescenza nei gap
3. Testa multiple soglie (dal più conservativo al più aggressivo)
4. Chiude progressivamente solo i cicli dove l'intensità supporta la connessione

**Output**:
- `mask_closed.png`: Maschera con cicli chiusi
- `gap_regions.png`: Visualizzazione dei gap identificati
- `multithreshold_visualization.png`: Comparazione prima/dopo + preview soglie
- `closing_statistics.json`: Statistiche di miglioramento (pixel chiusi, % improvement)

**Vantaggi rispetto a morphological closing semplice**:
- Usa informazioni di intensità (non solo morfologia)
- Più intelligente: chiude solo dove l'intensità supporta la connessione
- Meno rumore: evita connessioni spurie

### 3. Analisi Contorni e Cicli Chiusi

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