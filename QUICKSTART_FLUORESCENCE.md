# Quick Start Guide - Fluorescence Laminina Segmentation

Questa guida spiega come usare la pipeline completa per segmentare immagini di fluorescenza della laminina e chiudere automaticamente i cicli delle fibre.

## 🎯 Obiettivo

Automatizzare il processo che prima richiedeva:
- Photoshop/ImageJ per creare maschere manualmente
- QuantiMus (che fallisce su immagini a mosaico con intensità variabile)

## 📋 Pipeline Completa (2 Step)

### STEP 1: Segmentazione Iniziale

Crea la maschera automaticamente dall'immagine di fluorescenza originale.

```bash
python scripts/segment_laminina.py \
  --input data/laminina_originale.png \
  --reference data/laminina_maschera_reference.png \
  --output output_segmentation
```

**Cosa fa**:
- Testa 3 metodi di segmentazione (adaptive, CLAHE+Otsu, gradient)
- Confronta ogni metodo con la maschera di riferimento
- Calcola metriche: IoU, Dice, Precision, Recall, F1-score
- Genera visualizzazioni comparative

**Output**:
```
output_segmentation/
  ├── mask_adaptive.png          # Maschera metodo adaptive threshold
  ├── mask_clahe.png              # Maschera metodo CLAHE+Otsu
  ├── mask_gradient.png           # Maschera metodo gradient
  ├── comparison_adaptive.png     # Confronto visivo + metriche
  ├── comparison_clahe.png
  └── comparison_gradient.png
```

**Interpretazione risultati**:
- **IoU/Dice** > 0.80: Ottimo
- **IoU/Dice** 0.60-0.80: Buono
- **IoU/Dice** < 0.60: Richiede tuning parametri

**Quale metodo scegliere?**
Guarda i file `comparison_*.png` e scegli il metodo con:
1. Migliore F1-score (bilanciamento precision/recall)
2. Migliore somiglianza visiva con il riferimento

### STEP 2: Chiusura Intelligente Cicli

Usa l'intensità dell'immagine originale per chiudere cicli intelligentemente.

```bash
# Esempio: usa la maschera dal metodo adaptive
python scripts/close_cycles_multithreshold.py \
  --mask output_segmentation/mask_adaptive.png \
  --fluorescence data/laminina_originale.png \
  --output output_closed \
  --n-thresholds 5
```

**Cosa fa**:
1. Identifica regioni di gap (cicli non chiusi)
2. Analizza intensità della fluorescenza nei gap
3. Testa 5 soglie diverse (da conservativa ad aggressiva)
4. Chiude progressivamente solo dove l'intensità supporta la connessione

**Output**:
```
output_closed/
  ├── mask_closed.png                      # Maschera finale con cicli chiusi
  ├── gap_regions.png                       # Visualizzazione gap identificati
  ├── multithreshold_visualization.png     # Confronto prima/dopo
  └── closing_statistics.json              # Statistiche numeriche
```

**Esempio closing_statistics.json**:
```json
{
  "new_pixels": 12500,
  "cycles_original": 45000,
  "cycles_closed": 32000,
  "cycles_improvement": 13000,
  "improvement_percentage": 28.9
}
```

## 🚀 Test Rapido con Immagini di Esempio

Se vuoi testare la pipeline su immagini piccole prima di usare quelle grandi:

```bash
# 1. Usa un'immagine di test piccola
python scripts/segment_laminina.py \
  --input data/ciclo_esempio.png \
  --reference data/ciclo_esempio.png \
  --output test_output \
  --method adaptive

# 2. Chiudi cicli
python scripts/close_cycles_multithreshold.py \
  --mask test_output/mask_adaptive.png \
  --fluorescence data/ciclo_esempio.png \
  --output test_closed
```

## 📊 Interpretazione Metriche

### STEP 1 - Qualità Segmentazione

| Metrica | Significato | Valore Ottimo |
|---------|-------------|---------------|
| **IoU (Jaccard)** | Overlap maschera predetta vs riferimento | > 0.80 |
| **Dice** | Simile a IoU, più sensibile a piccole variazioni | > 0.85 |
| **Precision** | % pixel predetti che sono corretti | > 0.85 |
| **Recall** | % pixel riferimento catturati | > 0.85 |
| **F1-score** | Media armonica Precision/Recall | > 0.85 |

### STEP 2 - Miglioramento Chiusura

- **cycles_improvement**: Pixel di gap chiusi
- **improvement_percentage**: % miglioramento rispetto all'originale
- Tipicamente 20-40% di miglioramento è un buon risultato

## 🔧 Tuning Parametri

### Se la segmentazione è troppo aggressiva (troppi pixel bianchi)

**Adaptive threshold**:
```bash
# Aumenta C (più conservativo)
python scripts/segment_laminina.py ... --method adaptive
# Poi modifica nel codice: C=5 invece di C=2
```

**CLAHE+Otsu**:
```bash
# Riduci clip_limit (meno equalizzazione)
# Modifica nel codice: clip_limit=1.5 invece di 2.0
```

### Se la chiusura cicli è troppo aggressiva

```bash
# Riduci numero soglie (più conservativo)
python scripts/close_cycles_multithreshold.py \
  --n-thresholds 3 \
  ...
```

### Se la chiusura cicli non chiude abbastanza

```bash
# Aumenta numero soglie (più aggressivo)
python scripts/close_cycles_multithreshold.py \
  --n-thresholds 8 \
  ...
```

## ⚙️ Pipeline Completa - Comando Singolo

Per processare tutto in sequenza (su Windows PowerShell):

```powershell
# STEP 1
python scripts/segment_laminina.py --input data/laminina_originale.png --reference data/laminina_maschera_reference.png --output output_segmentation

# STEP 2 (usa il metodo migliore da STEP 1)
python scripts/close_cycles_multithreshold.py --mask output_segmentation/mask_adaptive.png --fluorescence data/laminina_originale.png --output output_closed
```

## 📝 Note Importanti

1. **Immagini grandi**: Il processing di immagini da 273 MB può richiedere qualche minuto
2. **Memoria RAM**: Assicurati di avere almeno 4-8 GB RAM disponibili
3. **Confronto visivo**: Controlla SEMPRE le visualizzazioni, non solo le metriche numeriche
4. **Riferimento**: La maschera di riferimento è usata solo in STEP 1 per valutare la qualità

## ❓ Troubleshooting

### Errore: "Impossibile caricare immagine"
- Verifica che il path sia corretto (usa quote se contiene spazi)
- Controlla che l'immagine esista: `ls data/`

### Errore: "Dimensioni diverse"
- Maschera e fluorescenza devono avere stesse dimensioni
- Usa la stessa immagine originale in STEP 1 e STEP 2

### Risultati non buoni
1. Prova tutti e 3 i metodi in STEP 1
2. Guarda le visualizzazioni per capire dove sbaglia
3. Fai tuning dei parametri (vedi sezione sopra)
4. Considera pre-processing (es. denoising) dell'immagine originale

## 📧 Supporto

Per problemi o domande, apri un issue su GitHub con:
- Comando eseguito
- Errore ricevuto (se presente)
- Screenshot delle visualizzazioni output
- Info sulle immagini (dimensioni, dtype, range valori)
