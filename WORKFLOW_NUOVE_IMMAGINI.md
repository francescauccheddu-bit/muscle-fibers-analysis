# Workflow: Analisi Nuove Immagini con Staining Migliorato

## 📋 Panoramica

Questo documento descrive il processo per analizzare nuove immagini di laminina con staining migliorato e confrontare i risultati con le analisi precedenti.

---

## 🔄 Workflow Completo

### Step 1: Preparazione Immagine

1. **Posiziona la nuova immagine** nella cartella `data/`:
   ```powershell
   # Da Windows
   cd "C:\FRANCESCAdocs\unipd\COLLABORAZIONI\VITIELLO MUSCOLI\CODE\muscle-fibers-analysis"

   # Copia l'immagine nella cartella data/
   copy "percorso\nuova_immagine.tif" data\
   ```

2. **Verifica che l'immagine sia presente**:
   ```powershell
   dir data\
   ```

### Step 2: Esegui Analisi Pipeline Completa

Usa lo script `analyze_laminina_complete.py` con calibrazione:

```powershell
python scripts/analyze_laminina_complete.py `
  --input data/nuova_immagine.tif `
  --output output_nuova_immagine `
  --kernel-size 15 `
  --min-fiber-area 1000 `
  --dot-radius 5 `
  --pixel-size 0.41026
```

**Parametri**:
- `--input`: Percorso all'immagine di input (fluorescenza laminina)
- `--output`: Directory dove salvare i risultati
- `--kernel-size 15`: Dimensione kernel per morphological closing (raccomandato)
- `--min-fiber-area 1000`: Area minima fibra in pixel² (filtra rumore)
- `--dot-radius 5`: Raggio pallini rossi centroidi (3-10)
- `--pixel-size 0.41026`: Calibrazione µm/pixel (da 2.4375 px/µm)

**Output generato**:
```
output_nuova_immagine/
  ├── laminina_with_centroids.png           # Fluorescenza + gap blu + pallini rossi
  ├── skeleton_with_centroids.png           # Circuiti chiusi + gap blu + pallini
  ├── closing_gaps_visualization.png        # Gap in verde su grigio
  ├── area_distribution.png                 # Istogramma distribuzione aree
  ├── fibers_statistics.csv                 # Dati dettagliati per ogni fibra
  ├── summary_statistics.csv                # Statistiche sommarie
  └── metadata.json                         # Metadati analisi
```

### Step 3: Confronta con Analisi Precedente

Usa lo script di confronto `compare_analyses.py`:

```powershell
python scripts/compare_analyses.py `
  --analyses output_final output_nuova_immagine `
  --labels "Vecchia Immagine (Oct 2024)" "Nuova Immagine (Staining Migliorato)" `
  --output comparison_vecchia_vs_nuova
```

**Parametri**:
- `--analyses`: Lista di directory output da confrontare (spazio-separate)
- `--labels`: Etichette descrittive per ogni analisi (stesso ordine)
- `--output`: Directory dove salvare il report comparativo

**Output generato**:
```
comparison_vecchia_vs_nuova/
  ├── COMPARISON_REPORT.md                  # Report markdown dettagliato
  ├── comparison_summary.csv                # Tabella comparativa CSV
  ├── comparison_area_distributions.png     # Istogrammi sovrapposti
  ├── comparison_boxplot.png                # Box plot comparativo
  └── comparison_metrics.png                # Grafici a barre metriche principali
```

### Step 4: Visualizza Risultati

1. **Apri il report**:
   ```powershell
   # Apri con editor markdown (es. VS Code)
   code comparison_vecchia_vs_nuova\COMPARISON_REPORT.md
   ```

2. **Visualizza le immagini**:
   - `comparison_area_distributions.png`: Confronta le distribuzioni delle aree
   - `comparison_boxplot.png`: Visualizza statistiche distribuzionali
   - `comparison_metrics.png`: Confronta numero fibre, area media, coverage

3. **Analizza i CSV**:
   ```powershell
   # Apri con Excel
   start excel comparison_vecchia_vs_nuova\comparison_summary.csv
   ```

---

## 📊 Risultati Attesi

### Scenario 1: Stesso Tessuto, Staining Migliorato

**Aspettative**:
- ✅ **Numero fibre simile** (±10%)
- ✅ **Area media simile** (±20%)
- ✅ **Distribuzione aree simile**
- ✅ **Coverage maschera maggiore** (meno gap)

**Esempio**:
```
Vecchia Immagine:  2,456 fibre | Area media: 1,357 µm²
Nuova Immagine:    2,650 fibre | Area media: 1,420 µm²
Variazione:        +194 (+7.9%) | +63 µm² (+4.6%)
```

### Scenario 2: Campione Diverso

**Indicatori**:
- ⚠️ **Numero fibre molto diverso** (>30% differenza)
- ⚠️ **Area media molto diversa** (>2x rapporto)
- ⚠️ **Dimensioni immagine diverse**

**Esempio** (risultati reali dalle analisi precedenti):
```
Vecchia Immagine:  2,456 fibre | Area media: 1,357 µm² | 10,110×8,984 px
Nuova Immagine:    4,025 fibre | Area media: 5,351 µm² | 14,459×20,734 px
Conclusione:       Campioni diversi (area 3.9x maggiore, 63% più fibre)
```

---

## 🔧 Parametri di Calibrazione

### Calibrazione Pixel ↔ Micrometri

**Dalla microscopia**:
- **2.4375 pixel = 1 µm**
- **1 pixel = 0.41026 µm**

**Conversioni**:
```python
# Pixel → µm
area_um2 = area_px * (0.41026 ** 2)
perimeter_um = perimeter_px * 0.41026

# µm → Pixel
area_px = area_um2 / (0.41026 ** 2)
perimeter_px = perimeter_um / 0.41026
```

**Verifica calibrazione**:
Se hai una nuova immagine con calibrazione diversa, aggiorna il parametro `--pixel-size`:

```powershell
# Esempio: se calibrazione è 3.0 px/µm
python scripts/analyze_laminina_complete.py `
  --input data/nuova_immagine.tif `
  --output output_nuova `
  --pixel-size 0.333333  # = 1/3.0
```

---

## 📈 Interpretazione Metriche

### Numero Fibre

- **Cosa misura**: Quante fibre muscolari sono state identificate e segmentate
- **Fattori che influenzano**:
  - Qualità dello staining
  - Risoluzione immagine
  - Parametro `--min-fiber-area` (filtra fibre piccole)
  - Efficacia del morphological closing

### Area Media

- **Cosa misura**: Dimensione media delle fibre muscolari
- **Interpretazione**:
  - Varia con tipo muscolare (fibre tipo I vs tipo II)
  - Può indicare ipertrofia/atrofia
  - Influenzata da taglio istologico (sezione trasversale vs obliqua)

### Coverage Maschera

- **Cosa misura**: Percentuale di immagine coperta dalle fibre
- **Valori tipici**: 70-85%
- **Interpretazione**:
  - Basso (<60%): possibile danneggiamento tessuto, gap grandi
  - Alto (>85%): buona qualità staining, pochi gap

### Gap Chiusi (Morphological Closing)

- **Cosa misura**: Quanti pixel sono stati aggiunti per chiudere gap nelle fibre
- **Visualizzazione**: Pixel blu chiaro nelle immagini output
- **Interpretazione**:
  - 1-3%: Ottimo staining, pochi gap
  - 3-7%: Buono, gap moderati
  - >10%: Molti gap, considerare migliorare staining

---

## 🎯 Raccomandazioni

### Per Ottenere Risultati Ottimali

1. **Qualità Immagine**:
   - Usa staining di alta qualità (immunofluorescenza laminina)
   - Evita artefatti da fissazione
   - Assicura illuminazione uniforme (evita vignetting)

2. **Parametri Pipeline**:
   - **Kernel size 15**: Ottimale per la maggior parte dei casi
   - Se gap molto piccoli (<5 px): prova `--kernel-size 11`
   - Se gap grandi (>20 px): prova `--kernel-size 21` (attenzione a fusioni)

3. **Validazione**:
   - Confronta sempre con conteggio manuale su subset
   - Visualizza `laminina_with_centroids.png` per controllo qualità
   - Verifica `closing_gaps_visualization.png` per vedere gap chiusi

4. **Confronto con Letteratura**:
   - Area media fibre umane: ~3,000-6,000 µm² (varia con muscolo)
   - Numero fibre per sezione: dipende da dimensione ROI

---

## 🐛 Troubleshooting

### Problema: Troppo poche fibre identificate

**Possibili cause**:
- `--min-fiber-area` troppo alto → prova valori più bassi (500, 750)
- `--kernel-size` troppo grande → prova valori più piccoli (11, 9)
- Immagine di bassa qualità → considera pre-processing

### Problema: Troppe "fibre" false (rumore)

**Soluzioni**:
- Aumenta `--min-fiber-area` (1500, 2000)
- Verifica soglia di segmentazione nell'immagine originale

### Problema: Fibre fuse insieme

**Soluzioni**:
- Riduci `--kernel-size` (11, 9, 7)
- Controlla qualità staining (gap troppo grandi tra fibre)

### Problema: Coverage molto bassa (<50%)

**Possibili cause**:
- Immagine danneggiata o artefatti
- Staining non uniforme
- Threshold di segmentazione non ottimale

---

## 📚 File di Riferimento

### Script Disponibili

1. **`analyze_laminina_complete.py`**: Pipeline completa all-in-one
   - Segmentazione → Closing → Identificazione → Statistiche → Visualizzazioni

2. **`compare_analyses.py`**: Confronto tra multiple analisi
   - Genera report comparativi dettagliati con grafici

3. **`segment_laminina.py`**: Solo segmentazione (se vuoi testare metodi)
   - Adaptive threshold, CLAHE+Otsu, Morphological gradient

4. **`close_cycles_multithreshold.py`**: Closing intelligente (sperimentale)
   - Usa intensità fluorescenza per chiudere gap

### Documentazione

- **`README.md`**: Panoramica generale progetto
- **`QUICKSTART_FLUORESCENCE.md`**: Guida rapida per nuovi utenti
- **`ANALYSIS_COMPARISON_REPORT.md`**: Report confronto kernel K15 vs K21
- **`WORKFLOW_NUOVE_IMMAGINI.md`**: Questo documento

---

## 💡 Workflow Esempio Completo

```powershell
# 1. Posiziona immagine
copy "D:\microscopia\sample_202501.tif" data\sample_202501.tif

# 2. Esegui analisi
python scripts/analyze_laminina_complete.py `
  --input data/sample_202501.tif `
  --output output_sample_202501 `
  --kernel-size 15 `
  --min-fiber-area 1000 `
  --dot-radius 5 `
  --pixel-size 0.41026

# 3. Confronta con baseline
python scripts/compare_analyses.py `
  --analyses output_final output_sample_202501 `
  --labels "Baseline (Oct 2024)" "Sample Jan 2025" `
  --output comparison_baseline_vs_jan2025

# 4. Visualizza report
code comparison_baseline_vs_jan2025\COMPARISON_REPORT.md

# 5. Commit risultati (opzionale - solo CSV/JSON, non PNG)
git add output_sample_202501/*.csv output_sample_202501/*.json
git add comparison_baseline_vs_jan2025/*.md comparison_baseline_vs_jan2025/*.csv
git commit -m "Add analysis for sample 202501 and comparison with baseline"
git push origin main
```

---

**Ultimo aggiornamento**: 2025-11-18
**Autore**: Claude Code AI Assistant
