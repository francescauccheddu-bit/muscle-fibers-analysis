# Script Obsoleti - Da Rimuovere

I seguenti script erano necessari durante lo sviluppo ma ora sono obsoleti perché tutte le funzionalità sono integrate in `analyze_laminina_complete.py`.

## ❌ Script da Rimuovere

### 1. `scripts/segment_laminina.py`
**Motivo**: Segmentazione ora integrata in `analyze_laminina_complete.py` (STEP 2)
**Sostituito da**: Pipeline completa

### 2. `scripts/close_cycles_multithreshold.py`
**Motivo**: Approccio multi-threshold aveva bug logico (0% improvement). Morphological closing K15 funziona meglio.
**Sostituito da**: `analyze_laminina_complete.py` usa morphological closing

### 3. `scripts/close_cycles_simple.py`
**Motivo**: Era per testare diversi kernel sizes. Ora usiamo K15 come standard.
**Sostituito da**: `analyze_laminina_complete.py` con `--kernel-size 15`

### 4. `scripts/overlay_centroids_on_fluorescence.py`
**Motivo**: Funzionalità integrata in `analyze_laminina_complete.py` (crea automaticamente laminina_with_centroids.png)
**Sostituito da**: Output automatico della pipeline completa

### 5. `scripts/analyze_contours.py`
**Motivo**: Script originale per analizzare maschere pre-esistenti. Ora creiamo maschere automaticamente.
**Potrebbe essere utile per**: Analisi di maschere manuali legacy
**Raccomandazione**: Mantieni per retrocompatibilità, ma non è più lo script principale

### 6. `scripts/close_contours.py`
**Motivo**: Vecchio approccio
**Stato**: Probabilmente non più usato

### 7. `scripts/contour_analysis.py`
**Motivo**: Vecchio approccio
**Stato**: Probabilmente non più usato

### 8. `scripts/fiber_analysis.py`
**Motivo**: Vecchio approccio
**Stato**: Probabilmente non più usato

### 9. `scripts/separate_and_close_fibers.py`
**Motivo**: Vecchio approccio
**Stato**: Probabilmente non più usato

## ✅ Script da Mantenere

### `scripts/analyze_laminina_complete.py` ✅
**SCRIPT PRINCIPALE** - Pipeline completa end-to-end

## Come Rimuovere gli Script Obsoleti

```bash
# Windows PowerShell
cd scripts

# Crea backup prima di rimuovere
mkdir obsolete_scripts
move segment_laminina.py obsolete_scripts/
move close_cycles_multithreshold.py obsolete_scripts/
move close_cycles_simple.py obsolete_scripts/
move overlay_centroids_on_fluorescence.py obsolete_scripts/
move close_contours.py obsolete_scripts/
move contour_analysis.py obsolete_scripts/
move fiber_analysis.py obsolete_scripts/
move separate_and_close_fibers.py obsolete_scripts/

# analyze_contours.py lo puoi mantenere per retrocompatibilità
```

## Linux/Mac

```bash
cd scripts
mkdir obsolete_scripts
mv segment_laminina.py close_cycles_multithreshold.py close_cycles_simple.py \
   overlay_centroids_on_fluorescence.py close_contours.py contour_analysis.py \
   fiber_analysis.py separate_and_close_fibers.py obsolete_scripts/
```

## Risultato Finale

Dopo la pulizia, `scripts/` conterrà solo:
- `analyze_laminina_complete.py` (PRINCIPALE)
- `analyze_contours.py` (opzionale, retrocompatibilità)
- `obsolete_scripts/` (backup)
