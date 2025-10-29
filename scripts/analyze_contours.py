#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per analizzare contorni di fibre muscolari in sezione.

1. Pulisce il rumore dalla maschera binaria
2. Crea lo scheletro (linee di 1 pixel) dei contorni
3. Identifica i cicli chiusi (percorsi che tornano su se stessi)
4. Riempie le regioni dentro i cicli chiusi con verde
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage as ndi


def clean_and_skeletonize(binary_mask, min_area=500, border_exclusion=0, min_skeleton_size=50, close_gaps=3):
    """
    Pulisce la maschera dal rumore e crea lo scheletro dei contorni.

    Args:
        binary_mask: Maschera binaria con contorni
        min_area: Area minima per mantenere un oggetto
        border_exclusion: Larghezza del bordo da escludere
        min_skeleton_size: Dimensione minima (pixel) per componenti connesse dello scheletro
        close_gaps: Dimensione kernel per chiusura morfologica (0=disabilitato). Chiude gap nei contorni.

    Returns:
        skeleton: Scheletro dei contorni (linee di 1 pixel)
        cleaned: Maschera pulita prima della scheletonizzazione
    """
    print("Pulizia maschera...")

    # Converti in booleano
    mask_bool = binary_mask > 0

    # Rimuovi piccoli oggetti (rumore)
    cleaned = remove_small_objects(mask_bool, min_size=min_area)

    # Chiusura morfologica per connettere piccoli gap nei contorni
    if close_gaps > 0:
        print(f"Chiusura morfologica gap (kernel={close_gaps}) per connettere contorni...")
        from scipy import ndimage
        kernel = morphology.disk(close_gaps)
        closed = ndimage.binary_closing(cleaned, structure=kernel)

        # Conta oggetti prima e dopo
        n_before = measure.label(cleaned).max()
        n_after = measure.label(closed).max()
        print(f"  Regioni prima: {n_before}, dopo chiusura: {n_after}")

        cleaned = closed

    # Escludi bordi se richiesto
    if border_exclusion > 0:
        print(f"Esclusione bordo di {border_exclusion} pixel...")
        h, w = cleaned.shape
        cleaned[:border_exclusion, :] = False
        cleaned[-border_exclusion:, :] = False
        cleaned[:, :border_exclusion] = False
        cleaned[:, -border_exclusion:] = False

    # Conta oggetti rimasti
    labeled = measure.label(cleaned)
    n_objects = labeled.max()
    print(f"Oggetti dopo pulizia: {n_objects}")

    # Scheletonizza
    print("Creazione scheletro...")
    skeleton = skeletonize(cleaned)

    # Conta componenti scheletro
    skeleton_bool = skeleton > 0
    skeleton_labeled = measure.label(skeleton_bool, connectivity=2)
    n_skeleton_components = skeleton_labeled.max()
    print(f"  Componenti scheletro: {n_skeleton_components}")

    # Rimuovi componenti connesse piccole dello scheletro (pixel isolati/frammenti)
    if min_skeleton_size > 0:
        print(f"Rimozione frammenti scheletro < {min_skeleton_size} pixel...")

        # Rimuovi componenti piccole
        skeleton_cleaned = remove_small_objects(skeleton_bool, min_size=min_skeleton_size)

        # Conta componenti rimaste
        skeleton_labeled_after = measure.label(skeleton_cleaned, connectivity=2)
        n_after = skeleton_labeled_after.max()
        removed = n_skeleton_components - n_after
        print(f"  Componenti scheletro dopo rimozione: {n_after} (rimosse: {removed})")

        skeleton = skeleton_cleaned
    else:
        print(f"  (Rimozione frammenti disabilitata)")

    # Converti in uint8 per visualizzazione
    cleaned_uint8 = (cleaned * 255).astype(np.uint8)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    return skeleton_uint8, cleaned_uint8


def identify_closed_contours(skeleton, min_area=100, max_area=None, debug_single_cycle=None):
    """
    Identifica i percorsi chiusi (cicli) nello scheletro.

    Riempie tutti i "buchi" circondati completamente dallo scheletro.

    Args:
        skeleton: Scheletro binario (linee 1 pixel)
        min_area: Area minima per considerare una regione chiusa
        max_area: Area massima per considerare una regione chiusa (None = nessun limite)
        debug_single_cycle: Se specificato, colora solo questo ciclo (1-based)

    Returns:
        filled_mask: Maschera con cicli chiusi riempiti
        cycle_lines_mask: Linee dello scheletro che formano cicli
        stats: Statistiche sui percorsi
    """
    if debug_single_cycle is not None:
        print(f"*** MODALITA DEBUG: Colorerò SOLO il ciclo #{debug_single_cycle} ***")
    print("Identificazione cicli chiusi nello scheletro...")

    # Approccio:
    # 1. Etichetta componenti scheletro separate
    # 2. Per OGNI componente, riempi i buchi
    # 3. Sottrai lo scheletro per ottenere le aree interne

    # Etichetta componenti scheletro
    skeleton_bool = skeleton > 0
    skeleton_labeled = measure.label(skeleton_bool, connectivity=2)
    n_components = skeleton_labeled.max()

    print(f"Componenti scheletro: {n_components}")
    print(f"Riempimento buchi per ogni componente (potrebbe richiedere alcuni secondi)...")

    # Per ogni componente, riempi i buchi
    filled_all = np.zeros_like(skeleton, dtype=bool)

    # Kernel per dilatazione (ispessisce lo scheletro)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for comp_id in range(1, n_components + 1):
        # Mostra progresso ogni 10 componenti
        if comp_id % 10 == 0 or comp_id == n_components:
            print(f"  Processate {comp_id}/{n_components} componenti...")

        # Estrai questa componente
        component_mask = (skeleton_labeled == comp_id).astype(np.uint8) * 255

        # DILATA lo scheletro per ispessirlo (aiuta binary_fill_holes)
        dilated = cv2.dilate(component_mask, dilate_kernel, iterations=2)

        # Riempi i buchi nella versione dilatata
        filled_component = ndi.binary_fill_holes(dilated > 0)

        # Aggiungi al risultato totale
        filled_all = filled_all | filled_component

    print(f"  Completato!")
    filled_all = (filled_all * 255).astype(np.uint8)

    # Le aree interne ai cicli = filled - skeleton originale
    internal_areas = cv2.subtract(filled_all, skeleton)

    print(f"Pixel riempiti totali: {np.sum(internal_areas > 0):,}")

    # Ora etichetta le componenti connesse delle aree interne
    labeled = measure.label(internal_areas, connectivity=2)

    print(f"Trovate {labeled.max()} regioni interne")

    # Crea maschere
    filled_mask = np.zeros_like(skeleton)
    cycle_lines_mask = np.zeros_like(skeleton)

    closed_count = 0
    closed_areas = []
    skipped_border = 0
    skipped_small = 0
    skipped_large = 0

    h, w = skeleton.shape

    # Analizza ogni regione interna
    for region in measure.regionprops(labeled):
        label = region.label
        area = region.area

        # Ottieni la maschera di questa regione
        region_mask = (labeled == label).astype(np.uint8) * 255

        # Controlla se tocca il bordo
        touches_border = (
            np.any(region_mask[0, :] > 0) or np.any(region_mask[-1, :] > 0) or
            np.any(region_mask[:, 0] > 0) or np.any(region_mask[:, -1] > 0)
        )

        if touches_border:
            skipped_border += 1
            continue

        # Controlla area minima
        if area < min_area:
            skipped_small += 1
            continue

        # Controlla area massima (esclude regioni troppo grandi come lo sfondo)
        if max_area is not None and area > max_area:
            skipped_large += 1
            print(f"  Ciclo troppo grande ESCLUSO: area={area:.0f} px (max={max_area}), bbox={region.bbox}")
            continue

        # Questa è un'area interna a un ciclo chiuso!
        closed_count += 1
        print(f"  Ciclo chiuso {closed_count}: area={area:.0f} px, bbox={region.bbox}")

        # Se siamo in modalità debug, colora solo il ciclo specificato
        if debug_single_cycle is not None:
            if closed_count != debug_single_cycle:
                closed_areas.append(area)
                continue  # Salta questo ciclo
            else:
                print(f"  >>> QUESTO È IL CICLO DA COLORARE! <<<")

        # Aggiungi alla maschera riempita
        filled_mask[region_mask > 0] = 255

        # Trova le linee dello scheletro che circondano quest'area
        dilated = cv2.dilate(region_mask, np.ones((3, 3), np.uint8), iterations=1)
        cycle_skeleton = cv2.bitwise_and(skeleton, dilated)
        cycle_lines_mask = cv2.bitwise_or(cycle_lines_mask, cycle_skeleton)

        closed_areas.append(area)

    # Conta pixel riempiti
    filled_pixels = np.sum(filled_mask > 0)
    total_pixels = filled_mask.size

    print(f"\nRisultati:")
    print(f"  Cicli chiusi interni: {closed_count}")
    print(f"  Esclusi (toccano bordi): {skipped_border}")
    print(f"  Esclusi (troppo piccoli): {skipped_small}")
    print(f"  Esclusi (troppo grandi): {skipped_large}")
    print(f"  Pixel riempiti: {filled_pixels:,} ({filled_pixels/total_pixels*100:.2f}% dell'immagine)")

    stats = {
        'total_regions': labeled.max(),
        'closed': closed_count,
        'open': labeled.max() - closed_count,
        'avg_closed_area': np.mean(closed_areas) if closed_areas else 0,
        'min_closed_area': np.min(closed_areas) if closed_areas else 0,
        'max_closed_area': np.max(closed_areas) if closed_areas else 0
    }

    return filled_mask, cycle_lines_mask, stats


def analyze_fiber_contours(binary_mask, min_contour_area=100, border_exclusion=0):
    """
    Analizza i contorni nella maschera per identificare quali sono chiusi.

    Args:
        binary_mask: Maschera binaria con contorni delle fibre
        min_contour_area: Area minima per considerare un contorno
        border_exclusion: Larghezza del bordo da escludere (in pixel)

    Returns:
        closed_mask, open_mask, stats
    """
    print("Ricerca contorni...")

    # Se richiesto, escludi il bordo
    mask_to_analyze = binary_mask.copy()
    if border_exclusion > 0:
        print(f"Esclusione bordo di {border_exclusion} pixel...")
        # Crea maschera senza bordi
        h, w = binary_mask.shape
        mask_to_analyze[:border_exclusion, :] = 0  # Bordo superiore
        mask_to_analyze[-border_exclusion:, :] = 0  # Bordo inferiore
        mask_to_analyze[:, :border_exclusion] = 0  # Bordo sinistro
        mask_to_analyze[:, -border_exclusion:] = 0  # Bordo destro

    # Trova contorni usando OpenCV
    contours, hierarchy = cv2.findContours(
        mask_to_analyze,
        cv2.RETR_EXTERNAL,  # Solo contorni esterni
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Trovati {len(contours)} contorni (dopo esclusione bordi)")

    closed_mask = np.zeros_like(binary_mask)
    open_mask = np.zeros_like(binary_mask)

    closed_count = 0
    open_count = 0
    closed_contours_list = []
    open_contours_list = []

    print("Analisi contorni...")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        # Calcola il perimetro
        perimeter = cv2.arcLength(contour, closed=True)

        # Verifica se il contorno è chiuso analizzando la "compattezza"
        # Contorni chiusi hanno circolarità vicina a 1
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Un contorno è considerato "chiuso" se ha buona circolarità
        # e l'area interna è ben definita
        if circularity > 0.3:  # Soglia empirica
            # Contorno chiuso - riempi l'interno
            cv2.drawContours(closed_mask, [contour], -1, 255, thickness=cv2.FILLED)
            closed_count += 1
            closed_contours_list.append(contour)
        else:
            # Contorno aperto o molto irregolare
            cv2.drawContours(open_mask, [contour], -1, 255, thickness=2)
            open_count += 1
            open_contours_list.append(contour)

    stats = {
        'total': len(contours),
        'closed': closed_count,
        'open': open_count,
        'closed_contours': closed_contours_list,
        'open_contours': open_contours_list
    }

    return closed_mask, open_mask, stats


def close_open_contours(binary_mask, open_contours, max_gap=10):
    """
    Prova a chiudere i contorni aperti colmando i gap.

    Args:
        binary_mask: Maschera originale
        open_contours: Lista di contorni aperti
        max_gap: Distanza massima di gap da chiudere

    Returns:
        Maschera con contorni chiusi
    """
    print(f"Chiusura contorni aperti (max gap: {max_gap} pixel)...")

    result = binary_mask.copy()

    for contour in open_contours:
        # Prova con dilation morfologica per chiudere piccoli gap
        mask_single = np.zeros_like(binary_mask)
        cv2.drawContours(mask_single, [contour], -1, 255, thickness=2)

        # Dilata per chiudere gap piccoli
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap, max_gap))
        dilated = cv2.dilate(mask_single, kernel, iterations=1)

        # Poi erodi per ritornare alla dimensione originale
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Riempi l'area interna
        closed_filled = ndi.binary_fill_holes(eroded > 0).astype(np.uint8) * 255

        # Aggiungi al risultato
        result = cv2.bitwise_or(result, closed_filled)

    return result


def visualize_skeleton(original, cleaned, skeleton, cycle_lines_mask, filled_mask, stats, output_dir, base_name):
    """Visualizza il risultato della scheletonizzazione e identificazione percorsi chiusi."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creazione visualizzazioni...")

    # Aree riempite in verde
    filled_green = np.zeros((*original.shape, 3), dtype=np.uint8)
    filled_green[filled_mask > 0] = [0, 255, 0]  # Aree chiuse riempite in VERDE

    # Overlay su originale: aree riempite in verde
    overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay[filled_mask > 0] = [0, 255, 0]  # Aree chiuse in VERDE

    # Visualizzazione principale - 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Originale
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Maschera Originale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Dopo pulizia
    axes[0, 1].imshow(cleaned, cmap='gray')
    axes[0, 1].set_title('Dopo Pulizia Rumore', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Scheletro
    axes[0, 2].imshow(skeleton, cmap='gray')
    axes[0, 2].set_title('Scheletro (linee 1 pixel)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Aree riempite in verde
    axes[1, 0].imshow(filled_green)
    axes[1, 0].set_title(f'Cicli Chiusi Riempiti ({stats["closed"]})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay su originale
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay: Aree Cicli Chiusi in Verde', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Statistiche come testo
    stats_text = f"""STATISTICHE:

Regioni interne totali: {stats['total_regions']}
Cicli CHIUSI (interni): {stats['closed']}
Altri: {stats['open']}

Area media chiusi: {stats['avg_closed_area']:.0f} px
Area min: {stats['min_closed_area']:.0f} px
Area max: {stats['max_closed_area']:.0f} px
"""
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Salva
    vis_path = output_dir / f"{base_name}_skeleton_analysis.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {vis_path}")
    plt.close()

    # Salva maschere separate
    skeleton_path = output_dir / f"{base_name}_skeleton.png"
    cv2.imwrite(str(skeleton_path), skeleton)
    print(f"Scheletro salvato in: {skeleton_path}")

    cleaned_path = output_dir / f"{base_name}_cleaned.png"
    cv2.imwrite(str(cleaned_path), cleaned)
    print(f"Maschera pulita salvata in: {cleaned_path}")

    # Salva linee dei cicli chiusi
    cycle_lines_path = output_dir / f"{base_name}_cycle_lines.png"
    cv2.imwrite(str(cycle_lines_path), cycle_lines_mask)
    print(f"Linee cicli chiusi salvate in: {cycle_lines_path}")

    # Salva aree riempite (bianco e nero)
    filled_path = output_dir / f"{base_name}_closed_filled.png"
    cv2.imwrite(str(filled_path), filled_mask)
    print(f"Aree chiuse riempite (B/N) salvate in: {filled_path}")

    # Crea e salva immagine GRANDE con cicli in VERDE su sfondo NERO
    print("\nCreazione immagine cicli verdi...")
    cycles_green_image = np.zeros((*original.shape, 3), dtype=np.uint8)
    cycles_green_image[filled_mask > 0] = [0, 255, 0]  # VERDE per cicli chiusi

    # Salva come PNG
    cycles_green_path = output_dir / f"{base_name}_CICLI_VERDI.png"
    cv2.imwrite(str(cycles_green_path), cv2.cvtColor(cycles_green_image, cv2.COLOR_RGB2BGR))
    print(f"\n*** IMMAGINE CICLI VERDI salvata in: {cycles_green_path} ***")
    print(f"    Apri questo file per vedere i {stats['closed']} cicli riempiti di verde!")

    # Salva anche versione con overlay su originale
    overlay_large = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay_large[filled_mask > 0] = [0, 255, 0]
    overlay_path = output_dir / f"{base_name}_OVERLAY_CICLI.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_large, cv2.COLOR_RGB2BGR))
    print(f"    OVERLAY cicli su originale salvato in: {overlay_path}")



def visualize_classification(original, closed_mask, open_mask, stats, output_dir, base_name):
    """Visualizza la classificazione dei contorni (chiusi vs aperti)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creazione visualizzazioni...")

    # Crea immagine colorata
    colored = np.zeros((*original.shape, 3), dtype=np.uint8)

    # Contorni originali in bianco
    colored[original > 0] = [255, 255, 255]

    # Contorni chiusi riempiti di VERDE
    colored[closed_mask > 0] = [0, 255, 0]

    # Contorni aperti in ROSSO
    colored[open_mask > 0] = [255, 0, 0]

    # Visualizzazione principale - 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Originale
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Maschera Originale (Contorni)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Classificazione colorata
    axes[1].imshow(colored)
    axes[1].set_title(f'Verde=Chiusi ({stats["closed"]}), Rosso=Aperti ({stats["open"]})',
                        fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Solo contorni chiusi (riempiti di verde)
    green_only = np.zeros((*original.shape, 3), dtype=np.uint8)
    green_only[closed_mask > 0] = [0, 255, 0]
    axes[2].imshow(green_only)
    axes[2].set_title(f'Solo Contorni Chiusi Riempiti ({stats["closed"]})',
                        fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # Salva
    vis_path = output_dir / f"{base_name}_contour_analysis.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualizzazione salvata in: {vis_path}")
    plt.close()

    # Salva maschere separate
    closed_path = output_dir / f"{base_name}_closed_filled.png"
    cv2.imwrite(str(closed_path), closed_mask)
    print(f"Contorni chiusi riempiti salvati in: {closed_path}")

    colored_path = output_dir / f"{base_name}_colored_classification.png"
    cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
    print(f"Classificazione colorata salvata in: {colored_path}")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description='Analizza contorni di fibre muscolari in sezione'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Percorso alla maschera binaria con contorni'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Directory di output (default: output)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=500,
        help='Area minima per rimuovere rumore (default: 500 pixel)'
    )
    parser.add_argument(
        '--exclude-border',
        type=int,
        default=50,
        help='Larghezza del bordo da escludere dall\'analisi (default: 50 pixel)'
    )
    parser.add_argument(
        '--min-cycle-area',
        type=int,
        default=1000,
        help='Area minima per considerare un ciclo chiuso (default: 1000 pixel)'
    )
    parser.add_argument(
        '--max-cycle-area',
        type=int,
        default=100000,
        help='Area massima per considerare un ciclo chiuso (default: 100000 pixel). Esclude regioni troppo grandi come lo sfondo.'
    )
    parser.add_argument(
        '--close-gaps',
        type=int,
        default=0,
        help='Dimensione kernel per chiusura morfologica (default: 0=disabilitato). Connette gap nei contorni se necessario.'
    )
    parser.add_argument(
        '--min-skeleton-size',
        type=int,
        default=50,
        help='Dimensione minima componenti scheletro da mantenere (default: 50 pixel). Rimuove frammenti isolati sconnessi.'
    )
    parser.add_argument(
        '--debug-single-cycle',
        type=int,
        default=None,
        help='DEBUG: Colora solo il ciclo numero N (1-based). Utile per verificare un singolo ciclo.'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCHELETONIZZAZIONE CONTORNI FIBRE MUSCOLARI")
    print("=" * 60)

    # Carica
    print(f"\nCaricamento: {args.input}")
    original = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Errore: Impossibile caricare: {args.input}")
        return

    print(f"Dimensioni: {original.shape}")

    # Pulisci e scheletonizza
    print("\nPulizia e scheletonizzazione...")
    skeleton, cleaned = clean_and_skeletonize(
        original,
        min_area=args.min_area,
        border_exclusion=args.exclude_border,
        min_skeleton_size=args.min_skeleton_size,
        close_gaps=args.close_gaps
    )

    # Identifica percorsi chiusi
    print("\nIdentificazione percorsi chiusi...")
    filled_mask, cycle_lines_mask, stats = identify_closed_contours(
        skeleton,
        min_area=args.min_cycle_area,
        max_area=args.max_cycle_area,
        debug_single_cycle=args.debug_single_cycle
    )

    # Salva risultati
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    # Visualizza
    print("\nCreazione visualizzazioni...")
    visualize_skeleton(original, cleaned, skeleton, cycle_lines_mask, filled_mask, stats, args.output, base_name)

    print("\n" + "=" * 60)
    print("COMPLETATO")
    print("=" * 60)
    if args.debug_single_cycle is not None:
        print(f"MODALITÀ DEBUG: Colorato SOLO ciclo #{args.debug_single_cycle}")
    else:
        print(f"Percorsi CHIUSI: {stats['closed']}")
        print(f"Percorsi APERTI: {stats['open']}")
    print("=" * 60)

    print("\nANALISI COMPLETATA!")


if __name__ == '__main__':
    main()
