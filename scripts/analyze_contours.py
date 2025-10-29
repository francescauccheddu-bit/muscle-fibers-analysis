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


def identify_closed_contours(skeleton, min_area=100, max_area=None, exclude_largest=True, debug_single_cycle=None):
    """
    Identifica i percorsi chiusi (cicli) nello scheletro.

    NUOVO APPROCCIO: Inverti lo skeleton e trova contorni (le aree bianche diventano cicli)

    Args:
        skeleton: Scheletro binario (linee 1 pixel)
        min_area: Area minima per considerare una regione chiusa
        max_area: Area massima per considerare una regione chiusa (None = nessun limite)
        exclude_largest: Se True, esclude automaticamente il ciclo più grande (sfondo) (default: True)
        debug_single_cycle: Se specificato, colora solo questo ciclo (1-based)

    Returns:
        filled_mask: Maschera con cicli chiusi riempiti
        cycle_lines_mask: Linee dello scheletro che formano cicli
        centroids: Lista di coordinate (y, x) dei centroidi dei cicli chiusi
        stats: Statistiche sui percorsi
    """
    if debug_single_cycle is not None:
        print(f"*** MODALITA DEBUG: Colorerò SOLO il ciclo #{debug_single_cycle} ***")
    print("Identificazione cicli chiusi nello scheletro...")

    # NUOVO APPROCCIO:
    # 1. DILATA lo skeleton per chiudere piccoli gap (linee 1-pixel hanno spesso gap)
    # 2. INVERTI lo skeleton dilatato: linee diventano nere, sfondo diventa bianco
    # 3. I cicli (aree circondate da linee) diventano "isole bianche"
    # 4. cv2.findContours trova facilmente queste isole!

    print("Dilatazione skeleton per chiudere gap...")
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skeleton_dilated = cv2.dilate(skeleton, dilate_kernel, iterations=2)

    print("Inversione skeleton...")
    inverted = cv2.bitwise_not(skeleton_dilated)

    # Trova contorni nell'immagine invertita
    print("Ricerca contorni nelle aree chiuse...")
    contours, hierarchy = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Trovati {len(contours)} contorni totali")

    # Analizza ogni contorno per trovare i cicli validi
    h, w = skeleton.shape
    candidate_cycles = []

    for idx, contour in enumerate(contours):
        # Calcola area
        area = cv2.contourArea(contour)

        # Filtra per area minima
        if area < min_area:
            continue

        # Filtra per area massima
        if max_area is not None and area > max_area:
            continue

        # Crea maschera per questo contorno
        contour_mask = np.zeros_like(skeleton)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Verifica se tocca i bordi
        touches_border = (
            np.any(contour_mask[0, :] > 0) or np.any(contour_mask[-1, :] > 0) or
            np.any(contour_mask[:, 0] > 0) or np.any(contour_mask[:, -1] > 0)
        )

        if touches_border:
            continue

        # Calcola centroide usando moments
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroid = (cy, cx)  # (row, col) = (y, x)
        else:
            # Fallback: usa boundingRect
            x, y, w_box, h_box = cv2.boundingRect(contour)
            centroid = (y + h_box/2, x + w_box/2)

        # Questo è un ciclo candidato valido
        candidate_cycles.append({
            'contour': contour,
            'mask': contour_mask,
            'area': area,
            'centroid': centroid
        })

    print(f"Cicli candidati (dopo filtri area e bordi): {len(candidate_cycles)}")

    # Ordina per area (decrescente) per identificare il più grande
    candidate_cycles.sort(key=lambda x: x['area'], reverse=True)

    # Se richiesto, identifica ed escludi il ciclo più grande
    # MA SOLO se ci sono almeno 2 cicli (se c'è solo 1 ciclo, è valido!)
    start_idx = 0
    if exclude_largest and len(candidate_cycles) >= 2:
        largest = candidate_cycles[0]
        print(f"  Ciclo PIÙ GRANDE identificato: area={largest['area']:.0f} px (VERRÀ ESCLUSO)")
        start_idx = 1  # Salta il primo (più grande)
    elif exclude_largest and len(candidate_cycles) == 1:
        print(f"  Solo 1 ciclo trovato - NON viene escluso (è l'unico ciclo valido!)")

    # Crea maschere e raccogli statistiche
    filled_mask = np.zeros_like(skeleton)
    cycle_lines_mask = np.zeros_like(skeleton)
    centroids = []
    closed_areas = []

    closed_count = 0
    skipped_border = 0
    skipped_small = 0
    skipped_large = 0
    skipped_largest = 1 if (exclude_largest and len(candidate_cycles) >= 2) else 0

    # Processa i cicli validi (partendo da start_idx per escludere il più grande se necessario)
    for idx in range(start_idx, len(candidate_cycles)):
        cycle = candidate_cycles[idx]

        closed_count += 1

        print(f"  Ciclo chiuso {closed_count}: area={cycle['area']:.0f} px, centroid=({cycle['centroid'][0]:.1f}, {cycle['centroid'][1]:.1f})")

        # Se siamo in modalità debug, colora solo il ciclo specificato
        if debug_single_cycle is not None:
            if closed_count != debug_single_cycle:
                closed_areas.append(cycle['area'])
                centroids.append(cycle['centroid'])
                continue  # Salta questo ciclo
            else:
                print(f"  >>> QUESTO È IL CICLO DA COLORARE! <<<")

        # Aggiungi alla maschera riempita
        filled_mask = cv2.bitwise_or(filled_mask, cycle['mask'])

        # Trova le linee dello scheletro che circondano quest'area
        dilated = cv2.dilate(cycle['mask'], np.ones((3, 3), np.uint8), iterations=1)
        cycle_skeleton = cv2.bitwise_and(skeleton, dilated)
        cycle_lines_mask = cv2.bitwise_or(cycle_lines_mask, cycle_skeleton)

        closed_areas.append(cycle['area'])
        centroids.append(cycle['centroid'])

    # Conta pixel riempiti
    filled_pixels = np.sum(filled_mask > 0)
    total_pixels = filled_mask.size

    print(f"\nRisultati:")
    print(f"  Contorni totali trovati: {len(contours)}")
    print(f"  Cicli candidati (dopo filtri): {len(candidate_cycles)}")
    print(f"  Cicli chiusi accettati: {closed_count}")
    if exclude_largest and skipped_largest > 0:
        print(f"  Esclusi (ciclo più grande): {skipped_largest}")
    print(f"  Pixel riempiti: {filled_pixels:,} ({filled_pixels/total_pixels*100:.2f}% dell'immagine)")

    stats = {
        'total_regions': len(contours),
        'closed': closed_count,
        'open': len(contours) - closed_count,
        'avg_closed_area': np.mean(closed_areas) if closed_areas else 0,
        'min_closed_area': np.min(closed_areas) if closed_areas else 0,
        'max_closed_area': np.max(closed_areas) if closed_areas else 0
    }

    return filled_mask, cycle_lines_mask, centroids, stats


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


def visualize_skeleton(original, cleaned, skeleton, cycle_lines_mask, filled_mask, centroids, stats, output_dir, base_name, visualize_mode='fill', dot_radius=5):
    """Visualizza il risultato della scheletonizzazione e identificazione percorsi chiusi.

    Args:
        visualize_mode: 'fill' (riempi cicli in verde) o 'dot' (puntino rosso al centro)
        dot_radius: Raggio del puntino rosso in pixel (default: 5)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creazione visualizzazioni (modalità: {visualize_mode})...")

    # Scegli visualizzazione in base alla modalità
    if visualize_mode == 'dot':
        # Modalità PUNTINO ROSSO: disegna puntini rossi ai centroidi
        filled_visual = np.zeros((*original.shape, 3), dtype=np.uint8)
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

        print(f"  Disegno {len(centroids)} puntini rossi ai centroidi...")
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            # Disegna cerchio rosso pieno
            cv2.circle(filled_visual, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso
            cv2.circle(overlay, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso
    else:
        # Modalità RIEMPIMENTO VERDE (default)
        filled_visual = np.zeros((*original.shape, 3), dtype=np.uint8)
        filled_visual[filled_mask > 0] = [0, 255, 0]  # BGR: verde

        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        overlay[filled_mask > 0] = [0, 255, 0]  # BGR: verde

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

    # Visualizzazione cicli (riempimento o puntini)
    if visualize_mode == 'dot':
        viz_title = f'Puntini Rossi ai Centroidi ({stats["closed"]} cicli)'
        overlay_title = 'Overlay: Centroidi in Rosso'
    else:
        viz_title = f'Cicli Chiusi Riempiti ({stats["closed"]})'
        overlay_title = 'Overlay: Aree Cicli Chiusi in Verde'

    axes[1, 0].imshow(filled_visual)
    axes[1, 0].set_title(viz_title, fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Overlay su originale
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(overlay_title, fontsize=14, fontweight='bold')
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

    # Crea e salva immagine GRANDE con cicli visualizzati
    if visualize_mode == 'dot':
        print("\nCreazione immagine con puntini rossi...")
        cycles_image = np.zeros((*original.shape, 3), dtype=np.uint8)

        # Disegna puntini rossi
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            cv2.circle(cycles_image, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso

        # Salva come PNG
        cycles_path = output_dir / f"{base_name}_CICLI_PUNTINI_ROSSI.png"
        cv2.imwrite(str(cycles_path), cycles_image)
        print(f"\n*** IMMAGINE PUNTINI ROSSI salvata in: {cycles_path} ***")
        print(f"    Apri questo file per vedere {stats['closed']} puntini rossi ai centroidi dei cicli!")

        # Salva anche versione con overlay su originale
        overlay_large = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            cv2.circle(overlay_large, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso
        overlay_path = output_dir / f"{base_name}_OVERLAY_PUNTINI.png"
        cv2.imwrite(str(overlay_path), overlay_large)
        print(f"    OVERLAY puntini su originale salvato in: {overlay_path}")

        # Salva anche versione con pallini SULLO SKELETON (skeleton bianco + pallini rossi)
        skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            cv2.circle(skeleton_rgb, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso
        skeleton_dots_path = output_dir / f"{base_name}_SKELETON_PUNTINI.png"
        cv2.imwrite(str(skeleton_dots_path), skeleton_rgb)
        print(f"    SKELETON con puntini rossi salvato in: {skeleton_dots_path}")
        print(f"    *** CONTROLLA QUESTO FILE per vedere se i pallini sono sui cicli giusti! ***")
    else:
        print("\nCreazione immagine cicli verdi...")
        cycles_image = np.zeros((*original.shape, 3), dtype=np.uint8)
        cycles_image[filled_mask > 0] = [0, 255, 0]  # BGR: verde

        # Salva come PNG
        cycles_path = output_dir / f"{base_name}_CICLI_VERDI.png"
        cv2.imwrite(str(cycles_path), cycles_image)
        print(f"\n*** IMMAGINE CICLI VERDI salvata in: {cycles_path} ***")
        print(f"    Apri questo file per vedere i {stats['closed']} cicli riempiti di verde!")

        # Salva anche versione con overlay su originale
        overlay_large = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        overlay_large[filled_mask > 0] = [0, 255, 0]  # BGR: verde
        overlay_path = output_dir / f"{base_name}_OVERLAY_CICLI.png"
        cv2.imwrite(str(overlay_path), overlay_large)
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
        default=0,
        help='Larghezza del bordo da escludere dall\'analisi (default: 0=disabilitato)'
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
        default=None,
        help='Area massima per considerare un ciclo chiuso (default: None=disabilitato). Esclude regioni troppo grandi.'
    )
    parser.add_argument(
        '--no-exclude-largest',
        action='store_true',
        help='NON escludere automaticamente il ciclo più grande (default: False, ovvero esclude sempre il più grande)'
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
        default=0,
        help='Dimensione minima componenti scheletro da mantenere (default: 0=disabilitato). Rimuove frammenti isolati sconnessi.'
    )
    parser.add_argument(
        '--debug-single-cycle',
        type=int,
        default=None,
        help='DEBUG: Colora solo il ciclo numero N (1-based). Utile per verificare un singolo ciclo.'
    )
    parser.add_argument(
        '--visualize-mode',
        type=str,
        choices=['fill', 'dot'],
        default='dot',
        help='Modalità visualizzazione: "fill" (riempi cicli in verde) o "dot" (puntino rosso al centro) (default: dot)'
    )
    parser.add_argument(
        '--dot-radius',
        type=int,
        default=5,
        help='Raggio del puntino rosso in pixel (solo per --visualize-mode=dot) (default: 5)'
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
    filled_mask, cycle_lines_mask, centroids, stats = identify_closed_contours(
        skeleton,
        min_area=args.min_cycle_area,
        max_area=args.max_cycle_area,
        exclude_largest=not args.no_exclude_largest,  # Escludi il più grande per default
        debug_single_cycle=args.debug_single_cycle
    )

    # Salva risultati
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    # Visualizza
    print("\nCreazione visualizzazioni...")
    visualize_skeleton(
        original, cleaned, skeleton, cycle_lines_mask, filled_mask, centroids, stats,
        args.output, base_name,
        visualize_mode=args.visualize_mode,
        dot_radius=args.dot_radius
    )

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
