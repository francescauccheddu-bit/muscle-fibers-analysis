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


def find_skeleton_endpoints(skeleton):
    """
    Trova gli endpoint dello skeleton (pixel con esattamente 1 vicino).

    Args:
        skeleton: Skeleton binario (uint8)

    Returns:
        Lista di coordinate (y, x) degli endpoint
    """
    # Kernel per contare i vicini (8-connectivity)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Non contare il pixel centrale

    # Conta i vicini per ogni pixel
    skeleton_bool = (skeleton > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(skeleton_bool, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # Endpoint = pixel dello skeleton con esattamente 1 vicino
    endpoints_mask = np.logical_and(skeleton_bool == 1, neighbor_count == 1)

    # Estrai coordinate
    endpoints = np.argwhere(endpoints_mask)  # Restituisce array di (y, x)

    return endpoints


def connect_skeleton_endpoints(skeleton, max_distance=30, max_angle_deg=45):
    """
    Connette gli endpoint dello skeleton se sono vicini e l'angolo è ragionevole.

    Args:
        skeleton: Skeleton binario (uint8)
        max_distance: Distanza massima in pixel per connettere endpoint (default: 30)
        max_angle_deg: Angolo massimo in gradi dalla direzione della linea (default: 45)

    Returns:
        Skeleton con endpoint connessi
    """
    print(f"Ricerca endpoint dello skeleton...")
    endpoints = find_skeleton_endpoints(skeleton)
    print(f"  Trovati {len(endpoints)} endpoint")

    if len(endpoints) == 0:
        return skeleton

    print(f"Connessione endpoint entro {max_distance} pixel...")
    skeleton_connected = skeleton.copy()
    connections_made = 0

    # Per ogni endpoint, cerca altri endpoint vicini
    for i, (y1, x1) in enumerate(endpoints):
        # Mostra progresso ogni 100 endpoint
        if (i + 1) % 100 == 0:
            print(f"  Processati {i + 1}/{len(endpoints)} endpoint...")

        # Calcola direzione della linea all'endpoint (guarda 5 pixel indietro)
        direction_vec = get_skeleton_direction_at_point(skeleton, y1, x1, look_back=5)

        for j in range(i + 1, len(endpoints)):
            y2, x2 = endpoints[j]

            # Calcola distanza
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if distance > max_distance:
                continue

            # Vettore di connessione
            connection_vec = np.array([x2 - x1, y2 - y1])
            connection_vec = connection_vec / (np.linalg.norm(connection_vec) + 1e-10)

            # Verifica angolo solo se abbiamo una direzione valida
            if direction_vec is not None:
                # Calcola angolo tra direzione della linea e connessione
                dot_product = np.dot(direction_vec, connection_vec)
                angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)

                # L'angolo dovrebbe essere vicino a 0° (stessa direzione) o 180° (direzione opposta)
                angle_diff = min(angle_deg, 180 - angle_deg)

                if angle_diff > max_angle_deg:
                    continue  # Angolo troppo grande, skip

            # Connetti i due endpoint con una linea
            cv2.line(skeleton_connected, (x1, y1), (x2, y2), 255, thickness=1)
            connections_made += 1

    print(f"  Connessioni create: {connections_made}")
    return skeleton_connected


def get_skeleton_direction_at_point(skeleton, y, x, look_back=5):
    """
    Stima la direzione della linea skeleton ad un punto guardando indietro di alcuni pixel.

    Args:
        skeleton: Skeleton binario
        y, x: Coordinate del punto
        look_back: Numero di pixel da guardare indietro

    Returns:
        Vettore direzione normalizzato [dx, dy] o None se non trovato
    """
    # Trova il path dello skeleton partendo dal punto
    visited = set()
    current = (y, x)
    path = [current]

    for _ in range(look_back):
        visited.add(current)
        cy, cx = current

        # Cerca il prossimo pixel vicino
        found_next = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = cy + dy, cx + dx

                # Controlla bounds
                if ny < 0 or ny >= skeleton.shape[0] or nx < 0 or nx >= skeleton.shape[1]:
                    continue

                # Se è un pixel dello skeleton e non è stato visitato
                if skeleton[ny, nx] > 0 and (ny, nx) not in visited:
                    path.append((ny, nx))
                    current = (ny, nx)
                    found_next = True
                    break

            if found_next:
                break

        if not found_next:
            break

    # Se abbiamo almeno 2 punti, calcola direzione
    if len(path) >= 2:
        start = path[0]
        end = path[-1]
        direction = np.array([end[1] - start[1], end[0] - start[0]], dtype=float)
        norm = np.linalg.norm(direction)
        if norm > 0:
            return direction / norm

    return None


def identify_closed_contours_from_mask(original_mask, cleaned_mask, skeleton, min_area=100, max_area=None, exclude_largest=True, closing_size=3, connect_endpoints=0, debug_single_cycle=None):
    """
    Identifica i percorsi chiusi (cicli) dalla MASCHERA PULITA con closing opzionale.

    DOPPIO APPROCCIO:
    1. Se connect_endpoints > 0: Connetti endpoint dello skeleton (veloce, intelligente)
    2. Altrimenti: Closing morfologico sulla maschera (più lento ma completo)

    Args:
        original_mask: Maschera originale (non processata)
        cleaned_mask: Maschera pulita (su cui applichiamo il closing se usato)
        skeleton: Scheletro binario (su cui connettiamo endpoint se usato)
        min_area: Area minima per considerare una regione chiusa
        max_area: Area massima per considerare una regione chiusa (None = nessun limite)
        exclude_largest: Se True, esclude automaticamente il ciclo più grande (sfondo) (default: True)
        closing_size: Dimensione kernel per chiusura morfologica sulla maschera (default: 3, 0=disabilitato)
        connect_endpoints: Se > 0, usa endpoint connection invece di closing. Valore = distanza max in pixel (default: 0=disabilitato)
        debug_single_cycle: Se specificato, colora solo questo ciclo (1-based)

    Returns:
        filled_mask: Maschera con cicli chiusi riempiti
        centroids: Lista di coordinate (y, x) dei centroidi dei cicli chiusi
        contours_list: Lista dei contorni (cv2 contours) dei cicli identificati
        stats: Statistiche sui percorsi
    """
    if debug_single_cycle is not None:
        print(f"*** MODALITA DEBUG: Colorerò SOLO il ciclo #{debug_single_cycle} ***")

    # DOPPIO APPROCCIO: Endpoint connection O closing morfologico
    if connect_endpoints > 0:
        # METODO 1: Connessione endpoint (veloce, intelligente, per gap grandi)
        print(f"METODO: Connessione endpoint dello skeleton (distanza max: {connect_endpoints} px)")

        # Connetti gli endpoint dello skeleton
        skeleton_connected = connect_skeleton_endpoints(skeleton, max_distance=connect_endpoints, max_angle_deg=45)

        # Riempi i buchi nello skeleton connesso
        print("Riempimento buchi nello skeleton connesso...")
        skeleton_bool = skeleton_connected > 0
        filled = ndi.binary_fill_holes(skeleton_bool).astype(np.uint8) * 255

        # Sottrai la maschera pulita per ottenere solo i cicli
        print("Sottrazione per ottenere solo i cicli riempiti...")
        cycles_only = cv2.subtract(filled, cleaned_mask)

    else:
        # METODO 2: Closing morfologico sulla maschera (completo ma più lento)
        print(f"METODO: Closing morfologico sulla maschera")

        # Applica closing sulla maschera se richiesto
        if closing_size > 0:
            print(f"  Chiusura gap sulla maschera (kernel={closing_size}x{closing_size})...")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
            mask_closed = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            n_pixels_added = np.sum(mask_closed > 0) - np.sum(cleaned_mask > 0)
            print(f"  Pixel aggiunti dal closing sulla maschera: {n_pixels_added}")
        else:
            mask_closed = cleaned_mask

        print("Riempimento buchi nella maschera...")
        mask_bool = mask_closed > 0
        filled = ndi.binary_fill_holes(mask_bool).astype(np.uint8) * 255

        print("Sottrazione per ottenere solo i cicli riempiti...")
        cycles_only = cv2.subtract(filled, cleaned_mask)

    print("Ricerca contorni nei cicli...")
    contours, hierarchy = cv2.findContours(cycles_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Trovati {len(contours)} cicli (regioni riempite)")

    # Analizza ogni contorno per trovare i cicli validi
    h, w = skeleton.shape
    candidate_cycles = []

    # OTTIMIZZAZIONE: Riutilizza la stessa maschera invece di crearne migliaia
    # Usa uint8 perché cv2.drawContours richiede uint8
    contour_mask = np.zeros((h, w), dtype=np.uint8)

    for idx, contour in enumerate(contours):
        # OTTIMIZZAZIONE: Mostra progresso ogni 1000 cicli invece di 500
        if (idx + 1) % 1000 == 0:
            print(f"  Analizzati {idx + 1}/{len(contours)} cicli...")

        # Calcola area
        area = cv2.contourArea(contour)

        # Filtra per area minima
        if area < min_area:
            continue

        # Filtra per area massima
        if max_area is not None and area > max_area:
            continue

        # Verifica se tocca i bordi usando bounding box (più veloce!)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        touches_border = (x == 0 or y == 0 or (x + w_box) >= w or (y + h_box) >= h)

        if touches_border:
            continue

        # Calcola centroide usando moments
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroid = (cy, cx)  # (row, col) = (y, x)
        else:
            centroid = (y + h_box/2, x + w_box/2)

        # Questo è un ciclo candidato valido - MA NON salviamo la maschera per risparmiare memoria!
        candidate_cycles.append({
            'contour': contour,
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
    centroids = []  # Lista di coordinate (y, x)
    contours_list = []  # Lista dei contorni per disegnare i bordi
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

        # OTTIMIZZAZIONE: Riduci print statements per velocizzare
        # Mostra progresso ogni 1000 cicli invece di 500
        if closed_count % 1000 == 0:
            print(f"  Processati {closed_count}/{len(candidate_cycles) - start_idx} cicli...")

        # Mostra dettagli solo per primi 5 cicli e multipli di 1000
        if closed_count % 1000 == 0 or closed_count <= 5:
            print(f"  Ciclo chiuso {closed_count}: area={cycle['area']:.0f} px, centroid=({cycle['centroid'][0]:.1f}, {cycle['centroid'][1]:.1f})")

        # Se siamo in modalità debug, colora solo il ciclo specificato
        if debug_single_cycle is not None:
            if closed_count != debug_single_cycle:
                closed_areas.append(cycle['area'])
                centroids.append(cycle['centroid'])
                contours_list.append(cycle['contour'])
                continue  # Salta questo ciclo
            else:
                print(f"  >>> QUESTO È IL CICLO DA COLORARE! <<<")

        # Ricrea la maschera al volo (per risparmio memoria)
        contour_mask.fill(0)  # Pulisci array esistente
        cv2.drawContours(contour_mask, [cycle['contour']], -1, 255, thickness=cv2.FILLED)

        # Aggiungi alla maschera riempita
        filled_mask = cv2.bitwise_or(filled_mask, contour_mask)

        # Salva centroid e contorno
        closed_areas.append(cycle['area'])
        centroids.append(cycle['centroid'])
        contours_list.append(cycle['contour'])

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

    return filled_mask, centroids, contours_list, closed_areas, stats


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


def visualize_skeleton(original, cleaned, skeleton, filled_mask, centroids, contours_list, closed_areas, stats, output_dir, base_name, closing_size=3, visualize_mode='fill', dot_radius=10):
    """Visualizza il risultato della scheletonizzazione e identificazione percorsi chiusi.

    Args:
        contours_list: Lista dei contorni dei cicli per disegnare i bordi
        closing_size: Dimensione kernel del morphological closing (usato nel nome file)
        visualize_mode: 'fill' (riempi cicli in verde) o 'dot' (puntino rosso al centro)
        dot_radius: Raggio del puntino rosso in pixel (default: 10)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creazione visualizzazioni (modalità: {visualize_mode})...")

    if visualize_mode == 'dot':
        # Modalità PUNTINO: disegna puntini rossi ai centroidi

        print(f"  Disegno {len(centroids)} puntini rossi...")

        # Suffisso per nome file con closing_size
        closing_suffix = f"_closing{closing_size}" if closing_size > 0 else ""

        # 1. SKELETON con puntini rossi
        skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            cv2.circle(skeleton_rgb, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso

        skeleton_dots_path = output_dir / f"{base_name}_SKELETON_PUNTINI{closing_suffix}.png"
        cv2.imwrite(str(skeleton_dots_path), skeleton_rgb)
        print(f"  Salvato: {skeleton_dots_path}")

        # 2. MASCHERA ORIGINALE con bordi cicli rossi e puntini rossi
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

        # Disegna bordi dei cicli in rosso
        if len(contours_list) > 0:
            print(f"  Disegno {len(contours_list)} bordi cicli in rosso...")
            cv2.drawContours(original_rgb, contours_list, -1, (0, 0, 255), 2)  # BGR: rosso, thickness=2

        # Aggiungi pallini rossi ai centroidi DOPO i bordi
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            cv2.circle(original_rgb, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso

        overlay_path = output_dir / f"{base_name}_OVERLAY_PUNTINI{closing_suffix}.png"
        cv2.imwrite(str(overlay_path), original_rgb)
        print(f"  Salvato: {overlay_path}")

        # 3. SKELETON RIMPOLPATO (thick) - solo puntini, senza bordi
        print(f"  Creazione skeleton rimpolpato (thick)...")
        # Dilata lo skeleton per renderlo MOLTO più spesso, simile alla maschera originale
        kernel_thick = np.ones((5, 5), np.uint8)
        skeleton_thick = cv2.dilate(skeleton, kernel_thick, iterations=5)  # Aumentate iterazioni
        skeleton_thick_rgb = cv2.cvtColor(skeleton_thick, cv2.COLOR_GRAY2RGB)

        # Aggiungi SOLO pallini rossi ai centroidi (NO bordi)
        for centroid in centroids:
            cy, cx = int(centroid[0]), int(centroid[1])
            cv2.circle(skeleton_thick_rgb, (cx, cy), dot_radius, (0, 0, 255), -1)  # BGR: rosso

        skeleton_thick_path = output_dir / f"{base_name}_SKELETON_THICK{closing_suffix}.png"
        cv2.imwrite(str(skeleton_thick_path), skeleton_thick_rgb)
        print(f"  Salvato: {skeleton_thick_path}")

        # 4. ISTOGRAMMA distribuzione aree
        if len(closed_areas) > 0:
            print(f"  Creazione istogramma distribuzione aree...")
            fig, ax = plt.subplots(figsize=(12, 6))

            # Istogramma con bins automatici
            n, bins, patches = ax.hist(closed_areas, bins=50, edgecolor='black', alpha=0.7)

            ax.set_xlabel('Area (pixel)', fontsize=12)
            ax.set_ylabel('Frequenza (numero di cicli)', fontsize=12)

            # Titolo con info sul closing
            title = f'Distribuzione Aree dei Cicli (n={len(closed_areas)})'
            if closing_size > 0:
                title += f' - Closing size: {closing_size}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Aggiungi statistiche
            stats_text = f'Media: {stats["avg_closed_area"]:.0f} px\n'
            stats_text += f'Min: {stats["min_closed_area"]:.0f} px\n'
            stats_text += f'Max: {stats["max_closed_area"]:.0f} px'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            histogram_path = output_dir / f"{base_name}_ISTOGRAMMA_AREE{closing_suffix}.png"
            plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
            print(f"  Salvato: {histogram_path}")
            plt.close()
    else:
        # Modalità RIEMPIMENTO VERDE (legacy)
        filled_visual = np.zeros((*original.shape, 3), dtype=np.uint8)
        filled_visual[filled_mask > 0] = [0, 255, 0]  # BGR: verde

        cycles_path = output_dir / f"{base_name}_CICLI_VERDI.png"
        cv2.imwrite(str(cycles_path), filled_visual)
        print(f"  Salvato: {cycles_path}")



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
        '--closing-size',
        type=int,
        default=3,
        help='Dimensione kernel per chiudere gap nei cicli aperti (default: 3). Chiude piccoli gap di pochi pixel nei contorni circolari. 0=disabilitato.'
    )
    parser.add_argument(
        '--connect-endpoints',
        type=int,
        default=0,
        help='NUOVO: Connetti endpoint dello skeleton entro N pixel (default: 0=disabilitato). Metodo veloce e intelligente per chiudere gap grandi (20-30px). Se specificato, sovrascrive --closing-size.'
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
        default=10,
        help='Raggio del puntino rosso in pixel (solo per --visualize-mode=dot) (default: 10)'
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
    filled_mask, centroids, contours_list, closed_areas, stats = identify_closed_contours_from_mask(
        original,  # Maschera originale per identificare pixel completamente nuovi
        cleaned,   # Maschera pulita (dopo remove_small_objects)
        skeleton,  # Passa skeleton per visualizzazione
        min_area=args.min_cycle_area,
        max_area=args.max_cycle_area,
        exclude_largest=not args.no_exclude_largest,  # Escludi il più grande per default
        closing_size=args.closing_size,  # Chiude gap piccoli nei cicli
        connect_endpoints=args.connect_endpoints,  # NUOVO: Connetti endpoint dello skeleton
        debug_single_cycle=args.debug_single_cycle
    )

    # Salva risultati
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input).stem

    # Visualizza
    print("\nCreazione visualizzazioni...")
    visualize_skeleton(
        original, cleaned, skeleton, filled_mask, centroids, contours_list, closed_areas, stats,
        args.output, base_name,
        closing_size=args.closing_size,
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
