# -*- coding: utf-8 -*-
"""
Auteur  : Djokhar BETELGUERIEV
Date    : 2025-03-18
Description : Brève description du programme.
"""

import matplotlib.pyplot as plt
import copy
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
from scipy.optimize import curve_fit
from matplotlib.widgets import Button, RectangleSelector

drct = "C:/Program Files (x86)/WaveCatcher_64ch/Run_Data/"
samples = 1024
fixed_header = 0x186
max_events_per_file = 100000


class InteractiveGaussFit:
    def __init__(self, x, y, ax):
        self.x = x
        self.y = y
        self.ax = ax  # Référence à l'axe spécifique pour ce canal
        self.fits = []

        # Initialiser les attributs xmin et xmax aux limites des données
        if len(x) > 0:
            self.xmin = min(x)
            self.xmax = max(x)
        else:
            self.xmin = 0
            self.xmax = 1

        # Créer le sélecteur rectangle spécifique à cet axe
        self.rect_selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)
        )

        # Positionner les boutons spécifiques à cet axe
        pos = self.ax.get_position()
        button_width = 0.08
        button_height = 0.03
        button_y = pos.y0 + 0.01

        # Positions pour les boutons - ajuster pour chaque axe
        self.button_ax = plt.axes([pos.x1 - button_width - 0.01, button_y, button_width, button_height])
        self.clear_button_ax = plt.axes([pos.x1 - 2 * button_width - 0.02, button_y, button_width, button_height])

        # Créer les boutons spécifiques à cet axe
        self.button = Button(self.button_ax, 'Add Fit')
        self.button.on_clicked(self.add_fit)

        self.clear_button = Button(self.clear_button_ax, 'Clear Fits')
        self.clear_button.on_clicked(self.clear_fits)

    def gaussian(self, x, amp, mu, sigma):
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def on_select(self, eclick, erelease):
        """Appelé lorsqu'une région est sélectionnée"""
        # Récupérer les coordonnées des coins
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # S'assurer que les coordonnées sont valides
        if x1 is None or x2 is None:
            return

        # Mettre à jour les limites de la sélection
        self.xmin, self.xmax = min(x1, x2), max(x1, x2)

        # Vérifier immédiatement si la sélection est valide
        mask = (self.x >= self.xmin) & (self.x <= self.xmax)
        num_points = sum(mask)

        # Informer l'utilisateur du nombre de points sélectionnés
        if num_points < 3:
            print(f"Attention: Seulement {num_points} points dans la sélection. Élargissez votre sélection.")
        else:
            print(f"Sélection valide: {num_points} points.")

    def add_fit(self, event):
        """Ajoute un ajustement gaussien à la région sélectionnée"""
        # Vérifier que xmin et xmax sont définis et différents
        if not hasattr(self, 'xmin') or not hasattr(self, 'xmax') or self.xmin == self.xmax:
            # Si aucune sélection n'a été faite, utiliser toutes les données
            self.xmin = min(self.x)
            self.xmax = max(self.x)
            print("Utilisation de toutes les données pour l'ajustement")

        # Vérifier que nous avons des données
        if len(self.x) == 0 or len(self.y) == 0:
            print("Pas de données disponibles pour l'ajustement")
            return

        try:
            # Créer le masque pour sélectionner les points dans la région
            mask = (self.x >= self.xmin) & (self.x <= self.xmax)

            # Vérifier que nous avons suffisamment de points (au moins 3 pour un ajustement gaussien)
            num_points = sum(mask)
            if num_points < 3:
                print(f"Pas assez de points dans la sélection ({num_points}). Minimum requis: 3")
                print("Utilisation de toutes les données disponibles à la place")
                x_fit = self.x
                y_fit = self.y
            else:
                x_fit = self.x[mask]
                y_fit = self.y[mask]
                print(f"Ajustement avec {num_points} points")

            # Paramètres initiaux pour l'ajustement
            max_y_index = np.argmax(y_fit)
            max_y = y_fit[max_y_index]
            mean_x = x_fit[max_y_index]  # Utiliser le x correspondant au maximum de y

            # Estimer sigma à partir de la largeur à mi-hauteur (FWHM)
            half_height = max_y / 2
            above_half = y_fit >= half_height
            if np.sum(above_half) > 1:
                # Trouver les points les plus à gauche et à droite au-dessus de la mi-hauteur
                x_above = x_fit[above_half]
                fwhm = np.max(x_above) - np.min(x_above)
                # sigma = FWHM / (2 * sqrt(2 * ln(2)))
                sigma = fwhm / 2.355
            else:
                # Utiliser un sigma par défaut si nous ne pouvons pas estimer le FWHM
                sigma = (np.max(x_fit) - np.min(x_fit)) / 10

            # Paramètres initiaux
            p0 = [max_y, mean_x, sigma]

            # Essayer l'ajustement
            popt, pcov = curve_fit(self.gaussian, x_fit, y_fit, p0=p0)

            # Calculer la courbe ajustée pour l'ensemble des données
            fit_y = self.gaussian(self.x, *popt)

            # Tracer la courbe avec une étiquette formatée
            label = f'Fit μ={popt[1]:.2f}, σ={popt[2]:.2f}, A={popt[0]:.2f}'
            line, = self.ax.plot(self.x, fit_y, '--', label=label)

            # Stocker la référence à la ligne
            self.fits.append(line)

            # Mettre à jour la légende
            self.ax.legend()

            # Rafraîchir l'affichage
            plt.draw()

        except Exception as e:
            print(f"Erreur lors de l'ajustement: {e}")
            print("Le fitting n'a pas convergé. Essayez une autre région.")

    def clear_fits(self, event):
        """Supprime tous les ajustements du graphique"""
        # Supprimer toutes les courbes d'ajustement
        for line in self.fits:
            line.remove()

        # Réinitialiser la liste
        self.fits = []

        # Mettre à jour la légende
        self.ax.legend()

        # Rafraîchir l'affichage
        plt.draw()


def ret_4bytes(rawdata, k):
    if k >= 0 and k + 3 < len(rawdata):
        return rawdata[k] + (rawdata[k + 1] << 8) + (rawdata[k + 2] << 12) + (rawdata[k + 3] << 16)
    else:
        return None


def read_single_wave(rawdata, idx):
    wtmp = []
    for i in range(samples):
        val = rawdata[idx] | (rawdata[idx + 1] << 8)
        # Vérifier si le bit de poids fort est à 1 (nombre négatif)
        if val & 0x8000:
            # Conversion en complément à 2 pour obtenir la valeur négative
            val = -((val ^ 0xFFFF) + 1)
        wtmp.append(val)
        idx += 2
    return idx, wtmp




def read_event_fullwave(rawdata, idx, amp, prev_timestamp=None, header_offset=None, expected_channels=None):
    """
    Lit un événement dans les données binaires.
    Returns None si erreur,
    Returns (idx, trigger, amplitudes, NumberOfChannel, hit_counts, rates, current_timestamp, EventNumber, False) si succès,
    Returns (None, None, None, None, None, None, None, None, True) si fin de fichier naturelle
    """
    # Si l'index est au-delà de la fin du fichier ou très proche,
    # c'est probablement une fin de fichier naturelle
    if idx is None or idx >= len(rawdata) - 100:
        return (None, None, None, None, None, None, None, None, True)

    if rawdata is None:
        return None

    try:
        # Lecture des métadonnées de l'événement
        EventNumber = ret_4bytes(rawdata, idx + 0)
        Year = ret_4bytes(rawdata, idx + 12)
        Month = ret_4bytes(rawdata, idx + 16)
        Day = ret_4bytes(rawdata, idx + 20)
        Hour = ret_4bytes(rawdata, idx + 24)
        Minute = ret_4bytes(rawdata, idx + 28)
        Second = ret_4bytes(rawdata, idx + 32)
        Millisecond = ret_4bytes(rawdata, idx + 36)
        NumberOfChannel = ret_4bytes(rawdata, idx + 48)
        channel = ret_4bytes(rawdata, idx + 52)

        # Vérification précoce des valeurs de date/heure pour détecter des valeurs aberrantes
        if Year is not None and Month is not None and Day is not None and Hour is not None and Minute is not None and Second is not None:
            if Month <= 0 or Month > 12 or Day <= 0 or Day > 31 or Hour < 0 or Hour > 23 or Minute < 0 or Minute > 59 or Second < 0 or Second > 59:
                print(
                    f"ALERTE: Valeurs de date/heure invalides: {Day}/{Month}/{Year} - {Hour}:{Minute}:{Second}.{Millisecond}")
                return None

        # Vérification et correction du nombre de canaux pour le premier décodage
        if NumberOfChannel is None or NumberOfChannel <= 0 or NumberOfChannel > 10:
            found_valid = False
            # Liste des décalages à essayer pour l'en-tête
            for test_offset in [0, 4, -4, 8, -8, 12, -12]:
                test_idx = idx + test_offset
                test_number_of_channels = ret_4bytes(rawdata, test_idx + 48)

                # Vérifier si ce décalage donne un nombre de canaux valide
                if test_number_of_channels is not None and 0 < test_number_of_channels <= 3:
                    # Vérifier si c'est cohérent avec le nombre de canaux attendu
                    if expected_channels is None or test_number_of_channels == expected_channels:
                        # Mettre à jour les valeurs avec le nouveau décalage
                        idx = test_idx
                        EventNumber = ret_4bytes(rawdata, idx + 0)
                        Year = ret_4bytes(rawdata, idx + 12)
                        Month = ret_4bytes(rawdata, idx + 16)
                        Day = ret_4bytes(rawdata, idx + 20)
                        Hour = ret_4bytes(rawdata, idx + 24)
                        Minute = ret_4bytes(rawdata, idx + 28)
                        Second = ret_4bytes(rawdata, idx + 32)
                        Millisecond = ret_4bytes(rawdata, idx + 36)
                        NumberOfChannel = test_number_of_channels
                        channel = ret_4bytes(rawdata, idx + 52)
                        found_valid = True
                        break

            # Si aucun décalage valide n'est trouvé
            if not found_valid:
                print("Aucun décalage valide trouvé pour le nombre de canaux")
                return None

        # Vérifier la cohérence avec le nombre de canaux attendu (si spécifié)
        if expected_channels is not None and NumberOfChannel != expected_channels:
            print(f"ALERTE: Nombre de canaux incohérent: attendu {expected_channels}, trouvé {NumberOfChannel}")

            # Essayons de trouver un meilleur index qui donne le bon nombre de canaux
            for test_idx_offset in range(-100, 100, 4):  # Essayer des décalages dans une plage de ±100 octets
                test_idx = idx + test_idx_offset
                if test_idx < 0 or test_idx >= len(rawdata) - 60:
                    continue

                test_number_of_channels = ret_4bytes(rawdata, test_idx + 48)

                if test_number_of_channels == expected_channels:
                    print(f"Index corrigé trouvé à {test_idx_offset:+d} octets du point initial")
                    # Mettre à jour toutes les valeurs avec le nouveau décalage
                    idx = test_idx
                    EventNumber = ret_4bytes(rawdata, idx + 0)
                    Year = ret_4bytes(rawdata, idx + 12)
                    Month = ret_4bytes(rawdata, idx + 16)
                    Day = ret_4bytes(rawdata, idx + 20)
                    Hour = ret_4bytes(rawdata, idx + 24)
                    Minute = ret_4bytes(rawdata, idx + 28)
                    Second = ret_4bytes(rawdata, idx + 32)
                    Millisecond = ret_4bytes(rawdata, idx + 36)
                    NumberOfChannel = test_number_of_channels
                    channel = ret_4bytes(rawdata, idx + 52)
                    break

            # Vérifier si nous avons réussi à corriger
            if NumberOfChannel != expected_channels:
                print(f"ERREUR: Impossible de trouver un index cohérent pour le nombre de canaux attendu")
                return None

        # Validation des données essentielles
        if EventNumber is None or NumberOfChannel is None or NumberOfChannel <= 0:
            print(f"Erreur: Données d'en-tête incomplètes ou invalides à l'index 0x{idx:x}")
            return None

        if Day is None or Hour is None or Minute is None or Second is None or Millisecond is None:
            print(f"Erreur: Données de timestamp incomplètes à l'index 0x{idx:x}")
            return None

        # Calcul du timestamp
        current_timestamp = ((Day * 24 + Hour) * 3600 + Minute * 60 + Second) * 1000 + Millisecond
        print(
            f'EventNumber : {EventNumber}, {NumberOfChannel} channels, {current_timestamp} - {Day}/{Month}/{Year} - {Hour}:{Minute}:{Second}.{Millisecond} - {channel}')

        # Si c'est le premier appel, déterminer le décalage approprié entre l'en-tête et les données
        if header_offset is None:
            # Essayer différents décalages pour trouver le bon format
            possible_offsets = [52, 56, 60, 64, 48, 68]
            found_offset = False

            for offset in possible_offsets:
                test_idx = idx + offset
                if test_idx + 28 < len(rawdata):  # Vérifier qu'on peut lire le hit_count
                    hit_count = ret_4bytes(rawdata, test_idx + 28)
                    if hit_count is not None and hit_count >= 0:
                        print(f"Décalage trouvé: {offset} octets")
                        header_offset = offset
                        found_offset = True
                        break

            if not found_offset:
                print("Impossible de déterminer le décalage approprié, utilisation de la valeur par défaut (56)")
                header_offset = 56

        # Appliquer le décalage pour accéder aux données des canaux
        idx += header_offset

        # Initialiser les structures pour les données des canaux
        trigger = [False] * NumberOfChannel
        amplitudes = [[] for _ in range(NumberOfChannel)]
        hit_counts = [0] * NumberOfChannel
        rates = [0] * NumberOfChannel

        # Lire les données pour chaque canal
        for i in range(NumberOfChannel):
            # Lire le hit_count
            hit_count = ret_4bytes(rawdata, idx + 28)
            if hit_count is None:
                print(f"Erreur: Impossible de lire le hit_count pour le canal {i}")
                return None

            # Mettre à jour hit_counts et rates
            hit_counts[i] = hit_count
            rates[i] = hit_count / (2.56 * 10 ** (-6)) if hit_count is not None else 0

            # Avancer l'index pour la forme d'onde
            idx += 36

            # Vérifier qu'on a assez de données pour la forme d'onde
            if idx + 2 * samples > len(rawdata):
                print(f"Erreur: Pas assez de données pour la forme d'onde (idx={idx}, canal {i})")
                return None

            # Lire la forme d'onde
            idx, wtmp = read_single_wave(rawdata, idx)

            # Déterminer si le canal a été déclenché
            trigger[i] = (hit_count > 0)
            amplitudes[i] = wtmp

        # Retourner - avec un flag False pour indiquer que ce n'est pas une fin de fichier
        return idx, trigger, amplitudes, NumberOfChannel, hit_counts, rates, current_timestamp, EventNumber, False

    except Exception as e:
        print(f"Exception lors de la lecture de l'événement à l'index 0x{idx:x}: {str(e)}")
        return None


def load_and_process_single_file_with_minimums(filename, base_dir):
    """
    Charge et traite un fichier binaire en extrayant aussi les valeurs minimales.
    Vérifie également la séquence des numéros d'événements.
    """
    print(f"Traitement du fichier: {filename}")

    # Référence à la variable globale fixed_header
    global fixed_header
    successful_offset = None  # Variable pour stocker le décalage qui a réussi

    # Ouvrir et lire le fichier sélectionné
    with open(os.path.join(base_dir, filename), mode='rb') as file:
        data = file.read()

    # Trouver le premier 0a 01 dans le fichier
    idx = find_first_0a01_header(data)
    if idx is None:
        print(f"Utilisation de la valeur par défaut pour fixed header: 0x{fixed_header:x}")
        idx = fixed_header
    else:
        # Mettre à jour la variable globale fixed_header
        fixed_header = idx

    # Liste des décalages possibles à essayer
    possible_offsets = [52, 56, 60, 64, 48, 68]

    print(f"Liste des décalages à essayer: {possible_offsets}")

    # Variable pour stocker la meilleure tentative
    best_attempt = {
        "offset": None,
        "amplitude": None,
        "num_channels": 0,
        "event_minimums": [],
        "events_read": 0,
        "raw_waveforms": []  # Pour stocker les formes d'onde brutes
    }

    # Essayer chaque décalage jusqu'à ce qu'un marche correctement
    for current_offset in possible_offsets:
        print(f"\n===== Essai avec header_offset = {current_offset} octets =====")

        amplitude = None
        num_channels = None
        event_minimums = []  # Liste pour stocker les minimums par événement
        raw_waveforms = []  # Pour stocker les formes d'onde brutes
        current_idx = idx  # Réinitialiser l'index de départ
        previous_event_number = None  # Pour vérifier la continuité des événements
        sequence_error = False  # Pour détecter les erreurs de séquence
        events_read = 0  # Compteur d'événements lus

        # Lire tous les événements du fichier avec ce décalage
        for i in range(max_events_per_file):
            # Utiliser le décalage courant sans détection automatique
            readed_event = read_event_fullwave(data, current_idx, [], None, current_offset)

            if readed_event is None:
                # C'est une erreur de lecture (pas une fin de fichier naturelle)
                if i == 0:
                    print(f"Échec de lecture du premier événement avec header_offset = {current_offset}")
                else:
                    print(f"Erreur de lecture après {events_read} événements")
                sequence_error = True  # Marquer comme erreur pour essayer le prochain décalage
                break

            # Vérifier si c'est une fin de fichier naturelle
            if len(readed_event) == 9 and readed_event[8] is True:
                print(f"Fin naturelle du fichier atteinte après {events_read} événements")
                # Si on a lu au moins un événement, c'est un succès
                if events_read > 0:
                    # Stocker le décalage qui a réussi
                    successful_offset = current_offset

                    # Traiter les amplitudes pour l'histogramme
                    amp_buff = [[] for _ in range(num_channels)]
                    for o in range(num_channels):
                        for p in range(len(amplitude[o])):
                            if isinstance(amplitude[o][p], (list, np.ndarray)):
                                amp_buff[o].extend([x for x in amplitude[o][p] if x != 0])
                            elif isinstance(amplitude[o][p], (int, float)) and amplitude[o][p] != 0:
                                amp_buff[o].append(amplitude[o][p])

                    print(f"\n*** Lecture réussie avec header_offset = {current_offset} ***")
                    print(f"Nombre d'événements lus: {events_read}")
                    print("Fin du fichier atteinte naturellement sans erreurs. Arrêt des tests de décalage.")
                    return amp_buff, num_channels, event_minimums, successful_offset, raw_waveforms
                break

            # Un événement a été lu correctement
            events_read += 1

            # Utiliser la structure de retour avec le numéro d'événement inclus (ignorer le dernier élément qui est le flag de fin de fichier)
            current_idx, trigger, amplitudes, NumberOfChannel, hit_counts, rates, current_timestamp, EventNumber, _ = readed_event

            # Stocker les formes d'onde brutes pour cet événement
            # (Important : faire une copie car amplitudes sera modifié)
            raw_waveforms.append(copy.deepcopy(amplitudes))

            # Vérifier la continuité des numéros d'événements
            if previous_event_number is not None:
                expected_event_number = previous_event_number + 1
                if EventNumber != expected_event_number:
                    print(f"ALERTE: Rupture de séquence d'événements à l'événement #{EventNumber}")
                    print(f"  Attendu: Événement #{expected_event_number}")
                    print(f"  Trouvé: Événement #{EventNumber}")
                    print(f"  Tentative avec décalage suivant...")
                    sequence_error = True
                    break

            # Mettre à jour previous_event_number pour la prochaine itération
            previous_event_number = EventNumber

            # Initialiser les tableaux de données pour le premier événement
            if amplitude is None:
                num_channels = NumberOfChannel
                amplitude = [[] for _ in range(num_channels)]
                event_minimums = [[] for _ in range(num_channels)]

            # Calculer et stocker les minimums pour chaque canal de cet événement
            for o in range(min(num_channels, len(amplitudes))):
                if amplitudes[o]:
                    # Trouver le minimum de ce canal pour cet événement
                    waveform_min = min(amplitudes[o])
                    event_minimums[o].append(waveform_min)

                # Stocker les amplitudes pour l'histogramme original
                amplitude[o].extend(amplitudes[o])

        # Mettre à jour la meilleure tentative si celle-ci a lu plus d'événements
        if amplitude is not None and events_read > best_attempt["events_read"]:
            best_attempt["offset"] = current_offset
            best_attempt["amplitude"] = amplitude
            best_attempt["num_channels"] = num_channels
            best_attempt["event_minimums"] = event_minimums
            best_attempt["events_read"] = events_read
            best_attempt["raw_waveforms"] = raw_waveforms

    # Si nous arrivons ici, aucun des décalages n'a permis d'atteindre la fin du fichier naturellement sans erreur
    print("\n===== Aucun décalage n'a permis une lecture parfaite =====")

    # Utiliser la meilleure tentative si disponible
    if best_attempt["amplitude"] is not None:
        print(
            f"Utilisation du meilleur résultat: offset={best_attempt['offset']}, {best_attempt['events_read']} événements lus")
        successful_offset = best_attempt["offset"]

        # Traiter les amplitudes pour l'histogramme
        amp_buff = [[] for _ in range(best_attempt["num_channels"])]
        for o in range(best_attempt["num_channels"]):
            for p in range(len(best_attempt["amplitude"][o])):
                if isinstance(best_attempt["amplitude"][o][p], (list, np.ndarray)):
                    amp_buff[o].extend([x for x in best_attempt["amplitude"][o][p] if x != 0])
                elif isinstance(best_attempt["amplitude"][o][p], (int, float)) and best_attempt["amplitude"][o][p] != 0:
                    amp_buff[o].append(best_attempt["amplitude"][o][p])

        return amp_buff, best_attempt["num_channels"], best_attempt["event_minimums"], successful_offset, best_attempt[
            "raw_waveforms"]

    print("ERREUR: Échec de lecture avec tous les décalages possibles")
    return [], 0, [], None, []

def find_first_0a01_header(data):
    """
    Trouve simplement le premier 0a 01 dans le fichier binaire et retourne l'index de 01.
    """
    print("Recherche de l'index dans le fichier...")

    for i in range(len(data) - 1):
        if data[i] == 0x0a and data[i + 1] == 0x01:
            print(f"Le fixed header est à 0x{i + 1:x}")
            return i + 1  # Retourne l'index de 01

    print("Aucune séquence 0a 01 trouvée dans le fichier.")
    return None


class FileSelector:
    def __init__(self, directory):
        self.directory = directory
        self.selected_file = None

    def create_window(self):
        self.window = tk.Tk()
        self.window.title("File Selector")
        self.window.geometry("400x300")

        label = ttk.Label(self.window, text="Select a file to analyze:")
        label.pack(pady=10)

        self.listbox = tk.Listbox(self.window, width=50)
        self.listbox.pack(pady=10, padx=10)

        files = self.get_files_list()
        for file in files:
            self.listbox.insert(tk.END, file)

        analyze_button = ttk.Button(self.window, text="Analyze", command=self.on_analyze)
        analyze_button.pack(pady=10)

        quit_button = ttk.Button(self.window, text="Quit", command=self.on_quit)
        quit_button.pack(pady=5)

        self.window.mainloop()

    def get_files_list(self):
        # Liste tous les fichiers dans le dossier binaryread
        return [f for f in os.listdir(self.directory)
                if os.path.isfile(os.path.join(self.directory, f))]

    def on_analyze(self):
        if self.listbox.curselection():
            self.selected_file = self.listbox.get(self.listbox.curselection())
            self.window.quit()
            self.window.destroy()

    def on_quit(self):
        self.selected_file = None
        self.window.quit()
        self.window.destroy()


def create_min_amplitude_window(event_minimums, filename, n_bins=100):
    num_channels = len(event_minimums)
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
    fig.canvas.manager.set_window_title(f'Valeurs ADC minimales - {filename}')
    fig.suptitle(f'Histogramme des amplitudes minimales par événement - {filename} - {num_channels} canaux')

    if num_channels == 1:
        axs = [axs]

    interactive_fits = []

    for idx in range(num_channels):
        if event_minimums[idx]:
            # Créer l'histogramme pour ce canal
            n, bins, patches = axs[idx].hist(event_minimums[idx], bins=n_bins,
                                             label=f'Canal {idx}', alpha=0.6, color='red')
            x = (bins[:-1] + bins[1:]) / 2

            # Centrer et ajuster l'affichage sur les données de ce canal spécifique
            min_val = min(event_minimums[idx])
            max_val = max(event_minimums[idx])

            # Calculer une marge proportionnelle à l'étendue des données
            data_range = max_val - min_val
            margin = max(10, data_range * 0.1)

            # Définir les limites X pour ce canal uniquement
            axs[idx].set_xlim([min_val - margin, max_val + margin])

            # Ajuster les limites Y pour ce canal
            if len(n) > 0:
                max_height = max(n) * 1.1
                axs[idx].set_ylim([0, max_height])

            axs[idx].legend()
            axs[idx].set_ylabel(f'Canal {idx}')
            axs[idx].grid(True, linestyle='--', alpha=0.4)

            # Créer un objet InteractiveGaussFit distinct pour chaque canal
            # et le stocker avec son index de canal pour référence
            interactive_fit = InteractiveGaussFit(x, n, axs[idx])
            interactive_fits.append((idx, interactive_fit))  # Stocker le tuple (index_canal, objet_interactif)
        else:
            axs[idx].text(0.5, 0.5, 'Pas de données disponibles',
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[idx].transAxes)
            axs[idx].set_ylabel(f'Canal {idx}')

    axs[-1].set_xlabel('Valeur ADC minimale')
    plt.subplots_adjust(hspace=0.3)

    return fig, interactive_fits



def create_waveform_window(amplitudes, filename):
    """
    Crée une nouvelle fenêtre pour afficher les formes d'onde de chaque canal
    """
    num_channels = len(amplitudes)

    # Créer la figure et les sous-graphiques
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels))
    fig.canvas.manager.set_window_title(f'Formes d\'onde - {filename}')
    fig.suptitle(f'Formes d\'onde - {filename} - {num_channels} canaux')

    # Si un seul canal, axs n'est pas un tableau, le convertir en liste pour l'uniformité
    if num_channels == 1:
        axs = [axs]

    x = np.arange(1024)  # 1024 points d'échantillonnage

    # Extraire tous les événements pour chaque canal
    all_events = []
    for channel_idx in range(num_channels):
        channel_events = []
        for event_idx in range(len(amplitudes[channel_idx])):
            # Prendre les premiers 'samples' points de l'événement
            if amplitudes[channel_idx][event_idx]:
                # Ne conserver que les 1024 premiers points si nécessaire
                waveform = amplitudes[channel_idx][event_idx][:1024] if len(
                    amplitudes[channel_idx][event_idx]) > 1024 else amplitudes[channel_idx][event_idx]
                channel_events.append(waveform)
        all_events.append(channel_events)

    # Tracer les formes d'onde pour chaque canal
    for idx in range(num_channels):
        if all_events[idx]:
            print(f'Canal {idx}, événements : {len(all_events[idx])}')

            # Calculer les valeurs min et max pour ce canal
            try:
                channel_min = min(min(event) for event in all_events[idx] if event)
                channel_max = max(max(event) for event in all_events[idx] if event)

                # Ajouter une marge de 5% au-dessus et en dessous
                y_margin = (channel_max - channel_min) * 0.05
                y_min = channel_min - y_margin - 10000
                y_max = channel_max + y_margin + 10000

                for event in all_events[idx]:
                    if len(event) == len(x):  # Assurez-vous que les dimensions correspondent
                        axs[idx].plot(x, event, alpha=0.4,
                                      linewidth=1)  # Alpha faible pour voir la superposition des courbes

                axs[idx].set_title(f'Canal {idx}')
                axs[idx].set_xlabel('Point d\'échantillonnage')
                axs[idx].set_ylabel('Valeur ADC')
                axs[idx].set_ylim(y_min, y_max)  # Définir les limites Y optimisées

                # Ajouter une grille pour une meilleure lisibilité
                axs[idx].grid(True, linestyle='--', alpha=0.4)
            except (ValueError, TypeError) as e:
                print(f"Erreur lors du tracé du canal {idx}: {e}")
                # Définir des limites par défaut si nous ne pouvons pas calculer les min/max
                axs[idx].set_title(f'Canal {idx} (données manquantes)')
                axs[idx].set_xlabel('Point d\'échantillonnage')
                axs[idx].set_ylabel('Valeur ADC')
                axs[idx].set_ylim(-10000, 10000)  # Limites Y par défaut
                axs[idx].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    return fig


def main():
    """Fonction principale du programme"""
    drct = "C:/Program Files (x86)/WaveCatcher_64ch/Run_Data/"  # Chemin direct
    base_dir = os.path.join(drct, 'binaryread')
    my_bins = 250

    while True:
        # Créer et afficher le sélecteur de fichier
        file_selector = FileSelector(base_dir)
        file_selector.create_window()

        if file_selector.selected_file is None:
            print("Sortie du programme...")
            break

        # Charger et traiter le fichier sélectionné, incluant les minimums
        # La fonction retourne maintenant aussi le décalage qui a réussi et les formes d'onde brutes
        amp, num_channels, event_minimums, successful_offset, raw_waveforms = load_and_process_single_file_with_minimums(
            file_selector.selected_file,
            base_dir
        )
        # Après l'appel à load_and_process_single_file_with_minimums
        print(f"Nombre de canaux détectés : {num_channels}")
        for i in range(num_channels):
            print(f"Canal {i} : {len(event_minimums[i])} valeurs minimales")
            if event_minimums[i]:
                print(f"  Exemple de valeurs : {event_minimums[i][:5]}")
        # Vérifier si des données ont été obtenues
        if num_channels == 0 or not amp:
            print("Aucune donnée valide n'a été obtenue. Essayez un autre fichier.")
            continue

        # Créer les graphiques pour l'histogramme d'amplitude standard
        fig_hist, axs = plt.subplots(num_channels, 1, figsize=(10, 5 * num_channels), sharex=True)
        fig_hist.canvas.manager.set_window_title(f'Histogramme d\'amplitude - {file_selector.selected_file}')
        fig_hist.suptitle(f'Histogramme d\'amplitude - {file_selector.selected_file} - {num_channels} canaux')

        # Si un seul canal, axs n'est pas un tableau, le convertir en liste pour l'uniformité
        if num_channels == 1:
            axs = [axs]

        interactive_fits = []

        # Créer les histogrammes pour chaque canal
        for idx in range(num_channels):
            if amp[idx]:
                n, bins, patches = axs[idx].hist(amp[idx], bins=my_bins,
                                                 label=f'Canal {idx}', alpha=0.6)

                x = (bins[:-1] + bins[1:]) / 2

                # Ajuster l'échelle X en fonction des données
                if len(amp[idx]) > 0:
                    max_amp = np.max(amp[idx])
                    if max_amp > 7000:
                        axs[idx].set_xlim([-10000, max_amp * 1.1])
                    else:
                        axs[idx].set_xlim([-10000, 7000])

                # Ajuster l'échelle Y
                if len(n) > 0:
                    axs[idx].set_ylim([0, np.max(n) * 1.1])

                axs[idx].legend()
                axs[idx].set_ylabel(f'Canal {idx}')

                # Ajouter l'ajustement interactif pour chaque canal
                interactive_fits.append(InteractiveGaussFit(x, n, axs[idx]))

        axs[-1].set_xlabel('ADC')
        plt.tight_layout()

        # Convertir les formes d'onde brutes au format attendu par create_waveform_window
        # (une liste de listes, où chaque sous-liste contient toutes les formes d'onde d'un canal)
        formatted_waveforms = [[] for _ in range(num_channels)]

        # Si aucune forme d'onde n'a été stockée, afficher un message et passer à l'étape suivante
        if not raw_waveforms:
            print("Aucune forme d'onde brute n'a été stockée. Impossible d'afficher les formes d'onde.")
        else:
            print(f"Préparation de {len(raw_waveforms)} formes d'onde pour l'affichage")
            # Pour chaque événement
            for event_waveforms in raw_waveforms:
                # Pour chaque canal
                for channel_idx in range(min(num_channels, len(event_waveforms))):
                    # Ajouter la forme d'onde de ce canal pour cet événement
                    formatted_waveforms[channel_idx].append(event_waveforms[channel_idx])

        # Créer la deuxième fenêtre pour les valeurs minimales
        # Création de la fenêtre pour les valeurs minimales
        fig_mins, interactive_fits_mins = create_min_amplitude_window(event_minimums, file_selector.selected_file,
                                                                      n_bins=my_bins)

        # Créer la troisième fenêtre pour afficher les formes d'onde brutes
        fig_waveform = create_waveform_window(formatted_waveforms, file_selector.selected_file)

        # Afficher toutes les fenêtres en même temps (non-bloquant)
        plt.ion()  # Activer le mode interactif de matplotlib
        fig_hist.show()
        fig_mins.show()
        fig_waveform.show()
        plt.draw()

        # Attendre que l'utilisateur ferme toutes les fenêtres
        print("Appuyez sur Entrée pour continuer (ou fermez toutes les fenêtres)...")
        plt.ioff()  # Désactiver le mode interactif
        plt.show(block=True)  # Attendre que toutes les fenêtres soient fermées
if __name__ == "__main__":
    main()
