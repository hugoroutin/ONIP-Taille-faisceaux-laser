# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:37:07 2024

@author: routi
"""
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt


z = []
nom = []
with open('data.csv', 'r') as f:
    reader = csv.reader(f , delimiter=',')
    for row in reader:
        z = np.append(z,float(row[0]))
        nom = np.append(nom,row[1])
'''Ces lignes de codes ouvrent le fichier data.csv et viennent
 lire le fichier et ecrivent dans des liestes z et nom un atribut
 z et le nom de ce qui semble etre un fichier .tif (une image)
 respectivmeent

 '''      


#################
# 2 OUVERTURE DE Profil1.tif 
#################

from PIL import Image  
import os  


dossier = r"C:\Users\routi\OneDrive\Documents\GitHub\ONIP-2\Profils_sans_bruit"  #dossier du fichier
nom_fichier = "Profil1.tif"  # nom du fichier


chemin_image = os.path.join(dossier, nom_fichier)


# try:
#     image = Image.open(chemin_image)
#     image.show()
# except FileNotFoundError:
#     print(f"Le fichier {chemin_image} est introuvable. Vérifiez le chemin.")


#################
# 3 Calcul des coord. des barycentres
#################


def get_bary_x_y(nom_fichier):
    """
    Calcule le barycentre d'intensité d'une image en niveaux de gris.
    
    :param nom_fichier: Nom de l'image .tif
    :return: liste des coord [x_barycentre, y_barycentre] en pixels
    """
    
    chemin_image=os.path.join(dossier, nom_fichier)
    
    # Conversion éventuelle en niveaux de gris
    image = Image.open(chemin_image).convert("L")
    
    # conversion de l'image en tableau numpy
    image_array = np.array(image, dtype=np.float64)
    
    # dimension de l'image
    hauteur, largeur = image_array.shape
    
    #création des matrices d'indices des pixels
    y_indices, x_indices = np.indices((hauteur, largeur))
    
    # Calculer la somme totale des intensités
    somme_intensites = np.sum(image_array)
    
    if somme_intensites < 0: #rentrer valeur arbitraire en deça de laquelle le barycentre n'a plus de signification physique à cause du bruit
        raise ValueError("Le barycentre est indéfini.")
    
    # Calculer les coordonnées du barycentre
    x_barycentre = np.sum(x_indices * image_array) / somme_intensites
    y_barycentre = np.sum(y_indices * image_array) / somme_intensites
    
    return [x_barycentre, y_barycentre]


def get_max_min(nom_fichier):
    """
    renvoie le max d'intensité et le min d'une image en .tif
    
    :param nom_fichier: Nom de l'image .tif
    :return: liste  [max, min] 
    """
    
    chemin_image=os.path.join(dossier, nom_fichier)
    
    # Conversion éventuelle en niveaux de gris
    image = Image.open(chemin_image).convert("L")
    
    # conversion de l'image en tableau numpy
    image_array = np.array(image, dtype=np.float64)
    max_int=np.max(image_array)
    min_int=np.min(image_array)
    
    return [max_int, min_int]

print('[x_barycentre, y_barycentre]=',get_bary_x_y('Profil1.tif'))
print('Intensité max=',get_max_min('Profil1.tif')[0])
print('Intensité min=',get_max_min('Profil1.tif')[1])



def tracer_droites_bary(nom_fichier):
    """
    Trace les droites passant par le barycentre d'intensité sur une image.
    
    :param nom_fichier: Nom du fichier à utiliser
    :param dossier: Dossier contenant le fichier image
    """
    chemin_image = os.path.join(dossier, nom_fichier)
    image_pil = Image.open(chemin_image)
    image_pil = image_pil.convert('L')  # Convertir en niveaux de gris
    image = np.array(image_pil)  # Convertir en tableau numpy pour OpenCV
    
    hauteur, largeur = image.shape
    
    # Obtenir les coordonnées du barycentre
    bary_list = get_bary_x_y(nom_fichier)
    x_barycentre, y_barycentre = int(bary_list[0]), int(bary_list[1])
    
    # Convertir en image couleur pour tracer les lignes
    image_couleur = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Tracer les lignes sur l'image
    couleur_ligne = (0, 0, 255)  # Rouge pour les lignes
    epaisseur = 1
    # Ligne verticale (x = x_barycentre)
    cv2.line(image_couleur, (x_barycentre, 0), (x_barycentre, hauteur - 1), couleur_ligne, epaisseur)
    # Ligne horizontale (y = y_barycentre)
    cv2.line(image_couleur, (0, y_barycentre), (largeur - 1, y_barycentre), couleur_ligne, epaisseur)
    
    # # Dessiner un cercle au barycentre
    # cv2.circle(image_couleur, (x_barycentre, y_barycentre), 5, (0, 255, 0), -1)  # Vert pour le barycentre
    
    # Créer une copie de l'image originale en couleur pour la superposition
    image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image2 = image_couleur
    
    # # Vérifier si les dimensions des images correspondent
    # if image1.shape != image2.shape:
    #     image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))  # Adapter image2 à image1
    
    
    
    # Afficher les deux images avant de superposer
    cv2.imshow("Image Originale", image1)
    cv2.imshow("Image avec Droites", image2)
    cv2.waitKey(0)  # Attendre une touche pour fermer
    
    

#tracer_droites_bary('Profil1.tif')

def tracer_profil_faisceau(nom_fichier):
    chemin_image=os.path.join(dossier, nom_fichier)
    
    # Conversion éventuelle en niveaux de gris
    image = Image.open(chemin_image).convert("L")
    
    # conversion de l'image en tableau numpy
    image_array = np.array(image, dtype=np.float64)
    bary_list = get_bary_x_y(nom_fichier)
    x_barycentre, y_barycentre = int(bary_list[0]), int(bary_list[1])
    array_max_x=image_array[x_barycentre]
    array_max_y=image_array[y_barycentre]
    
    x = np.linspace(0, len(array_max_x), len(array_max_x))
    
    plt.plot(x, array_max_x, label='Intensité selon laxe x', color='b')
    #plt.title('Graphique Sinus')
    plt.title('Intensité selon  laxe x')
    plt.ylabel('Intensité')
    plt.legend()
    plt.show()
    
    y = np.linspace(0, len(array_max_y), len(array_max_y))
    
    plt.plot(y, array_max_y, label='Intensité selon laxe y', color='r')
    plt.title('Intensité selon  laxe y')
    
    plt.ylabel('Intensité')
    plt.legend()

    plt.show()


#tracer_profil_faisceau('Profil1.tif')






