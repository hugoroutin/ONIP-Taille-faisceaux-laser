# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:37:07 2024

@author: routi
"""
import csv
import numpy as np
import cv2

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
    
    :param nom du fichier à utiliser
    """
    chemin_image=os.path.join(dossier, nom_fichier)
    
    # Charger l'image en niveaux de gris
    image = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)
    hauteur,largeur=image.shape
    
    bary_list=get_bary_x_y(nom_fichier)
    [x_barycentre, y_barycentre]=[int(bary_list[0]),int(bary_list[1])]
    
    
    
    # Convertir en image couleur pour tracer les lignes
    image_couleur = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Tracer les lignes sur l'image
    couleur_ligne = (0, 0, 255)  # Rouge pour les lignes
    epaisseur = 2
    # Ligne verticale (x = x_barycentre)
    cv2.line(image_couleur, (x_barycentre, 0), (x_barycentre, hauteur - 1), couleur_ligne, epaisseur)
    # Ligne horizontale (y = y_barycentre)
    cv2.line(image_couleur, (0, y_barycentre), (largeur - 1, y_barycentre), couleur_ligne, epaisseur)
    
    # Dessiner un cercle au barycentre
    cv2.circle(image_couleur, (x_barycentre, y_barycentre), 5, (0, 255, 0), -1)  # Vert pour le barycentre
    
    
    image1 =cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image2 = image_couleur
    alpha=0.80
    if image1 is None or image2 is None:
        raise FileNotFoundError("Une ou plusieurs images n'ont pas pu être chargées.")
    
    # Vérifier si les dimensions des images correspondent
    if image1.shape != image2.shape:
        print(f"Redimensionnement : {image1.shape} -> {image2.shape}")
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))  # Adapter image2 à image1
    
    # Superposer les images
    beta = 1 - alpha  # Coefficient de l'image 2
    image_superposee = cv2.addWeighted(image1, alpha*100, image2, beta/100, 0)
    
    # Afficher l'image superposée
    cv2.imshow("Image Superposée", image_superposee)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    # # Afficher l'image
    # cv2.imshow("Image avec barycentre et droites", image_couleur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

tracer_droites_bary('Profil1.tif')






