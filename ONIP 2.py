# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:37:07 2024

@author: routin
"""
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image  
import os  
import scipy
from scipy.optimize import curve_fit

utilisateur='Ariane'
utilisateur='Hugo'



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


#
#
#################
# 2 OUVERTURE DE Profil1.tif 
#################


if utilisateur=='Ariane':
    dossier=r"copie ici le chemin "
else:
    dossier=r"C:\Users\routi\OneDrive\Documents\GitHub\ONIP-2\Profils_sans_bruit"


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

# print('[x_barycentre, y_barycentre]=',get_bary_x_y('Profil1.tif'))
# print('Intensité max=',get_max_min('Profil1.tif')[0])
# print('Intensité min=',get_max_min('Profil1.tif')[1])



def tracer_droites_bary(nom_fichier):
    """
    Trace les droites passant par le barycentre d'intensité sur une image.
    
    :param nom_fichier: Nom du fichier à utiliser
    
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
    # maxmin=get_max_min(nom_fichier)
    # cv2.circle(image_couleur, (x_barycentre, y_barycentre), 5, (0, 255, 0), -1)  # Vert pour le barycentre
    
    # Créer une copie de l'image originale en couleur pour la superposition
    image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image2 = image_couleur
    
    # Afficher les deux images avant de superposer
    cv2.imshow("Image Originale", image1)
    cv2.imshow("Image avec Droites", image2)
    cv2.waitKey(0)  # Attendre une touche pour fermer
    
    
# for i in nom:
#    tracer_droites_bary(i)
#tracer_droites_bary('Profil2.tif')

def tracer_profil_faisceau(nom_fichier):
    chemin_image=os.path.join(dossier, nom_fichier)
    
    # Conversion éventuelle en niveaux de gris
    image = Image.open(chemin_image).convert("L")
    
    # conversion de l'image en tableau numpy
    image_array = np.array(image, dtype=np.float64)
    bary_list = get_bary_x_y(nom_fichier)
    x_barycentre, y_barycentre = int(bary_list[0]), int(bary_list[1])
    
    array_max_x=image_array[:,x_barycentre]
    
    array_max_y=image_array[y_barycentre,:]
    
    x = np.linspace(0, len(array_max_x), len(array_max_x))
    plt.subplot(2, 1,1)
    plt.plot(x, array_max_x, label='Intensité selon laxe x', color='b')
    #plt.title('Graphique Sinus')
    plt.title('Intensité selon  laxe x')
    plt.ylabel('Intensité')
    plt.legend()
    #plt.show()
    
    y = np.linspace(0, len(array_max_y), len(array_max_y))
    plt.subplot(2,1,2)
    plt.plot(y, array_max_y, label='Intensité selon laxe y', color='r')
    plt.title('Intensité selon  laxe y')
    
    plt.ylabel('Intensité')
    plt.legend()
    plt.tight_layout()

    plt.show()





def gaussienne(x, A, B, x0, omega):
    """
    Fonction gaussienne adaptée au traçage.
    """
    return A + B * np.exp(-2 * ((x - x0)**2) / omega**2)

def fit_gaussien(nom_fichier):
    chemin_image = os.path.join(dossier, nom_fichier)
    image = Image.open(chemin_image).convert("L")
    image_array = np.array(image, dtype=np.float64)

    # Simuler les coordonnées du barycentre si get_bary_x_y est indisponible
    bary_list = get_bary_x_y(nom_fichier)
    x_barycentre, y_barycentre = int(bary_list[0]), int(bary_list[1])

    # Extraction des profils de pixels
    array_max_x = image_array[:, x_barycentre]
    array_max_y = image_array[y_barycentre, :]

    # Axes pour le fit
    x = np.linspace(0, len(array_max_x) - 1, len(array_max_x))
    y = np.linspace(0, len(array_max_y) - 1, len(array_max_y))
    p0x=[get_max_min(nom_fichier)[1] ,get_max_min(nom_fichier)[0]-get_max_min(nom_fichier)[1], x_barycentre ,300]
    p0y=[get_max_min(nom_fichier)[1] ,get_max_min(nom_fichier)[0]-get_max_min(nom_fichier)[1], y_barycentre ,300]
    # Ajustement de la gaussienne
    params_x, covariance_x = curve_fit(gaussienne, x, array_max_x, p0x)
    params_y, covariance_y = curve_fit(gaussienne, y, array_max_y, p0y)
    
    # params_x, covariance_x = curve_fit(gaussienne, x, array_max_x, p0=params_x)
    # params_y, covariance_y = curve_fit(gaussienne, y, array_max_y, p0=params_y)
    return params_x, params_y

def tracer_profil_faisceau_avec_fit(nom_fichier):
    chemin_image = os.path.join(dossier, nom_fichier)
    image = Image.open(chemin_image).convert("L")
    image_array = np.array(image, dtype=np.float64)

    # Simuler les coordonnées du barycentre
    bary_list = get_bary_x_y(nom_fichier)
    x_barycentre, y_barycentre = int(bary_list[0]), int(bary_list[1])
    print(bary_list)
    # Extraction des profils
    array_max_y = image_array[:, x_barycentre]
    array_max_x = image_array[y_barycentre, :]

    # Fit des données
    params_y,params_x  = fit_gaussien(nom_fichier)

    # Axes pour le fit
    x = np.linspace(0, len(array_max_x) - 1, len(array_max_x))
    y = np.linspace(0, len(array_max_y) - 1, len(array_max_y))

    # Tracer les profils et les fits
    plt.figure(figsize=(10, 8))

    # Profil selon x
    plt.subplot(2, 1, 1)
    plt.plot(x, array_max_x, label="Profil X", color='red')
    plt.plot(x, gaussienne(x, *params_x), label="Fit Gaussien X", color='blue')
    plt.title("Profil et Fit selon l'axe X")
    plt.xlabel("Position")
    plt.ylabel("Intensité")
    plt.legend()

    # Profil selon y
    plt.subplot(2, 1, 2)
    plt.plot(y, array_max_y, label="Profil Y", color='red')
    plt.plot(y, gaussienne(y, *params_y), label="Fit Gaussien Y", color='blue')
    plt.title("Profil et Fit selon l'axe Y")
    plt.xlabel("Position")
    plt.ylabel("Intensité")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return params_x, params_y


#tracer_profil_faisceau_avec_fit('Profil1.tif')

"""Automatisation du processus"""

Donnees=[]
for k in range (len(nom)) : 
    
    
    #Donnees=np.zeros([len(nom),2])
    nom_fichier =str(nom[k])
    
    
    params_x,params_y=fit_gaussien(nom_fichier)
    Donnees.append([np.abs(params_x[3]),np.abs(params_y[3])])
    
    #Donnees[k,0],Donnees[k,1]=params_x[3],params_y[3]
    donnees_array=np.array(Donnees)
    # print(donnees_array)
    

plt.plot(z, donnees_array[:, 1], label='selon x', color='b')
plt.plot(z, donnees_array[:, 0], label='selon y', color='r')

plt.title('Largeur du faisceau')
plt.ylabel('largeur')
plt.legend()
plt.show()



def rayon(z,w,M):
    # z=np.linspace(0,15,15)
    landa=1.3e-6
    r=w*np.sqrt(1+((M*landa*z)/(np.pi*w**2))**2)
    print(r)
    return r 
print('bite')
params_w_x, ballec=curve_fit(rayon, z, donnees_array[:, 1], [80,1])
params_w_y, ballec=curve_fit(rayon, z, donnees_array[:, 0], [80,1])
#
print('bite')
print(params_w_x[1])
print(params_w_y[1])

# plt.plot(z, [params_w_x[1]*15], label='selon x', color='b')
# plt.plot(z, params_w_y[1], label='selon y', color='r')
# #plt.title('Graphique Sinus')
# plt.title('Largeur du faisceau')
# plt.ylabel('Intensité')
# plt.legend()
# plt.show()

# plt.plot(z, rayon(1,2,3), label='rayon', color='b')
# #plt.title('Graphique Sinus')
# plt.title('Rayon selon la position z')
# plt.ylabel('waist')
# plt.legend()
# plt.show()



# print get_bary_x_y(nom_fichier)
# print get_max_min(nom_fichier)
# print tracer_droites_bary(nom_fichier)
# print tracer_profil_faisceau(nom_fichier)
