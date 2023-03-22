import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Analyse"
sidebar_name = "Analyse"


def run():

    st.title(title)

    st.markdown("Compréhension et manipulation des données")
    st.markdown("1) Cadre ")

    st.markdown("""
        >Pour atteindre les différents objectifs du projet, nous avons à disposition une base de données provenant de plusieurs sources :
        >- https://sirm.org/category/senza-categoria/covid-19/
        >- https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png
        >- https://eurorad.org
        >- https://github.com/armiro/COVID-CXNet
        >- https://github.com/ieee8023/covid-chestxray-dataset
        >- https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711
        >- https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
        >- https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia    

        >Parmi ces images, il y a 4 types de radiographie : COVID, Normal, Lung Opacity et Viral Pneumonia correspondant respectivement à cas positifs au Covid-19, poumons normaux, poumons atteints d’opacité pulmonaire et des poumons atteints de pneumonie aiguë.

        >Au total il y a 21165 images, ainsi que leurs masques associés. Parmi elles, 3616 Covid, 10192 Normales, 6012 Lung Opacity et 1345 Pneumonies aiguës. On peut voir ainsi que les proportions des catégories sont différentes selon les sources :
    """)

    st.image(Image.open("../images/proporsion_categories.png"))

    st.markdown("""
    >Chaque image possède son masque respectif, cependant ils n’ont pas les mêmes dimensions :
    >- Image 299x299 pixels
    >- Masque 256x256 pixels
    """)
    st.image(Image.open("../images/image299.png"))

    st.markdown("2) Pertinence")
    st.markdown("""
    >Les variables principales de cette base de données sont donc les différentes images et les masques correspondant aux 4 catégories citées précédemment. La variable cible correspond aux labels des différentes radiographies, ainsi que leurs masques associés, c’est à dire : le diagnostic. Le jeu de données n’étant que des images avec les masques, les sources et les diagnostiques correspondants, il n’y a pas à proprement parler beaucoup de variété dans les données statistiques. Nous allons tout de même exploiter les différentes informations mises à notre disposition.
    """)

    st.markdown("3) Pre-processingetfeatureengineering")
    st.markdown(">Nous pouvons constater que les classes sont déséquilibrées au sein du dataset :")
    st.image(Image.open("../images/proporsion_sources.png"))

    st.markdown("""
    >En effet, comme dit précédemment, il y a 21165 images. Parmi elles, 3616 Covid, 10192 Normales, 6012 Lung Opacity et 1345 Pneumonies aiguës.
    >Cela pourra s’avérer être un problème lors de l’entrainement d’un modèle de prédiction. Nous avons commencé par redimensionner les images pour qu’elles correspondent aux masques :
    >- Image 256x256 pixels
    >- Masque 256x256 pixels
    """)
    st.image(Image.open("../images/image256.png"))


    st.markdown("""
    >Nous affichons pour chaque catégorie l’image la plus foncée, d’intensité moyenne et la plus claire avec son masque, le contour du masque appliqué pour s’assurer que le masque correspond bien, ainsi que l’image avec le masque appliqué :
    """)
    st.image(Image.open("../images/images_cm_COVID.png"))
    st.image(Image.open("../images/images_cm_Normal.png"))
    st.image(Image.open("../images/images_cm_Lung_Opacity.png"))
    st.image(Image.open("../images/images_cm_Viral Pneumonia.png"))

    st.markdown("""
    >Nous nous sommes rapidement aperçus que les images manquaient de contraste et nous nous sommes donc intéressés à la luminosité générale des différentes radiographies:
    """)
    st.image(Image.open("../images/dist_img.png"))

    st.markdown(">En affichant la distribution des pixels sur quelques images tirées de manière aléatoire, on remarque que la valeur médiane est différente suivant la catégorie de la radio et que les pixels ne sont pas repartis sur toute la plage (0, 255). Il va donc être nécessaire de normaliser les images.")
    st.markdown(">Nous avons choisi de réaliser une normalisation par histogramme. Cela consiste à considérer la représentation graphique des distributions d’intensité d’une image, et à l’étendre sur toute la plage de valeurs de 0 à 255. On peut alors quantifier le nombre de pixels pour chaque valeur d’intensité. A partir de l’histogramme de l’image, nous allons chercher à rendre la distribution d’intensité plus large et uniforme afin que les valeurs d’intensité soient mieux réparties. Ainsi, si beaucoup de pixels sont dans une gamme réduite d’intensité, la normalisation va répartir ces mêmes pixels sur une gamme d’intensité plus large, afin de représenter la même image avec une plus grande variété de niveaux de gris. Cela a pour effet d’améliorer le contraste et donc l’apparition de détails sur l’image finale. L’image ci-dessous permet de mieux comprendre la transformation de la répartitions des pixels.")
    st.image(Image.open("../images/img.jpg"))


    st.markdown(">Nous avons donc équilibré toutes nos images et nous affichons de nouveau la distribution des pixels :")
    st.image(Image.open("../images/dist_img_norm.png"))
    st.markdown(">En affichant la distribution des pixels sur quelques images normalisées tirées de manière aléatoire, on remarque que les valeurs médianes sont plus proches et que les pixels sont bien repartis sur toute la plage (0, 255). Nous pensons aussi que par ce moyen, une répartition plus uniforme de la luminosité des pixels permettra d’éliminer toute forme de biais de décision pour le modèle de catégorisation final.")

    st.markdown(">Nous affichons de nouveau pour chaque catégorie l’image la plus foncée, d’intensité moyenne et la plus claire avec son masque, le contour du masque appliqué pour s’assurer que le masque correspond bien, ainsi que l’image avec le masque appliqué")
    st.image(Image.open("../images/images_ncm_COVID.png"))
    st.image(Image.open("../images/images_ncm_Normal.png"))
    st.image(Image.open("../images/images_ncm_Lung_Opacity.png"))
    st.image(Image.open("../images/images_ncm_Viral Pneumonia.png"))

    st.markdown(">Nous constatons que le contraste est bien meilleur avec les images normalisées. Mais en comparant les images avec le masque on se rends compte qu’elle manque de contraste également :")
    st.image(Image.open("../images/masked.png"))

    st.markdown(">Nous avons aussi constaté qu’il faudrait aussi normaliser les images avec l’application du masque en affichant la distribution des pixels des images avec masque choisies aléatoirement :")
    st.image(Image.open("../images/dist_masked.png"))

    st.markdown(">On remarque que les pixels ne sont pas repartis sur toute la plage (0, 255). On normalise donc les images avec masque et on affiche de nouveau la distribution des pixels sur ces images avec masque normalisées :")
    st.image(Image.open("../images/dist_masked_norm.png"))


    st.markdown(">Cette distribution nous permet de confirmer que la normalisation des images avec masque a bien été exécutée en comparant avec des images avec masque sans normalisation et avec normalisation :")
    st.image(Image.open("../images/masked.png"))
    st.image(Image.open("../images/masked_n.png"))

    st.markdown(">On remarque que le contraste après normalisation sur les images avec masque est bien meilleur. Affichons maintenant les images finales :")
    st.image(Image.open("../images/rendu_final_COVID.png"))
    st.image(Image.open("../images/rendu_final_Lung_Opacity.png"))
    st.image(Image.open("../images/rendu_final_Normal.png"))
    st.image(Image.open("../images/rendu_final_Viral Pneumonia.png"))
