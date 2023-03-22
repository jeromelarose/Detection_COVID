import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from joblib import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm


title = "Remerciement"
sidebar_name = "Remerciement"

def run():

    st.title(title)
    st.markdown("""
            Nous souhaitons remercier :  
            - Notre mentor de projet, Okba Bentoumi, pour son soutien et ses nombreux conseils.
            - Datascientest, pour les cours de qualité et les bonnes méthodes
            - Maëlys de Datascientest, pour son temps et son implication
            - Lucas Rousse pour son aide à la rédaction de ce rapport
            - Cécile Chiron qui a été d’une patience extrême envers Mr Pagesy
            - Naruto et ses 220 épisodes, parfaits pour rester éveillé pendant les entrainements de modèles
            """)