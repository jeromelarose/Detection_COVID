import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from joblib import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm


title = "Prédiction"
sidebar_name = "Prédiction"

@st.cache(allow_output_mutation=True)
def load_model():
    Model2 = tf.keras.models.load_model("../models/model2")
    Model1 = tf.keras.models.load_model("../models/model1")
    unet = tf.keras.models.load_model("../models/model_unet/")
    unet.trainable = False

    # Charger l'historique d'entraînement
    # history = load('../models/history_model2')
    return unet, Model1, Model2


Unet, Model1, Model2 = load_model()


def display(display_list, num):
    fig, axs = plt.subplots(nrows=num, ncols=3, sharex=True, sharey=True, figsize=(7,2.5*num))
    fig.set_facecolor('lavender')

    title = ['Image', 'Masque prédit', 'Image masquée']
    if num > 1:
        for n in range(num):
            for i in range(len(display_list)):
                axs[n,i].set_title(title[i], color='g')
                axs[n,i].imshow(np.squeeze(display_list[i][n]*255), cmap = 'gray')
    else:
        for i in range(len(display_list)):
            axs[i].set_title(title[i], color='g', fontweight='bold', fontsize=12)
            axs[i].imshow(np.squeeze(display_list[i]*255), cmap = 'gray')

    plt.xticks([])
    plt.yticks([])
    st.pyplot(fig)
  

def show_predictions(dataset, model, num=1):
    pred_mask = model.predict(dataset)

    display([dataset[:num], np.round(pred_mask[:num]), np.multiply(np.round(pred_mask[:num]), dataset[:num])], num)
        # break


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Tout d'abord, nous créons un modèle qui mappe l'image d'entrée aux activations
    # de la dernière couche convolutive ainsi que les prédictions de sortie
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Ensuite, nous calculons le gradient de la classe prédite supérieure pour notre image d'entrée
    # par rapport aux activations de la dernière couche convolutive
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # C'est le gradient de la sortie neuronale (prédite supérieure ou choisie)
        # par rapport à la carte de caractéristiques de sortie de la dernière couche convolutive
        grads = tape.gradient(class_channel, last_conv_layer_output)
    grads2 = tape2.gradient(grads, last_conv_layer_output )

    # Ceci est un vecteur où chaque entrée est l'intensité moyenne du gradient
    # sur un canal spécifique de la carte de caractéristiques
    pooled_grads = tf.reduce_mean(grads2, axis=(0, 1, 2))
    pooled_grads, _ = tf.clip_by_global_norm([pooled_grads], clip_norm=1.0)

    # Nous multiplions chaque canal dans le tableau de carte de caractéristiques
    # par "l'importance de ce canal" par rapport à la classe prédite supérieure
    # puis nous additionnons tous les canaux pour obtenir l'activation de classe heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads = pooled_grads[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Pour la visualisation, nous allons également normaliser la heatmap entre 0 et 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index, preds[0]


def get_superposed_img(image, heatmap, cam_path="cam.jpg", alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

@st.cache_data
def prediction(img, _model, layer_name):
    heatmap, index, prob =  make_gradcam_heatmap(img, _model, layer_name)
    image = get_superposed_img(img[0], heatmap)
    return image, index, prob

key = 0

def run():
    global key
    uploaded_file = []
    st.title(title)
    model = Model1
    layer_name = 'separable_conv2d_31'

    # Demander à l'utilisateur de télécharger une image
    model_name = st.selectbox('Sélectionner un model', ['Model from scratch','Model Xception finetuned'])

    if model_name == 'Model Xception finetuned':
        model = Model2
        layer_name = 'grad'


    if st.button("Supprimer toutes les images"):
        key += 1

    uploaded_file = st.file_uploader("Télécharger des images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=key)
    uploaded_file2 = uploaded_file
    # Si l'utilisateur a téléchargé une image, l'afficher
    if uploaded_file is not None and len(uploaded_file):
        nb_images = 1
        if len(uploaded_file) > 1:
            max = 10 if len(uploaded_file) >= 10 else len(uploaded_file)
            nb_images = st.slider("Nombre d'images", min_value=1, max_value=max, value=max)
            suffle = st.button("Mélanger les images", key='m1')
            if suffle:
                random.shuffle(uploaded_file)
            # Redimensionner toutes les images en 64x64 pixels
        resized= []

        for path in uploaded_file:
            img = Image.open(path).convert('L')
            img = tf.expand_dims(img, axis=-1)
            img_resized = tf.image.resize(img, (256, 256))
            img_resized /= 255
            resized.append(img_resized)
        resized= np.array(resized)
        
        show_predictions(resized[:nb_images], Unet, nb_images)

        select = st.radio('Affichages', ['Images + Grad-CAM', 'Images', 'Grad-CAM'])

        if len(uploaded_file2) > 1:
            max = 10 if len(uploaded_file2) >= 10 else len(uploaded_file2)
            nb_images = st.slider("Nombre d'images Grad-CAM", min_value=1, max_value=max, value=max)
            suffle = st.button("Mélanger les images", key='m2')
            if suffle:
                random.shuffle(uploaded_file2)  
        
        resized= []

        images =  np.array(uploaded_file2)


        # nb_images = nb_line * 3 if len(images) > nb_line * 3 else (len(images) // 3) * 3
        for path in images[:nb_images]:
            img = Image.open(path).convert('L')
            img = tf.expand_dims(img, axis=-1)
            img_resized /= 255
            img_resized = tf.image.resize(img, (299, 299))
            resized.append(img_resized)
        resized= np.array(resized)


        cat = ["COVID", "NON COVID"]
        if select == 'Images + Grad-CAM':
            for i in range(nb_images):

                img = resized[i]
                # redimensionne l'image si nécessaire
                if model_name == 'Model Xception finetuned':
                    img = tf.image.resize(img, (224,224))
                img = np.expand_dims(img, axis=0)
        

                image, index, prob = prediction(img, model, layer_name)

                fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(11,6))
            
                axs[0].set_title(f"{cat[index]} à {(prob[index].numpy()*100).round(2)} %", color='g', fontweight='bold', fontsize=20)
                axs[0].imshow(img[0], cmap='gray', vmin=0, vmax=255)
                axs[1].set_title(f"{cat[index]} à {(prob[index].numpy()*100).round(2)} %", color='g', fontweight='bold', fontsize=20)
                axs[1].imshow(image, cmap='gray', vmin=0, vmax=255)
                plt.xticks([])
                plt.yticks([])
                fig.patch.set_facecolor('lavender')
                st.pyplot(fig)

        elif select == 'Images':
            pred = model.predict(resized)
            labels = tf.argmax(pred, axis=1)
            for i in range(len(labels)):
                index = labels[i].numpy()
                prob = pred[i][index]
          
                fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5,5))
                
                axs.set_title(f"{cat[index]} à {(prob*100).round(2)} %", color='g', fontweight='bold', fontsize=20)
                axs.imshow(resized[i], cmap='gray', vmin=0, vmax=255)
                plt.xticks([])
                plt.yticks([])
                fig.patch.set_facecolor('lavender')
                st.pyplot(fig)

        else:
            st.write("je suis dans grad-cam")
            for i in range(nb_images):

                img = resized[i]
                # redimensionne l'image si nécessaire
                if model_name == 'Model Xception finetuned':
                    img = tf.image.resize(img, (224,224))
                img = np.expand_dims(img, axis=0)
        

                image, index, prob = prediction(img, model, layer_name)

                fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(11,6))
            
                axs.set_title(f"{cat[index]} à {(prob[index].numpy()*100).round(2)} %", color='g', fontweight='bold', fontsize=20)
                axs.imshow(image, cmap='gray', vmin=0, vmax=255)
                plt.xticks([])
                plt.yticks([])
                fig.patch.set_facecolor('lavender')
                st.pyplot(fig)
     
        





 