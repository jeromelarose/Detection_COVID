import streamlit as st


title = "DataScientest COVID project."
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title("Introduction")

    # st.markdown("---")

    # st.markdown(
    #     """
    #     Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

    #     You can browse streamlit documentation and demos to get some inspiration:
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into streamlit [documentation](https://docs.streamlit.io)
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #       Dataset] (https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset]
    #       (https://github.com/streamlit/demo-uber-nyc-pickups)
    #     """
    # )

    st.markdown("# **1) Contexte**")

    st.markdown("""
                >La radiographie pulmonaire s’est révélée essentielle dans la détection de cas positifs au Covid-19. Cette technique consiste à faire une impression des différences de densité du poumon sur un film radiographique. Elle permet d’étudier les poumons, la trachée, les bronches, le cœur, les vertèbres et les côtes. Initialement, elle était utilisée dans le domaine médical afin de détecter des infections pulmonaires, des cancers, des inflammations ou tout autre type d’anomalie.
                """)

    st.markdown("""
                >Ajoutant maintenant la détection du Covid-19 à son expertise, nous avons cherché à savoir s’il était possible de détecter les cas positifs au Covid-19 à l’aide du deep learning. Pour se faire, nous allons essayer de déterminer un modèle de prédiction sur une base de données contenant des radiographies pulmonaires afin de classifier les cas Covid-19.
                """)

    st.markdown("""
                >Cela pourrait notamment permettre d’utiliser cette méthode dans les hôpitaux et cliniques afin de soulager la charge de travail des médecins sans pour autant négliger l’avis médical d’un professionnel.
                """)

    st.markdown("# **2) Objectifs**")


    st.markdown("""
                >Les principaux objectifs du projet sont :
                >- L’exploration des données grâce à la visualisation de celles-ci et l’analyse des données.
                >- La normalisation des données
                >- L’implémentation d’une architecture U-NET pour la segmentation des images.
                >- L’implémentation d’une architecture CNN pour la classification des images.
                >- L’analyse des résultats obtenus et des performances des modèles.
                >- Éventuellement un processus de visualisation des zones d’intérêt sur les radios
                classées comme cas positif de covid-19
                >- Une démonstration via un streamlit.  

                >Il est tout à fait possible d’apporter un avis médical au projet, notamment sur la lecture de radiographie pulmonaire, grâce à l’intervention d’un professionnel du milieu médical. Cela apporterait une expertise supplémentaire sur la détection de différentes anomalies pulmonaires.
                """)

    