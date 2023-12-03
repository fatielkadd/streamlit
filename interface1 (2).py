import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon
import pandas as pd
from jenkspy import jenks_breaks
from folium.plugins import MarkerCluster
import leafmap.foliumap as leafmap
import altair as alt
from PIL import Image
import os
import folium
import requests
import geopy
import rasterio
from rasterio.plot import reshape_as_image
from branca.colormap import LinearColormap
import numpy as np
from folium.plugins import MarkerCluster
from streamlit_option_menu import option_menu
import rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling
from streamlit_folium import folium_static
from pyproj import Transformer
from rasterio.windows import Window
import requests
from io import BytesIO


with st.sidebar:
    selected = option_menu(None, ["Home","Map","Classified Map","SplitMap","Requetes ","Timelapes","timeseries","slider","Explore COG","Contact"], 
                       icons=['house', 'cloud-upload', "list-task", 'gear','search','clock','clock','columns','envelope'],  
                       menu_icon="cast", default_index=0,
                       styles={"container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#FF4B91", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "brown"},
    })



# Affichage de la page sélectionnée
if selected == "Home":
    # Page d'accueil
    st.markdown(
        """
        <style>
            body {
                background-color: #000;
                color: #fff;
                font-family: 'Arial', sans-serif;
            }
            .container {
                max-width: 800px;
                margin: auto;
            }
            .welcome-section {
                text-align: center;
                margin-top: 50px;
            }
            .membre-info {
                padding: 20px;
                background-color: #333; /* Fond légèrement plus clair que le noir pur */
                border-radius: 10px;
                margin-top: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            h1, h2 {
                color: #fff;
                font-weight: bold;
            }
            p {
                color: #ccc;
            }
            .summary {
                font-size: 1.2em;
                margin-top: 20px;
                color: #ccc;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Section de bienvenue
    st.markdown(
        """
        <div class='container welcome-section'>
            <h1>Bienvenue sur Notre Dashboard</h1>
            <p>
                Nous sommes ravis de vous accueillir sur notre plateforme dédiée à la visualisation et à l'interaction
                avec des données spatiales. Explorez les différentes fonctionnalités et découvrez de nouvelles façons
                innovantes de travailler avec vos données.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Section pour les objectifs de l'application
    st.markdown(
        """
        <div class='container membre-info'>
            <h2>Objectifs de l'Application</h2>
            <p>
                Notre objectif principal est de fournir une expérience utilisateur immersive et conviviale pour
                visualiser et interagir avec des données spatiales. Explorez les différentes visualisations,
                utilisez les outils interactifs et découvrez les fonctionnalités avancées de notre dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p class='summary'>Découvrez les fonctionnalités de notre dashboard et profitez de l'exploration de données spatiales.</p>", unsafe_allow_html=True)



if selected == "Contact":
    # Informations sur les membres du groupe
    membre1 = {"nom": "Nom 1", "photo": "C://Users//hp/Downloads//Compressed//streamlit-multipage-template-master//fati.jpeg", "info": "Informations sur le Membre 1"}
    membre2 = {"nom": "Nom 2", "photo": "C://Users//hp/Downloads//Compressed//streamlit-multipage-template-master//fatiha.jpeg", "info": "Informations sur le Membre 2"}





    # Présentation des membres du groupe
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    st.image(Image.open(membre1["photo"]), caption=f"{membre1['nom']} - Membre 1", use_column_width=True, width=100)
    st.markdown(f"<div class='membre-info'>{membre1['info']}</div>", unsafe_allow_html=True)

    st.image(Image.open(membre2["photo"]), caption=f"{membre2['nom']} - Membre 2", use_column_width=True, width=100)
    st.markdown(f"<div class='membre-info'>{membre2['info']}</div>", unsafe_allow_html=True)

    
if selected == "Classified Map":
    # Affichage de la première page uniquement lorsque l'utilisateur clique sur "Page 1"
    st.header("Page 1 - Aspect 1")
    st.subheader("Cartographie basée sur l'attribut sélectionné")
    
    


    url_to_geoparquet = "https://fatielkadd.github.io/app_streamlit/dataset_geoparquet_maroc.geoparquet"
    # Download the Parquet file
    response = requests.get(url_to_geoparquet)
    parquet_content = BytesIO(response.content)

    # Read the Parquet file with Geopandas
    gdf = gpd.read_parquet(parquet_content)

    def jenks_classifier(data, column, k=5):
        values = data[column].values
        breaks = jenks_breaks(values, k)
        return breaks

    A = '_Jour'

        # Sidebar for selecting the option (Attribute/Property)
    option = st.sidebar.radio("Choisir une option", ("PhenomeneJour-", "Propriété"))

    if option == "PhenomeneJour-":
            # List of attributes for selection
        attributs = ['temperature', 'elevation', 'humidity']
        selected_attribute = st.sidebar.selectbox("Sélectionner un attribut", attributs)

            # List of days for selection
        jours = [6, 5, 4, 3, 2, 1, 0]
        selected_day = st.sidebar.selectbox("Sélectionner un jour", jours)

            # Filter data based on selected attribute and day
        selected_column_day = f'{selected_attribute}{A}{selected_day}'
        filtered_data = gdf[(gdf[selected_column_day] >= 0)]
            # Reproject the geometries to a projected CRS (Web Mercator, EPSG:3857)

        # Create a Leaflet map centered on the average coordinates of the geometries
        m = leafmap.Map(locate_control=True, latlon_control=True, draw_export=True, minimap_control=True,zoom_start=5.5)
        basemap_options = list(leafmap.basemaps.keys())
        selected_basemap = st.sidebar.selectbox("Select a basemap:", basemap_options, basemap_options.index("OpenTopoMap"))
        m.add_basemap(selected_basemap)

        # Define the scale for proportional symbols
        scale = filtered_data[selected_column_day].max()

        # Obtain classes using Jenks classification
        breaks = jenks_classifier(filtered_data, selected_column_day, k=5)

        # Obtain classes using Jenks classification
        breaks = jenks_classifier(filtered_data, selected_column_day, k=5)

        # Create a dynamic legend HTML
        legend_html = f'''
            <div style="position: fixed; bottom: 50px; left: 50px; background-color: transparant; border: 2px solid grey; z-index: 9999; font-size: 14px;">
                <p><span style="background-color: blue; solid grey; border-radius: 50%; display: inline-block; height: 5px; width: 5px;"></span> 0-{breaks[1]:.2f}</p>
                <p><span style="background-color: blue; solid grey; border-radius: 50%; display: inline-block; height: 10px; width: 10px;"></span> {breaks[1]:.2f}-{breaks[2]:.2f}</p>
                <p><span style="background-color: blue; solid grey; border-radius: 50%; display: inline-block; height: 15px; width: 15px;"></span> {breaks[2]:.2f}-{breaks[3]:.2f}</p>
                <p><span style="background-color: blue; solid grey; border-radius: 50%; display: inline-block; height: 20px; width: 20px;"></span> {breaks[3]:.2f}-{breaks[4]:.2f}</p>
                <p><span style="background-color: blue; solid grey; border-radius: 50%; display: inline-block; height: 25px; width: 25px;"></span> {breaks[4]:.2f}-{scale:.2f}</p>
            </div>
        '''

        # Add the dynamic legend to the map
        m.add_html(html=legend_html)
        #marker_cluster = MarkerCluster().add_to(m)

        # Add data to the map as proportional symbols
        for idx, row in filtered_data.iterrows():
            popup = f"{selected_column_day}: {row[selected_column_day]}"
            folium.CircleMarker(
                location=[row['Geometry'].y, row['Geometry'].x],
                radius=(row[selected_column_day] / scale) * 10,
                popup=popup,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
            ).add_to(m)

        # Display the map in Streamlit
        m.fit_bounds([[27.6664, -13.1699], [35.8381, -0.9987]])
        folium_static(m,width=1000, height=600)

        # Histogram
        st.subheader(f'Histogramme de {selected_column_day}')
        chart_data = filtered_data[selected_column_day].dropna()
        chart = alt.Chart(chart_data.reset_index()).mark_bar(
            color='steelblue',  # Change bar color
            opacity=0.7,  # Adjust opacity
        ).encode(
            x=alt.X(f'{selected_column_day}:Q', bin=alt.Bin(maxbins=20)),
            y='count()',
            tooltip=['count()']
        ).properties(
            width=600,  # Adjust chart width
            height=300,  # Adjust chart height
        ).interactive()

        st.altair_chart(chart)
    else:
        # List of properties for selection
        proprietes = ['densité', 'croissance de la population']
        selected_property = st.sidebar.selectbox("Sélectionner une propriété", proprietes)

        # Classify values using the Jenks method
        breaks = jenks_classifier(gdf, selected_property, k=5)

        # Create a Leaflet map centered on the average coordinates of the geometries
        m = leafmap.Map(locate_control=True, latlon_control=True, draw_export=True, minimap_control=True,zoom_start=4.5)
        basemap_options = list(leafmap.basemaps.keys())
        selected_basemap = st.sidebar.selectbox("Select a basemap:", basemap_options, basemap_options.index("OpenTopoMap"))
        m.add_basemap(selected_basemap)

        # Define colors for classes
        colors = ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f','#006d2c']

        # Add data to the map with colors for classes
        for idx, row in gdf.iterrows():
            popup = f"{selected_property}: {row[selected_property]}"
            value = row[selected_property]
            class_idx = sum(value > i for i in breaks)
            color = colors[class_idx] if class_idx < len(colors) else '#006d2c'
            folium.CircleMarker(
                location=[row['Geometry'].y, row['Geometry'].x],
                radius=5,
                popup=popup,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
            ).add_to(m)
        colormap = LinearColormap( ['#edf8fb', '#b2e2e2', '#66c2a4', '#2ca25f','#006d2c'], vmin=0, vmax=100).to_step(10)
        colormap.add_to(m)

        # Display the map in Streamlit
        folium_static(m,width=1000, height=600)
        


        


# Ajoutez des blocs conditionnels pour les autres pages si nécessaire
elif selected == "Requetes ":
    st.header("Page 2 - Aspect 2")
    st.header("Requêtes sur les Données Géospatiales")

    
    selected1 = option_menu(None,["requétes attributaire","requétes spatiales "])
    if selected1=="requétes attributaire":
        st.subheader("Sous-objectif 1 : Filtrage attributaire")
        # Load geospatial data from GeoParquet file
        url_to_geoparquet = "https://fatielkadd.github.io/app_streamlit/dataset_geoparquet_maroc.geoparquet"
        # Download the Parquet file
        response = requests.get(url_to_geoparquet)
        parquet_content = BytesIO(response.content)

        # Read the Parquet file with Geopandas
        gdf = gpd.read_parquet(parquet_content)

        # Sidebar for filtering options
        st.sidebar.subheader("Filtrer les données")

        # Choix du type de filtrage
        filter_type = st.sidebar.radio("Choisir le type de filtrage", ["Par valeur", "Par intervalle de valeur"])

        if filter_type == "Par valeur":
            # Filtrer uniquement les colonnes numériques
            numeric_columns = gdf.select_dtypes(include='number').columns
            filter_column = st.sidebar.selectbox("Choisir une colonne numérique pour le filtre", numeric_columns)

            # Choisissez la valeur pour le filtre
            filter_value_input = st.sidebar.text_input(f"Valeur pour {filter_column}", str(gdf[filter_column].min()))
            filter_value = float(filter_value_input) if filter_value_input else gdf[filter_column].min()

            # Appliquer le filtre attributaire
            filtered_data = gdf[gdf[filter_column] == filter_value]

        elif filter_type == "Par intervalle de valeur":
            # Filtrer uniquement les colonnes numériques
            numeric_columns = gdf.select_dtypes(include='number').columns
            filter_column = st.sidebar.selectbox("Choisir une colonne numérique pour le filtre", numeric_columns)

            # Choisissez la plage pour le filtre
            min_value_input = st.sidebar.text_input(f"Valeur minimale pour {filter_column}", str(gdf[filter_column].min()))
            max_value_input = st.sidebar.text_input(f"Valeur maximale pour {filter_column}", str(gdf[filter_column].max()))

            min_value = float(min_value_input) if min_value_input else gdf[filter_column].min()
            max_value = float(max_value_input) if max_value_input else gdf[filter_column].max()

            # Appliquer le filtre attributaire
            filtered_data = gdf[(gdf[filter_column] >= min_value) & (gdf[filter_column] <= max_value)]

        else:
            # Autre type de filtrage (ajoutez votre logique de filtrage personnalisée ici)
            filtered_data = gdf

        # Reprojecter les géométries dans un CRS projeté (Web Mercator, EPSG:3857)
        
        # Créer une carte Leaflet centrée sur les coordonnées moyennes des géométries
        m = leafmap.Map(locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)
        basemap_options = list(leafmap.basemaps.keys())
        selected_basemap = st.sidebar.selectbox("Select a basemap:", basemap_options, basemap_options.index("OpenTopoMap"))
        m.add_basemap(selected_basemap)

        # Ajouter les données à la carte en tant que points
        for idx, row in filtered_data.iterrows():
            popup = f"{filter_column}: {row[filter_column]}"
            folium.Marker(
                location=[row['Geometry'].y, row['Geometry'].x],
                popup=popup,
                icon=folium.Icon(color='blue')
            ).add_to(m)

        # Afficher la carte dans Streamlit
        folium_static(m)
    if selected1=="requétes spatiales ":
        st.subheader("Sous-objectif 1 : Filtrage spatial")

        url_to_geoparquet = "https://fatielkadd.github.io/app_streamlit/dataset_geoparquet_maroc.geoparquet"
        # Download the Parquet file
        response = requests.get(url_to_geoparquet)
        parquet_content = BytesIO(response.content)

        # Read the Parquet file with Geopandas
        gdf = gpd.read_parquet(parquet_content)

        # Sidebar for spatial queries
        st.sidebar.header('Spatial Queries')
        m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)
        #   # Extraire les coordonnées de latitude et de longitude de la colonne Geometry
        gdf['Latitude'] = gdf['Geometry'].apply(lambda point: point.y)
        gdf['Longitude'] = gdf['Geometry'].apply(lambda point: point.x)

        st.write("Requêtes spatiales")
        uploaded_file = st.file_uploader("Uploader un fichier Shapefile", type=["shp"])
        # Print some debugging information
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_file_path = "C:\\Users\\hp\\Downloads\\Compressed\\streamlit-multipage-template-master\\hhhh.shp"

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            os.environ["SHAPE_RESTORE_SHX"] = "YES"

            # Read the file from the temporary location
            uploaded_gdf = gpd.read_file(temp_file_path)
            st.write("Nombre d'entités dans le fichier Shapefile chargé :", len(uploaded_gdf))
            st.write("Geometrie du fichier Shapefile chargé :", uploaded_gdf.geometry)
            

            # Bouton pour choisir si les points doivent être à l'intérieur ou à l'extérieur de la zone définie
            spatial_filter_choice = st.radio("Choisissez le filtre spatial :", ["À l'intérieur", "À l'extérieur"])

            # Appliquer le filtrage spatial
            if spatial_filter_choice == "À l'intérieur":
                filtered_data_spatial = gdf[gdf.geometry.within(uploaded_gdf.unary_union)]
            else:
                filtered_data_spatial = gdf[~gdf.geometry.within(uploaded_gdf.unary_union)]
            # Print some debugging information
            st.write("Nombre d'entités après le filtrage spatial :", len(filtered_data_spatial))
            st.write("Geometrie des entités après le filtrage spatial :", filtered_data_spatial.geometry)

            # Mettre à jour la carte avec les points filtrés spatialement
            st.map(filtered_data_spatial, latitude='Latitude', longitude='Longitude')
    
# Ajoutez les autres pages selon le même modèle
elif selected=="SplitMap":


    st.title("Split-panel Map")

    # Raw URLs to your GeoTIFF images on GitHub
    base_url = "https://fatielkadd.github.io/imagesraster/raster"

    # Options for each attribute
    attribute_options = {
        "elevation": list(range(7)),  # 0 to 6
        "temperature": list(range(1, 7)),  # 1 to 6
        "humidity": list(range(1, 7)),  # 1 to 6
    }

    # Allow user to choose attributes
    left_attribute = st.sidebar.selectbox("Select Left Attribute", list(attribute_options.keys()))
    right_attribute = st.sidebar.selectbox("Select Right Attribute", list(attribute_options.keys()))

    # Allow user to choose values within the selected attribute range
    left_value = st.sidebar.slider(f"Select Left {left_attribute} Value", min_value=0, max_value=6, value=0)
    right_value = st.sidebar.slider(f"Select Right {right_attribute} Value", min_value=0, max_value=6, value=0)

    # Construct the image URLs
    left_image_name = f"{left_attribute}_jour{left_value}.tif"
    right_image_name = f"{right_attribute}_jour{right_value}.tif"

    left_image_url = f"{base_url}/{left_image_name}"
    right_image_url = f"{base_url}/{right_image_name}"

    # Create a leafmap Map
    m = leafmap.Map()

    m.split_map(left_image_url, right_image_url)


    # Add legend (if needed)
    # m.add_legend(title='Your Legend Title', builtin_legend='Your_Builtin_Legend')

    # Render the map
    m.to_streamlit(height=700)
elif selected=="Map":
    st.title("visualisation des données ")
   # Load geospatial data from GeoParquet file
    url_to_geoparquet = "https://fatielkadd.github.io/app_streamlit/dataset_geoparquet_maroc.geoparquet"
    # Download the Parquet file
    response = requests.get(url_to_geoparquet)
    parquet_content = BytesIO(response.content)

    # Read the Parquet file with Geopandas
    gdf = gpd.read_parquet(parquet_content)

    A = '_Jour'

        # Sidebar for selecting the option (Attribute/Property)
    option = st.sidebar.radio("Choisir une option", ("PhenomeneJour-", "Propriété"))

    if option == "PhenomeneJour-":
            # List of attributes for selection
        attributs = ['temperature', 'elevation', 'humidity']
        selected_attribute = st.sidebar.selectbox("Sélectionner un attribut", attributs)

            # List of days for selection
        jours = [6, 5, 4, 3, 2, 1, 0]
        selected_day = st.sidebar.selectbox("Sélectionner un jour", jours)

            # Filter data based on selected attribute and day
        selected_column_day = f'{selected_attribute}{A}{selected_day}'
        filtered_data = gdf[(gdf[selected_column_day] >= 0)]

        # Reproject the geometries to a projected CRS (Web Mercator, EPSG:3857)

        # Create a Leaflet map centered on the average coordinates of the geometries
        m = leafmap.Map(locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)
        basemap_options = list(leafmap.basemaps.keys())
        selected_basemap = st.sidebar.selectbox("Select a basemap:", basemap_options, basemap_options.index("OpenTopoMap"))
        m.add_basemap(selected_basemap)
        for index, row in filtered_data.iterrows():
            data = {
                    'Jour': list(range(1, 7)),
                    'Elevation': row[[f'elevation_Jour{i}' for i in range(1, 7)]].values,
                    'Humidity': row[[f'humidity_Jour{i}' for i in range(1, 7)]].values,
                    'Temperature': row[[f'temperature_Jour{i}' for i in range(1, 7)]].values}
            df_chart = pd.DataFrame(data).melt('Jour')
            chart = alt.Chart(df_chart).mark_line().encode(
                    x='Jour',
                    y='value:Q',
                    color='variable:N',).properties(width=300, height=150)
            popup = folium.Popup(max_width=350).add_child(folium.VegaLite(chart, width=350, height=150))
            folium.CircleMarker(

                location=[row['Geometry'].y, row['Geometry'].x],
                radius=1,
                popup=popup,
                color='red',
                fill=True,
                fill_color='red',
                
            ).add_to(m)
        
        folium_static(m)
elif selected=="slider":
    st.title("slider Map")

    markdown = """
    Temp Jour 0 Tif Image in The Left 👈
    Elevation Jour 0 Tif Image in The Right 👉
    """

    st.sidebar.title("About")
    st.sidebar.info(markdown)
    logo = "https://i.imgur.com/UbOXYAU.png"
    st.sidebar.image(logo)

    st.title("Split-panel Map")

    # Raw URLs to your GeoTIFF images on GitHub
    base_url = "https://fatielkadd.github.io/imagesraster/raster"

    # Options for each attribute
    attribute_options = {
        "elevation": {"min": 0, "max": 20},
        "temperature": {"min": 0, "max": 100},
        "humidity": {"min": 0, "max": 50},
    }

    # Allow user to choose attributes
    left_attribute = st.selectbox("Select Left Attribute", list(attribute_options.keys()))

    # Allow user to choose values within the selected attribute range
    left_value = st.slider(f"Select Left {left_attribute} Value", min_value=0, max_value=6, value=0)

    # Construct the image URLs
    left_image_name = f"{left_attribute}_jour{left_value}.tif"

    #left_image_url = f"{base_url}/{left_image_name}"
    left_image_url = f"{base_url}/{left_image_name}"
    url='/vsicurl/' + left_image_url
# URL of the GeoTIFF image


    # Download the GeoTIFF image locally
    with rasterio.open(url) as dataset:
        data = dataset.read()
        img_array = reshape_as_image(data)
    # Get the bounds of the GeoTIFF
    bounds = [
        [dataset.bounds.bottom, dataset.bounds.left],
        [dataset.bounds.top, dataset.bounds.right],
    ]

    # Calculate the center of the bounding box
    center_lat = (dataset.bounds.bottom + dataset.bounds.top) / 2
    center_lon = (dataset.bounds.left + dataset.bounds.right) / 2

    # Create a Folium map centered at the image's midpoint
    map= folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add the GeoTIFF image as an overlay
    folium.raster_layers.ImageOverlay(
        img_array,
        bounds=bounds,
        colormap=lambda x: (0, 0, 0, x),
    ).add_to(map)

    # Fit the map to the bounds of the image
    map.fit_bounds(bounds)

    # Create colormap for the selected property
    min_value = attribute_options[left_attribute]["min"]
    max_value = attribute_options[left_attribute]["max"]

    colormap = LinearColormap(['#253494', '#00FF00', '#FFFF00', '#FF0000'], vmin=min_value, vmax=max_value).to_step(10)
    colormap.caption = f"{left_attribute.capitalize()} Legend"

    # Add colorbar to the map
    colormap.add_to(map)

    folium_static(map)
elif selected=="timeseries":

    markdown = """
    Temp Jour 0 Tif Image in The Left 👈
    Elevation Jour 0 Tif Image in The Right 👉
    """

    st.sidebar.title("About")
    st.sidebar.info(markdown)
    logo = "https://i.imgur.com/UbOXYAU.png"
    st.sidebar.image(logo)

    st.title("timeseries")


    def get_gif_path(property_name):
        # Dictionnaire associant chaque propriété à son chemin GIF correspondant
        property_gifs = {
            'timelapse_temperature': 'https://fatielkadd.github.io/gif/gif/timelapse_temperature.gif',
            'timelapse_humidity': 'https://fatielkadd.github.io/gif/gif/timelapse_humidity.gif',
            'timelapse_elevation': 'https://fatielkadd.github.io/gif/gif/timelapse_elevation.gif'
        }

        # Retourner le chemin du GIF pour la propriété spécifiée
        return property_gifs.get(property_name, None)

    # Créer la carte Folium
    m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)  # Coordonnées au centre du Maroc et niveau de zoom

    # Sélecteur pour choisir la propriété
    selected_property = st.selectbox("Choisissez une propriété", ['timelapse_temperature', 'timelapse_humidity', 'timelapse_elevation'])

    # Obtenir le chemin du GIF pour la propriété choisie
    gif_path = get_gif_path(selected_property)

    # Si le chemin du GIF existe, ajouter l'image (GIF) à la carte
    if gif_path:
        bounds_morocco = [
            [36, -17],  # Coin supérieur droit (nord-ouest)
            [20.8, -1]     # Coin inférieur gauche (sud-est)
        ]

        image = folium.raster_layers.ImageOverlay(
            image=gif_path,
            bounds=bounds_morocco,
            opacity=0.65,
            attr="Image from patricia_nasa",
        )

        image.add_to(m)

        # Ajuster la taille de la carte pour couvrir toute la zone spécifiée par les limites du Maroc
        m.fit_bounds(bounds_morocco)

        # Afficher la carte avec Streamlit
        folium_static(m)
    else:
        st.warning("Aucun GIF disponible pour la propriété sélectionnée.")
elif selected == "Timelapes":
    def get_gif_path(property_name):
        # Dictionnaire associant chaque propriété à son chemin GIF correspondant
        property_gifs = {
            'timelapse_temperature': 'https://fatielkadd.github.io/gif/gif/timelapse_temperature.gif',
            'timelapse_humidity': 'https://fatielkadd.github.io/gif/gif/timelapse_humidity.gif',
            'timelapse_elevation': 'https://fatielkadd.github.io/gif/gif/timelapse_elevation.gif'
        }

        # Retourner le chemin du GIF pour la propriété spécifiée
        return property_gifs.get(property_name, None)

    # Sélecteur pour choisir la propriété
    selected_property = st.selectbox("Choisissez une propriété", ['timelapse_temperature', 'timelapse_humidity', 'timelapse_elevation'])

    # Obtenir le chemin du GIF pour la propriété choisie
    gif_path = get_gif_path(selected_property)

    # Si le chemin du GIF existe, afficher l'image (GIF)
    if gif_path:
        st.image(gif_path, caption=f"{selected_property} gif", use_column_width=True)
    else:
        st.warning("Aucun GIF disponible pour la propriété sélectionnée.")
elif selected == "Explore COG":
    def get_tile_coordinates(cog_path, x_coord, y_coord):
        with rio.open(cog_path) as src:
            # Check if the coordinates are valid
            if 0 <= x_coord < src.width and 0 <= y_coord < src.height:
                # Calculate the coordinates of the tile
                tile_x = x_coord // src.profile['blockxsize'] * src.profile['blockxsize']
                tile_y = y_coord // src.profile['blockysize'] * src.profile['blockysize']
                return tile_x, tile_y
            else:
                return None

    def create_tile(cog_path, tile_coords, tile_size):
        with rio.open(cog_path) as src:
            # Define the window for the selected tile
            window = Window(tile_coords[0], tile_coords[1], tile_size, tile_size)

            # Read the tile from the COG
            tile = src.read(window=window, out_shape=(src.count, tile_size, tile_size))

            # Create a new GeoTIFF file for the selected tile
            output_path = "selected_tile.tif"
            profile = src.profile
            profile.update(width=tile_size, height=tile_size, transform=src.window_transform(window))
            with rio.open(output_path, 'w', **profile) as dst:
                dst.write(tile)

            return output_path

    # Set the path to the COG file
    jours = ["jour0", "jour1", "jour2", "jour3", "jour4", "jour5", "jour6"]
    attributs = ["temperature", "elevation", "humidity"]  # Ajoutez vos attributs ici

    # Interface utilisateur pour choisir le jour et l'attribut
    jour_selectionne = st.selectbox("Sélectionnez le jour", jours)
    attribut_selectionne = st.selectbox("Sélectionnez l'attribut", attributs)
    cog_path = f"https://fatielkadd.github.io/cog/cog1/{attribut_selectionne}_{jour_selectionne}.tif"

    st.title("Affichage de la Tuile COG sur une Carte")

    # Show the COG file using Folium
    m = folium.Map(location=[31.7917, -7.0926], zoom_start=5)

    # Obtenez les coordonnées de toutes les tuiles dans l'image COG
    with rio.open(cog_path) as src:
        coordonnees_tuiles = [(window.col_off, window.row_off) for ij, window in src.block_windows()]

    # Interface utilisateur pour choisir les coordonnées de la tuile parmi la liste des tuiles dans l'image COG
    tuile_selectionnee = st.selectbox("Sélectionnez les coordonnées de la tuile", coordonnees_tuiles)

    # Obtenez les coordonnées de la tuile
    x_coord, y_coord = tuile_selectionnee

    st.subheader("Coordonnées de la Tuile Sélectionnée:")
    st.write(f"Coordonnée X: {x_coord}, Coordonnée Y: {y_coord}")

    if tuile_selectionnee:
        st.subheader("Coordonnées de la Tuile Sélectionnée:")

        # Create and display the selected tile on the Folium map
        tile_size = 160  # Adjust the tile size as needed
        selected_tile_path = create_tile(cog_path, tuile_selectionnee, tile_size)

        with rio.open(selected_tile_path) as selected_tile:
            img = selected_tile.read()
            bounds = selected_tile.bounds

        bounds_fin = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        m.add_child(folium.raster_layers.ImageOverlay(img.transpose(1, 2, 0), opacity=.7, bounds=bounds_fin))

        # Display the Folium map
        folium_static(m)
    else:
        st.warning("Les coordonnées spécifiées sont en dehors de la plage valide.")
