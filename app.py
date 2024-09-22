import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
import re
import warnings
warnings.filterwarnings('ignore')

label_dict = {0: 'Apple scab',
 1: 'Apple Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}

def load_img(path):
    file_bytes = np.asarray(bytearray(path.read()), dtype=np.uint8)
    
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
   #mg = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img,axis=0)
    return img

st.title("Plant Disease Detection")

uploaded_img = st.file_uploader("Upload Image")
button = st.button("Detect")
if button:
    if uploaded_img is not None:
        try:
            img = load_img(uploaded_img)


            model = load_model("model.keras")
            pred = np.argmax(model.predict(img/255.0,verbose=False))

            plant_species = label_dict.get(pred,-1)
            plant_species = re.sub(r'_+', ' ', plant_species)
            plant_species = plant_species.split()
            plant,species_disease = plant_species[0],",".join(plant_species[1:])

            st.image(np.squeeze(img),caption="Plant Image")

            st.write(f"<h4>Plant:  {plant}<h4>",unsafe_allow_html=True)
            st.write(f"<h5>Species/Condition/Disease:  {species_disease}<h5>",unsafe_allow_html=True)

            st.session_state.prediction_done = True

            
        except Exception as e:
            st.error("Invalid Image")     
    else:
           st.warning("Please Upload an Image")
if st.session_state.get("prediction_done", False):
    # Reset the session state
    st.session_state.prediction_done = False
    uploaded_img = None