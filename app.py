import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_cropper import st_cropper
from PIL import Image
import PIL
import os

PIL.Image.MAX_IMAGE_PIXELS = 99999999999

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\openslide-win64-20221217\bin' #only this binary will work Ok you proceed now OK

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        from openslide import open_slide
        import openslide
else:
    from openslide import open_slide
    import openslide


def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image, model):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    predictions = model(image)
    confidence = tf.reduce_max(predictions, axis=1)
    idx = tf.argmax(predictions, axis=1).numpy()[0]
    labels = {0: 'Normal', 1: 'Non Amplified', 2: 'Amplified'}
    st.write(f'**Prediction:** *{labels[idx]}*')
    st.write(f'**Confidence:** *{(confidence.numpy()[0] * 100):.2f}%*')
    st.write('---')

def load_models():
    base_folder = 'D:/Paper Templates/Zaka HER2 WSI/cancer_region_assessment'
    try:
        model1 = tf.saved_model.load(f'{base_folder}/models/MobileNetBest')
        model2 = tf.saved_model.load(f'{base_folder}/models/DenseNet')
        return model1, model2
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

def main():
    st.markdown('''
    ### Weakly Supervised Deep Learning-Based Approach in Cancer Region Assessment from HER2-SISH Breast Histopathology Whole Slide Images
    ---  
    ''')

    model_names = ['MobileNet', 'DenseNet']
    model_name = st.sidebar.selectbox('Select Model', model_names)

    uploaded_file = st.sidebar.file_uploader('Choose an image', type=['svs', 'mrxs']) 

    if uploaded_file is not None:
        try:
            with st.spinner('Loading...'):
                temp_file_ext = uploaded_file.name.split('.')[-1].lower()
                temp_filename = f'temp_slide.{temp_file_ext}'

                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if temp_file_ext == 'mrxs':
                    slide = open_slide(temp_filename)
                    image = slide.get_thumbnail(size=(slide.level_dimensions[0][0], slide.level_dimensions[0][1]))
                elif temp_file_ext == 'svs':
                    slide = openslide.OpenSlide(uploaded_file)
                    image = slide.get_thumbnail(size=(slide.level_dimensions[0][0] // 50, slide.level_dimensions[0][1] // 50))
                else:
                    st.write('Something Went Wrong!')
                    return

                image = st_cropper(image, aspect_ratio=(1, 1))

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption='Selected Region', use_column_width=True)

                with col2:
                    if st.sidebar.button('Pre-Process'):
                        image = preprocess_image(image)
                        st.image(image[0], caption='Pre-Processed Image', use_column_width=True)

                    if st.sidebar.button('Predict'):
                        st.write('Making Prediction...!!!')
                        models = load_models()
                        if models:
                            model = models[model_names.index(model_name)]
                            predict_image(preprocess_image(image)[0], model)
        except openslide.OpenSlideUnsupportedFormatError:
            st.error("Unsupported or missing image file. Please upload a valid whole slide image file in SVS or MRXS format.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.sidebar.write('---')
    st.sidebar.write('Author: **Zaka Ur Rehman**')

if __name__ == '__main__':
    main()
