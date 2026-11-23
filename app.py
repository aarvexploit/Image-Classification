import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

data_cat = [
    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon',
]

img_width = 180
img_height = 180


@st.cache_resource
def get_model():
    return load_model('Image_Classification.keras')


model = get_model()

st.title('Fruit & Vegetable Classifier')

uploaded_file = st.file_uploader(
    'Upload an image', type=['jpg', 'jpeg', 'png', 'webp']
)

if uploaded_file is not None:
  image_load = tf.keras.utils.load_img(
      uploaded_file, target_size=(img_height, img_width)
  )
  img_arr = tf.keras.utils.img_to_array(image_load)
  img_batch = tf.expand_dims(img_arr, 0)

  predict = model.predict(img_batch)
  score = tf.nn.softmax(predict[0])

  st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
  st.write(
      'Image is **{}** with a confidence of **{:.2f}%**'.format(
          data_cat[np.argmax(score)], np.max(score) * 100
      )
  )
