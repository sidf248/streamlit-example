!pip install pandas numpy matplotlib joblib streamlit
!pip install tensorflow
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os 
import h5py



st.set_page_config(page_title="my_model_adam_blackbg_top_3", page_icon=":camera_flash:", layout="centered")
st.markdown("# second model🎈")
st.sidebar.success("select a model 🎈")

def main():
    st.header("my_model_adam_blackbg_top_3")

    image1 = st.camera_input("Take a picture")
    if image1 is not None:
        image1 = Image.open(image1)
        figure = plt.figure()
        plt.imshow(image1)
        plt.axis('off')
        
        top_3_predictions = predict_class(image1)

        top_3_predictions = sorted(top_3_predictions, key=lambda x:x[1], reverse=True)

        # Show the top 3 predictions in a table
        st.write("Top 3 Predictions:")
        
        st.pyplot(figure)#snapshot
        # Display the table with color images
        # table_data = []
        
        for prediction in top_3_predictions:
            st.write(prediction[0], prediction[1])
            st.image(prediction[2], width=200)
            
        
               
            # table_data.append([prediction[0], prediction[1], prediction[2]])
            # st.table(pd.DataFrame(table_data, columns=['Prediction', 'Score', 'Image']))
            
def predict_class(image1):
    #classifier_model = load_model('my_model.hdf5')
    #classifier_model = tf.keras.models.load_model(r'C:\Streamlitcode\pages\hdf5file\my_model_adam_blackbg_top_3.hdf5')
    classifier_model = tf.keras.models.load_model('https://deere-my.sharepoint.com/:u:/g/personal/guptasiddharth_johndeere_com/EUAs4NZ8zBxPqLgWbgFIRrMBTiKKAxfAKLt8m936lnRkpA?e=jfH4Tf')
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
    test_image = image1.resize((863,863))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    #class_names = ['t202639','t293782','t293833','t298427','t308680','t310081','t314893']
    class_names = ['t202639','t293782','t293833','t298427','t308680','t310081','t314893']

    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])

    scores = scores.numpy()
    # Get the top 3 classes and their corresponding probabilities
    top_3_classes = np.argsort(scores)[-3:]
    print(top_3_classes)
    top_3_probs = scores[top_3_classes]

    # Convert class indices to class names
    top_3_classes_names = [class_names[i] for i in top_3_classes]
    
    #filenames=['C:/Streamlitcode/tutorial1/trainingmodels/PRT_files/legend1/{}.png'.format(j) for j in top_3_classes]
    filenames=['https://deere-my.sharepoint.com/:f:/g/personal/guptasiddharth_johndeere_com/Ere4jImyzLlNtl4SblPL8ggB_2u8dP-fHXb-R5-uZIBMgQ?e=9a4Rcx/{}.png'.format(j) for j in top_3_classes]
    # Load the PNG images into a list
    images = [Image.open(filename) for filename in filenames]
    
    # Combine class names and probabilities into a list of tuples
    top_3_predictions = list(zip(top_3_classes_names, top_3_probs, images))
    
    return top_3_predictions

if __name__ == "__main__":
   main()
