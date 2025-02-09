from deepface import DeepFace
from PIL import Image
import numpy as np
import streamlit as st


st.title("Human Emotion Reco")


def analysis(img):
    results = DeepFace.analyze(img, actions=['age', 'gender','emotion'])
    return results[0]



upload = st.file_uploader("Choose file1", type=["png", "jpg", "jpeg", "webp"])
upload1 = st.file_uploader("Choose file2", type=["png", "jpg", "jpeg", "webp"])


if upload1 is not None and upload1 is not None:
    
    img = Image.open(upload).convert('RGB')
    img_np = np.array(img)
    
    img1 = Image.open(upload1).convert('RGB')
    img_np1 = np.array(img1)

    col1, col2 = st.columns(2)
    
    results = analysis(img_np)
    emotion = max(results['emotion'], key=results['emotion'].get)
    gender = max(results['gender'],key=results['gender'].get)
    results1 = analysis(img_np1)
    emotion1 = max(results1['emotion'], key=results1['emotion'].get)
    gender1 = max(results1['gender'],key=results1['gender'].get)
    
    with col1:
        st.image(img, channels="RGB",caption="Person 1",use_column_width=True)
        st.write("Detected Emotion:", emotion)
        st.write("Gender:",gender)
        st.write("Age:",results['age'])
    with col2:
        st.image(img1, channels="RGB",caption="Person 2",use_column_width=True)
        st.write("Detected Emotion:", emotion1)
        st.write("Gender:",gender1)
        st.write("Age:",results1['age'])
    
    verification_result = DeepFace.verify(img1_path=np.array(img), img2_path=np.array(img1), enforce_detection=False)


    
    st.write("They are similarly:",verification_result['verified'])
    
    
