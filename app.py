import json 
import streamlit as st                                                                                                  
import os 
from streamlit_option_menu import option_menu 
from streamlit_lottie import st_lottie 
from tensorflow import keras 
import numpy as np 
import cv2 
import os 
import pandas as pd 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
from PIL import Image 
from efficientnet.tfkeras import EfficientNetB0 

st.set_page_config(layout='wide') 

def cont(im1,t1,im2,t2,im3,t3): 
    """Displays an image and text side-by-side in a Streamlit app. 
    Args: 
        image_path (str): Path to the image file. 
        text (str): Text to display beside the image. 
    """ 
    col1, col2, col3 = st.columns(3)  # Adjust the number of columns for different layouts 
  # Load the image using st.image() 
    with col1: 
        st.image(im1, width=300)# Adjust width as needed 
        st.markdown(t1) 
  # Display the text using st.write() 
    with col2: 
        st.image(im2, width=300)# Adjust width as needed 
        st.markdown(t2) 
    with col3: 
        st.image(im3, width=300)# Adjust width as needed 
        st.markdown(t3) 
    # with col4: 
    #     st.image(im4, width=300)# Adjust width as needed                                                                                  
    #     st.markdown(t4) 
    #     st.markdown(t44) 

def get_file_type(file): 
    allowed_image_extensions = ['jpg', 'jpeg', 'png', 'gif'] 
    allowed_video_extensions = ['mp4', 'avi', 'mkv'] 
    file_extension = file.name.split('.')[-1].lower() 
    if file_extension in allowed_image_extensions: 
        return 'image' 
    elif file_extension in allowed_video_extensions: 
        return 'video' 
    else: 
        st.error(f"Unsupported file type: {file.name} ({file_extension})") 
        return 'unknown'   
def load_lottiefile(filepath: str): 
    """Loads a Lottie animation from a JSON file, handling potential encoding issues. 
    Args: 
        filepath (str): Path to the Lottie JSON file. 
    Returns: 
        dict: The loaded Lottie animation data. 
 
    Raises: 
        UnicodeDecodeError: If the encoding of the file cannot be determined. 
    """ 
    try: 
        # Attempt to open the file with UTF-8 encoding (most common) 
        with open(filepath, "r", encoding="utf-8") as f: 
            return json.load(f) 
    except UnicodeDecodeError: 
        # If UTF-8 fails, try common encodings for JSON files 
        for encoding in ("latin-1", "iso-8859-1"): 
            try: 
                with open(filepath, "r", encoding=encoding) as f: 
                    return json.load(f) 
            except UnicodeDecodeError:                                                                                               
                pass 
    # Raise an error if no encoding works 
    raise UnicodeDecodeError("Could not determine encoding of Lottie JSON file") 
try: 
    gif1 = load_lottiefile("code.json")  # Load the Lottie animation with error handling 
    gif2 = load_lottiefile("code2.json") 
    gif3 = load_lottiefile("new.json") 
except UnicodeDecodeError as e: 
    st.error(f"Error loading Lottie animation: {e}") 
    st.write("Please ensure your 'code.json' file is in a compatible encoding (e.g., UTF8).") 
else: 
    with st.container(): 
        st.write("##") 
        gif_column, text_column = st.columns((1, 2)) 
        with gif_column: 
            st_lottie( 
                gif1, 
                height=200 
                ) 
        with text_column: 
            st.markdown("## WElCOME TO APPLICATION ") 
 
st.markdown( 
    """ 
    <style> 
        body { 
            font-family: 'Arial', sans-serif; 
        } 
        h2 { 
            color: #007BFF; 
            font-size: 80px; 
        } 
    </style>                                                                                               
    """, 
    unsafe_allow_html=True 
) 

#-----------------------------------------------------------------------------------
    # Display skills in two columns 
# Main content 
with st.container(): 
    selected = option_menu( 
        menu_title=None, 
        options=['About', 'Projects', 'Credit'], 
        icons=['person', 'code-slash', 'credit-card-2-front', 'chat-left-text-fill'], 
        orientation="horizontal" 
    ) 
if selected == 'About': 
    with st.container(): 
        st.write("##") 
        st.markdown("# DEEP FAKE DETECTION") 
        st.write("##") 
        gif_column, text_column = st.columns((1, 2)) 
        with gif_column: 
            st_lottie( 
                gif3, 
                height=200 
                ) 
        with text_column: 
            st.markdown(""":small_blue_diamond:**Combating misinformation:** 
Deepfakes can be used to spread lies or propaganda, making it crucial to have tools to 
detect them.\n\n:small_blue_diamond:**Protecting reputations:** Deepfakes could be 
used to damage someone's image by putting them in compromising 
situations.Detection helps prevent this.\n\n:small_blue_diamond:**Building trust 
online:** By making it easier to spot deepfakes, deepfake detection can help maintain 
trust in the authenticity of online content.""") 
        text_cl , gif_cl = st.columns((2,1)) 
        with text_cl:                                                                                               
            st.markdown(":small_blue_diamond:**Combats fraud:**Deepfake detection helps law enforcement investigate deepfake-related crimes like CEO fraud, where impersonation is used for financial gain.\n\n:small_blue_diamond:**Protecting businesses:**Businesses can leverage deepfake detection to shield themselves from malicious attacks. Deepfakes can be used by competitors to damage a brand's image or sabotage marketing efforts. By identifying such manipulated content, businesses can take action to mitigate the harm before it impacts sales.")   
        with gif_cl: 
            st_lottie( 
                gif2, 
                height=200 
            )       
elif selected == "Projects": 
# Cache model load 
    @st.cache_resource 
    def load_trained_model(): 
        return load_model("./tmp_checkpoint/best_model.keras", 
custom_objects={'EfficientNetB0': EfficientNetB0}) 
    model = load_trained_model() 
    # Constants 
    INPUT_SIZE = 128 
    def preprocess_image(img): 
        img = img.resize((INPUT_SIZE, INPUT_SIZE)) 
        img_array = image.img_to_array(img) / 255.0 
        return np.expand_dims(img_array, axis=0) 
    def preprocess_frame(frame): 
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frame = frame.astype('float32') / 255.0 
        return np.expand_dims(frame, axis=0) 



#------------------------------------------------------------------------------------
    # App Interface 
    st.title("Deepfake Detector") 
    st.write("Upload an **video** to detect if it's real or fake.")   
                                                                                                   
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"], key="video") 
    if uploaded_video is not None: 
        st.video(uploaded_video) 
        st.info("Processing video frames...") 
            # Load video with OpenCV 
        temp_video_path = "temp_video.mp4" 
        with open(temp_video_path, "wb") as f: 
            f.write(uploaded_video.read()) 
        cap = cv2.VideoCapture(temp_video_path) 
        frame_preds = [] 
        frame_count = 0 
        while cap.isOpened(): 
            ret, frame = cap.read() 
            if not ret or frame_count > 100:  # Limit to 100 frames for speed 
                break 
            frame_count += 1 
            preprocessed = preprocess_frame(frame) 
            pred = model.predict(preprocessed)[0][0] 
            frame_preds.append(pred) 
        cap.release() 
        if frame_preds: 
            avg_pred = np.mean(frame_preds) 
            video_label = "Fake" if avg_pred > 0.007 else "Real" 
            confidence = avg_pred if avg_pred > 0.007 else 1 - avg_pred 
 
            st.subheader(f"Video Prediction: **{video_label}**") 
            st.write(f"Average Confidence: **{confidence * 100:.2f}%**") 
        else: 
            st.warning("No frames processed.") 
elif selected == 'Credit': 
    st.markdown("# Team Members") 
    im1="kiran.jpg" 
    t1="#### **M.Kiran Kumar**"                                                                                                    
    im2="phani.jpg" 
    t2="#### **G.Phaneendra**" 
    im3="kalyan.jpg" 
    t3="#### **G.Kalyan** "
    cont(im1,t1,im2,t2,im3,t3) 
st.markdown( 
    """ 
    <style> 
        body { 
            font-family: 'Arial', sans-serif; 
        } 
        h2 { 
            color: #007BFF; 
            font-size: 70px; 
        } 
        h1{ 
            text-align: center; 
        } 
        h3{ 
            color: #0078FF; 
            font-size: 50px; 
        } 
        h4{ 
            font-size: 40px; 
        } 
    </style> 
    """, 
    unsafe_allow_html=True 
) 
