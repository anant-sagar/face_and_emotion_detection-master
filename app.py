import os
from altair.vegalite.v4.api import value
from altair.vegalite.v4.schema.core import Align
from numpy.lib.shape_base import tile
from sqlalchemy.orm.session import Session
import streamlit as st
from PIL import Image
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from config import *
import matplotlib.pyplot as plt 
import db
import face_detection_utilities as fdu
from prediction import detect_face, detect_emotion,detect_emotion_in_video, detect_face_in_video
import cv2
import pandas as pd
import moviepy.editor as moviepy
import webcam


st.markdown('''
    <style>
        section.main{
            	background: linear-gradient(270deg, #cfadea, #9b51d5, #7b658c);
	            background-size: 400% 400%;
	            animation: gradient 7s ease infinite;
        }
        @keyframes gradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
    </style>
''',unsafe_allow_html=True)



def open_db():
    engine = create_engine(DB_PATH)
    Session = sessionmaker(bind=engine)
    return Session()

def convert_video(infile,outfile,ext):
    clip = moviepy.VideoFileClip(infile)
    path = os.path.join(VID_RESULT_FOLDER,f"{outfile}_output.{ext}")
    clip.write_videofile(path)
    return path

if not os.path.exists(IMAGE_FOLDER):
    os.mkdir(IMAGE_FOLDER)

#st.set_page_config(page_title="Mood Detection")

# variable
is_image_uploaded = False
im_path = None
vid_path = None
is_video_uploaded = False

# UI start here
st.title(TITLE)
choice = st.sidebar.radio("select options",MENU)

if choice=='About Project':
    st.markdown("""Humans share a universal and fundamental set of
emotions which are exhibited through consistent facial
expressions. An algorithm that performs detection, extraction,
and evaluation of these facial expressions will allow for automatic
recognition of human emotion in images and videos.""")
    st.image("abcd.gif")
    st.latex("Facial-Biometric")

    st.markdown("""This project is divided into 3 parts:""")
    st.markdown("""Facial Detection — Ability to detect the location of face in any input image or frame. 
    The output is the bounding box coordinates of the detected faces""")

    st.markdown("""Facial Recognition — Compare multiple faces together to identify which faces belong to the same person. 
    This is done by comparing face embedding vectors""")

    st.markdown("""Emotion Detection — Classifying the emotion on the face as happy, angry, sad, neutral, surprise, 
    disgust or fear""")

    st.title("Facial Detection")
    st.markdown("""Facial detection is the first part of our pipeline. 
    We have used the python library Face Recognition that we found easy to 
    install and very accurate in detecting faces.""")

    st.title("Facial Recognition")
    st.markdown("""Facial Recognition verifies if two faces are same. 
    The use of facial recognition is huge in security, bio-metrics, entertainment, 
    personal safety, etc. The same python library face_recognition used for face 
    detection can also be used for face recognition. Our testing showed it 
    had good performance. Given two faces match, they can be matched with 
    each other giving the result as True or False. The steps involved in 
    facial recognition are""")
    st.markdown("""Find face in an image""")
    st.markdown("""Analyze facial feature""")
    st.markdown("""Compare features for the 2 input faces""")
    st.markdown("""Returns True if matched or else False""")

    st.title("Interface")
    st.image("screenshots\\video.png")

if choice =='Upload Image':
    imgdata = st.file_uploader("select an image",type=['jpg','png','tif'])
    if imgdata:
        # load image as a Pillow object
        im = Image.open(imgdata)
        
        # create a address for image path
        path = os.path.join(IMAGE_FOLDER,imgdata.name)
        ext=imgdata.type.split('/')[1]
        # save file to upload folder
        im.save(path,format=ext)
        # saves info to db
        sess = open_db()
        imdb = db.Image(path=path)
        sess.add(imdb)
        sess.commit()
        sess.close()
        im_path = path
        is_image_uploaded = True
        # show a msg
        st.sidebar.image(im,use_column_width=True)
        st.success('image uploaded successfully')


if is_image_uploaded and im_path:
    ch2 = st.sidebar.selectbox("what do you want",SUB_MENU)
    if ch2 == 'detect face':
        st.subheader("face detection")
        bb = detect_face(im_path,FACE_MODEL)
        bb = cv2.cvtColor(bb,cv2.COLOR_BGR2RGB)
        st.image(bb,use_column_width=True)
        st.success('detection completed')
    if ch2 == 'detect face and mood':
        st.subheader("face and mood detection")
        with st.spinner("please wait, AI code working"):
            result = detect_emotion(img_path=im_path)
            if isinstance(result,list):
                frame = cv2.imread(im_path)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                bounding_box = result[0]["box"]
                emotions = result[0]["emotions"]
                cv2.rectangle(frame,(bounding_box[0], bounding_box[1]),(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 155, 255), 2,)
                df = pd.DataFrame(data=emotions.items(),columns=['mood','predicted_value'])
                st.sidebar.write(df)
                for idx, (emotion, score) in enumerate(emotions.items()):
                    color = (211, 211, 211) if score < 0.01 else (255, 0, 255)
                    emotion_score = f"{emotion}: {score:.2f}" if score > 0.01 else ""
                    cv2.putText(frame,emotion_score,(bounding_box[0], bounding_box[1] + bounding_box[3] + 10 + idx * 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
                st.image(frame,use_column_width=True)
                st.success('detection completed')
            else:
                st.error(result)

if choice =='Upload Video':
    videodata = st.file_uploader("select an image",type=['mp4','webm'])
    st.sidebar.markdown('''
    - face detection : will find faces in the video. It will start a popup window and you have to click on the window to view the output
    - face and mood detection : Please be carefull, very system intensive process. A one minute video will take 5 to 10 min to process.
    ''')
    if videodata:
        # create a address for image path
        path = os.path.join(VIDEO_FOLDER,videodata.name)
        ext=videodata.type.split('/')[1]
        # save file to upload folder
        
        with open(path,'wb') as f:
            f.write(videodata.getbuffer())
        # saves info to db
        sess = open_db()
        imdb = db.Video(path=path)
        sess.add(imdb)
        sess.commit()
        sess.close()
        vid_path = path
        is_video_uploaded = True
        st.sidebar.video(videodata)
        st.success('video uploaded successfully')     
if is_video_uploaded and vid_path:
    ch2 = st.sidebar.selectbox("what do you want",SUB_MENU_VIDEO)

    if ch2 == 'detect face in video':
        st.subheader("Please check the pop window")
        out = detect_face_in_video(vid_path,FACE_MODEL)
        st.write(out)
        root, ext = os.path.splitext(os.path.basename(out))
        new_path = convert_video(out,f'{root}_output{ext}','mp4')
        st.success("task completed")
        st.write(new_path)
        st.video(new_path)
        
    if ch2 == 'detect face and mood in video':
        st.subheader("face and mood detection in video")
        max_results =st.number_input('max results',min_value=10,max_value=1200,value=300)
        freq =st.number_input('frequency of frames',min_value=1,max_value=50,value=5)
        st.info("reduce this to 1 if you want full video analysis, and increase if you want to skip frames")
        with st.spinner("please wait, AI code working, take 5 to 10 mins or more"):
            result = detect_emotion_in_video(vid_path,max_results,freq)
            st.sidebar.write(result)
            if isinstance(result,list):
                root, ext = os.path.splitext(os.path.basename(vid_path))
                output = os.path.join(VID_RESULT_FOLDER, f"{root}_output{ext}")
                new_path = convert_video(output,root,'mp4')
                st.success("task completed")
                st.write(new_path)
                st.video(new_path)
            else:
                st.error(result)

if choice == 'Web CAM':
    st.sidebar.markdown('''
    - face detection : will find faces in the webcam. It will start a popup window and you have to click on the window to view the output
    ''')
    ch2 = st.sidebar.selectbox("what do you want",SUB_MENU_WEBCAM)
    if ch2 == 'detect face in webcam':
        try:
            st.subheader("Please check the pop window")
            out = webcam.detect_face_in_webcam(cascade_path=FACE_MODEL)
        except:
            st.subheader("This server does not have a camera")
            st.info("when deployed on system. camera wont work, it only works on pc, just for testing.")

if choice == 'Database':
    st.subheader('Database saved predictions')
    st.sidebar.image('project.png')
    sess = open_db()
    results = sess.query(db.Video).all()
    sess.close()
    if len(results):
        videos = list({item.path for item in results})
        video_path = st.sidebar.radio("Select a video",videos)
        if os.path.exists(video_path):
            col1,col2 = st.beta_columns(2)
            root, ext = os.path.splitext(os.path.basename(video_path))
            output = os.path.join(VID_RESULT_FOLDER, f"{root}_output{ext}")
            col1.video(video_path)
            col2.video(output)
        else:
            st.info('this file have been deleted')