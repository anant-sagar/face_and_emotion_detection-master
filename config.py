IMAGE_FOLDER = 'images'
DATA_FOLDER ='datasets'
DB_PATH ='sqlite:///database/db.sqlite3'
MODEL_FOLDER ='models'
VIDEO_FOLDER ='videos'
VID_RESULT_FOLDER ='video_results'
IMG_RESULT_FOLDER ='image_results'
WEBCAM = 0

FACE_MODEL = f'{MODEL_FOLDER}/face_finder.xml'
VGG_MODEL_WEIGHT = f'{MODEL_FOLDER}/model_weights.xml'

TITLE = "Face Recognizer & Emotion Detection"
MENU = ['About Project','Database','Upload Image','Upload Video','Web CAM',]
SUB_MENU = ['click here','detect face', 'detect face and mood']
SUB_MENU_VIDEO = ['click here','detect face in video','detect face and mood in video']
SUB_MENU_WEBCAM = ['click here','detect face in webcam','detect face and mood in webcam recording']