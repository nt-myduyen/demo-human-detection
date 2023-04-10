import firebase_admin
from firebase_admin import credentials, storage, db
import requests
import datetime

# Initialize the Firebase app
cred = credentials.Certificate('E:\PBL5\\firebaseModule\\falldetect_new.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://falldetect-1c1b5-default-rtdb.firebaseio.com',
    'storageBucket': "falldetect-1c1b5.appspot.com",
})


def uploadVideoToStorageFirebase(video_path):
    # Get a reference to the Firebase Storage bucket
    bucket = storage.bucket()

    # Upload the video to Firebase Storage
    blob = bucket.blob('Fall_detection$' + str(datetime.datetime.now()))
    blob.upload_from_filename(video_path)

    # Get the download URL for the video
    expires_in = datetime.timedelta(hours=720)
    video_url = blob.generate_signed_url(expires_in)

    ref = db.reference('users')
    user_ref = ref.child('xiBWLpqpu3TIt7afsum57tLg2mu1')
    videos_ref = user_ref.child('videos')
    videos_ref.push(video_url)


uploadVideoToStorageFirebase('E:\PBL5\data\\adl-01-cam0.webm')