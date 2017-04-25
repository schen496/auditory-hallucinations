# 25/4/2017
# This piece of code will extract the frames from the directory of videos, greyscale and concatenate them. It will then
# save a single video to the corresponding index of an h5py file, and pair it with the correct audio vector.
# The script will incrementally add to the h5py such that the total output is:
# 'video_dset' = (112392,224,224,3)
# 'audio_dset' = (112392,18)

from extract_image_features.video_utils import *
import numpy as np

### SET TO TRUE IF USING TITANX LINUX MACHINE
USE_TITANX = False

### DEFINE OUTPUT DIRECTORY ###
if USE_TITANX:
    data_extern_dest = '/home/zanoi/ZANOI/auditory_hallucinations_data/TopAngle_data/'
else:  # Working on MacBook Pro
    data_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

data_file_name = data_extern_dest + 'TopAngleFinal_dataX_dataY.h5'

### LOADING VIDEOS ###
print ("--- Loading video and audio filenames...")
if USE_TITANX:
    video_dir = '/home/zanoi/ZANOI/auditory_hallucination_videos'
else: # Working on MacBook Pro
    video_dir = "/Volumes/SAMSUNG_SSD_256GB/ADV_CV/4-24_VIDAUD/EXPORTS"

video_files = [os.path.join(video_dir, file_i)
         for file_i in os.listdir(video_dir)
         if file_i.endswith('.mp4')]

num_videos = len(video_files)
print("num_videos: ", num_videos)

### LOADING AUDIO ###
audio_feature_dir = "../audio_vectors"

audio_f_files = [os.path.join(audio_feature_dir, file_i)
                for file_i in os.listdir(audio_feature_dir)
                if file_i.endswith('.mat')]

num_audio_f = len(audio_f_files)
print("num_audio_f: ", num_audio_f)

### MAIN FUNCTION LOOP ###
for i in range(num_audio_f):  # Loop over all audio files
    print ("--------------------{ }-----------------------")
    audio_prefix, audio_vector_length, audio_features = returnAudioVectors(i, audio_f_files)

    # Find all the linked videos for the given audio vector
    linked_video_f = findMatchingVideos(audio_prefix, video_files)
    print(audio_f_files[i])
    print(linked_video_f)

    for video_filename in linked_video_f:
        # Return the angle_name to name the file correctly
        angle_name = returnAngleName(video_filename)
        print ("angle_name:", angle_name)

        # Process the videos linked to a particular audio vector
        ######## PROCESS VIDEO TO BLACK AND WHITE
        print("--- Processing video to greyscale...")
        processed_video = processOneVideo(audio_vector_length, video_filename, normalize=False)
        print("processed_video.shape:", processed_video.shape)

        ######### CONCATENATE INTO SPACETIME IMAGE
        print ("--- Concatenating into Spacetime image...")
        window = 3
        space_time_images = createSpaceTimeImagesforOneVideo(processed_video,window) # (8377, 224, 224, 3)
        print ("space_time_image.shape:", space_time_images.shape)
        (num_frames, frame_h, frame_w, channels) = space_time_images.shape

        ########### CREATE FINAL DATASET, concatenate FC output with audio vectors
        # To avoid memory problems, we incrementally add to h5py file. A single video is processed and dumped to the h5py
        if i == 0:
            # If this is the first video file, you need to create the first entry matrix
            with h5py.File(data_file_name, 'w') as f:
                print ("Writing data to file...")
                video_dset = f.create_dataset("dataX", space_time_images.shape, maxshape=(None, frame_h, frame_w, channels))  # maxshape = (None, 224,224,3)
                video_dset[:] = space_time_images
                print("video_dset.shape:", video_dset.shape)

                # Normalization of the audio_vectors occurs in this function -> Hanoi forgot to normalize in MATLAB!!!!
                final_audio_vector = createAudioVectorDatasetForOneVid(audio_features,space_time_images.shape)  # (8377, 18)
                print("final_audio_vector.shape:", final_audio_vector.shape)
                audio_dset = f.create_dataset("dataY", final_audio_vector.shape, maxshape=(None, 18))
                audio_dset[:] = final_audio_vector
                print("audio_dset.shape:", audio_dset.shape)
        else:
            with h5py.File(data_file_name, 'a') as hf:
                print ("Writing data to file...")
                video_dset = hf['dataX']
                numFrames = space_time_images.shape[0]
                video_dset.resize(video_dset.len() + numFrames, axis=0)
                video_dset[-numFrames:] = space_time_images
                print("video_dset.shape:", video_dset.shape)

                # Normalization of the audio_vectors occurs in this function -> Hanoi forgot to normalize in MATLAB!!!!
                final_audio_vector = createAudioVectorDatasetForOneVid(audio_features, space_time_images.shape)  # (8377, 18)
                print("final_audio_vector.shape:", final_audio_vector.shape)
                audio_dset = hf['dataY']
                audio_dset.resize(audio_dset.len() + numFrames, axis=0)
                audio_dset[-numFrames:] = final_audio_vector
                print("audio_dset.shape:", audio_dset.shape)

        print ("Current video complete!")

print ("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")