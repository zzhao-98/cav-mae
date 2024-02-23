import os, cv2, glob
import numpy as np
import pandas as pd
import re, time
from moviepy.editor import VideoFileClip

# Define the dataset folder
dir_dataset = "C:/Users/36394/Study/GWU/PHD in Biomedical Engineer/Research/FOS/Autism_dataset"
dir_raw = os.path.join(dir_dataset, 'Data')
dir_save_video = os.path.join(dir_dataset, 'Data_10s_clips')
if not os.path.exists(dir_save_video):
    os.makedirs(dir_save_video, exist_ok=False)

# Find all the video paths
g = os.walk(dir_raw)
paths_video = []
for path,dir_list,file_list in g:
    for file_name in file_list:
        if file_name[-4:] != 'xlsx' and file_name[-4:]!= '.zip':
            paths_video.append(os.path.join(path, file_name))

# Find the good videos (duration < 15) and without Korean characters in the name.
videos = {'name': [], 'duration': []}
for path_video in paths_video:
    cap = cv2.VideoCapture(path_video)
    if cap.isOpened():
        rate = cap.get(5)
        FrameNumber = cap.get(7)
        duration = (FrameNumber/rate) / 60
        name = path_video.split('Data\\')[1]
        videos['name'].append(name)
        videos['duration'].append(duration)
df_videos = pd.DataFrame(videos)
# Drop all the videos with duration longer than 15 minutes.
print(len(df_videos))
df_videos = df_videos.loc[df_videos['duration'] < 15]
df_videos = df_videos.sort_values('duration')
print(len(df_videos))
# Drop all the videos with Korean characters in the name.
df_videos = df_videos.loc[df_videos['name'].str.contains('[ㄱ-ㅎㅏ-ㅣ가-힣]') == False]
print(len(df_videos))
# Sort the index
df_videos = df_videos.reset_index(drop=True)
df_good_videos = df_videos.loc[(8 < df_videos['duration']) & (df_videos['duration'] < 15)]

# Process each good video
dict_processed = {'path': [], 'labels': []}
for count_video, name in enumerate(df_videos.name.tolist()):
    start_time = time.time()
    path_video = os.path.join(dir_raw, name)
    path_label = glob.glob(os.path.join(os.path.dirname(path_video), '*.xlsx'))[0]
    names_sheet = pd.ExcelFile(path_label).sheet_names
    if re.search(r"Alone", os.path.basename(path_video), re.IGNORECASE):
        df_label = pd.read_excel(path_label, sheet_name="alone" if "alone" in names_sheet else "Alone")
    elif re.search(r"instruction", os.path.basename(path_video), re.IGNORECASE):
        df_label = pd.read_excel(path_label,
                                 sheet_name="instruction" if "instruction" in names_sheet else "Instruction")
    elif re.search(r"playtime", os.path.basename(path_video), re.IGNORECASE):
        df_label = pd.read_excel(path_label, sheet_name="playtime" if "playtime" in names_sheet else "Playtime")
    else:
        print("Bad data here, need to fixed: print {}".format(path_video))
        break

    # Find the row that contains all the table head.
    index_head = 0
    index_tail = 0
    for index, col in df_label.items():
        index_head_tail = col[col == 'Time'].index
        if len(index_head_tail) > 0:
            index_head = index_head_tail[0]
            index_tail = index_head_tail[1]
            break

    # Check head and tail error:
    if index_head + index_tail < 20:
        print('Something wrong for the head and tail index.')
        break

    # Change the head of the dataframe
    df_label.columns = df_label.iloc[index_head].values

    # Crop the df_label to keep valuable information
    ind_col_head, ind_col_tail = np.where(df_label.columns == 'Time')[0]
    df_label = df_label.iloc[index_head + 1:index_tail]
    df_label = df_label.iloc[:, ind_col_head:ind_col_tail]

    # Change the duplicate header names (eg: Int to Int_parent and Int_child)
    index_Int = np.where(df_label.columns == 'Int')[0]
    df_label.columns.values[index_Int[0]] = 'Int_parent'
    df_label.columns.values[index_Int[1]] = 'Int_child'
    index_Int = np.where(df_label.columns == 'Aff')[0]
    df_label.columns.values[index_Int[0]] = 'Aff_parent'
    df_label.columns.values[index_Int[1]] = 'Aff_child'

    # Check if the label is empty
    if df_label.drop(columns=['Time']).fillna(0).applymap(lambda x: 0 if x != 1 else x).values.sum() < 10:
        print("There is no label for this video: {}".format(path_video))
        continue

    # Crop the label to match the video
    match = re.search(r'(alone|playtime|instructions)\s*(\d+)', name, re.IGNORECASE)
    duration = df_videos.loc[df_videos['name'] == name, 'duration'].values[0]
    if re.search(r"interaction", name, re.IGNORECASE):
        df_label = df_label[0:30]  # First 5 minutes
    elif re.search(r"activity", name, re.IGNORECASE):
        df_label = df_label[30:48]  # 5 - 8 minutes
    elif re.search(r"toy", name, re.IGNORECASE):
        df_label = df_label[48:60]  # 8 - 10 minutes
    elif re.search(r"clean", name, re.IGNORECASE):
        df_label = df_label[0:12]  # First 2 minutes
    elif re.search(r"list", name, re.IGNORECASE):
        df_label = df_label[12:30]  # 2 - 5 minutes
    elif re.search(r"freely", name, re.IGNORECASE):
        df_label = df_label[30:60]  # 5 - 10 minutes
    elif match:  # Crop the label to match the video if the video is like "Alone 1" or "Playtime 2"
        num_video = int(match.group(2))
        if duration < 4:
            print("The video {} is too short, skip it.".format(name))
            continue
        if num_video == 1:
            df_label = df_label[0:30]  # First 5 minutes
        elif num_video == 2:
            df_label = df_label[30:60]  # 5 - 10 minutes
        else:
            print("Something wrong with the video name: {}, break!!!!!!!!".format(name))
            break
    elif name in df_good_videos.name.tolist():  # Good video's label doesn't need to be cropped
        pass
    else:
        print("Something wrong with the video name: {}, can't be processed".format(name))
        continue

    ###
    # Preprocess the video:
    video = VideoFileClip(path_video)
    duration = video.duration

    count_10s = 0
    start_time = 0
    for index, row in df_label.iterrows():
        end_time = min(start_time + 10, duration)
        clip = video.subclip(start_time, end_time)

        # Processing the video saving path (original)
        # dir_save_video = os.path.dirname(path_video.replace('Data', '10s_clips_data'))
        # if not os.path.exists(dir_save_video):
        #     os.makedirs(dir_save_video, exist_ok=False)
        # name_save_video = '{}_{}.mp4'.format(os.path.basename(name)[:-4], count_10s)
        # count_10s += 1
        # path_save_video = os.path.join(dir_save_video, name_save_video)
        # clip.write_videofile(path_save_video)

        # Processing the video saving path (modified)
        name_subject = os.path.dirname(path_video.split('Data' + os.sep)[1]).replace(os.sep, "_")
        name_save_video = '{}_{}_{}.mp4'.format(name_subject, os.path.basename(name)[:-4], count_10s)
        count_10s += 1
        path_save_video = os.path.normpath(os.path.join(dir_save_video, name_save_video))
        clip.write_videofile(path_save_video)

        # Store the processed video path and labels
        labels = [i for i in row.index if row[i] == 1]
        dict_processed['path'].append(path_save_video)
        dict_processed['labels'].append(labels)
    ###

    df_processed = pd.DataFrame(dict_processed)
    df_processed.to_csv('10s_av_clips_with_labels.csv', index=False)
    print("Finish processing video: {}, time cost:{:.2f}min. ({}/{})".format(name,(time.time()-start_time)/60, count_video+1,len(df_videos)))
