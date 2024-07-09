# The purpose here is to take a data table of each annotation for the different files, and then split these up.
import csv
import pandas as pd
"""
video_annos = {'video_validation_0000051': [[39, 44, 3], [50, 53, 3], [82, 87, 3], [88, 89, 3]]}

video_annos_new = {}
for key in video_annos.keys():
    for i in range(len(video_annos[key])):
        value_new = video_annos[key][i] + [-1, -1]

        value_new[3] = 0 if i == 0 else (value_new[0] + video_annos[key][i-1][1]) // 2

        video_annos_new[key + '-' + str(i)] = [value_new]

print(video_annos_new)
"""

video_info = pd.read_csv('/root/models/SPOTworktree/data/thumos_annotations/val_video_info.csv')

with open('/root/models/SPOTworktree/data/thumos_annotations/val_Annotation.csv') as read_obj:
    csv_reader = csv.reader(read_obj)
    df = list(csv_reader)


df[0] = df[0] + ['clipStartFrame', 'clipEndFrame']
counter = 0
for r in range(1, len(df)):

    df[r] = df[r] + [-1, -1]

    # video name
    if df[r][0] != df[r-1][0].split('-')[0]:
        counter = 0

    df[r][0] = df[r][0] + '-' + str(counter) 

    #clipStartFrame
    df[r][7] = 0 if counter == 0 else (int(df[r][5]) + int(df[r-1][6])) // 2
    
    counter += 1


for r in range(1, len(df)-1):
    df[r][8] = int(video_info[video_info['video'] == df[r][0].split('-')[0]]['count'].item()) if df[r][0].split('-')[0] != df[r+1][0].split('-')[0] else int(df[r+1][7])-1 # HACK: replace with the length of the original video
df[len(df)-1][8] = int(video_info[video_info['video'] == df[r][0].split('-')[0]]['count'].item())


new_video_info = pd.DataFrame(df[1:], columns=df[0])

# clipStartFrame, clipEndFrame show where we want to make our cuts, so we use these to help us with getting what we want from the numpy array.

new_video_info['clipLength'] = new_video_info['endFrame'].astype(int) - new_video_info['startFrame'].astype(int) # The length of the new video clip.

# the new startFrame, endFrame for the annotation in the clipped video.
new_video_info['newStartFrame'] = new_video_info['startFrame'].astype(int) - new_video_info['clipStartFrame']
new_video_info['newEndFrame'] = new_video_info['endFrame'].astype(int) - new_video_info['clipStartFrame']



# With pandas
counter = 0
dfp = pd.read_csv('/root/models/SPOTworktree/data/thumos_annotations/val_Annotation.csv')
dfp['clipStartFrame'] = -1
dfp['clipEndFrame'] = -1

for r in range(0, len(df)-1):
    if dfp.loc[r, 'video'] != dfp.loc[max(r-1, 0), 'video'].split('-')[0]:
        counter = 0 
    dfp.loc[r, 'video'] = dfp.loc[r, 'video'] + '-' + str(counter)
    counter += 1
