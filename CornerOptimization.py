import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

pd.set_option('display.max_columns', None)
#Load the competition file
#Got this by searching 'how do I open json in Python'
with open('Statsbomb/data/competitions.json') as f:
    competitions = json.load(f)
    

pitchLengthX = 120
pitchWidthY = 80
#Load the list of matches for this competition
with open('Statsbomb/data/matches/9/27.json', encoding="utf8") as f:
    matches1 = json.load(f)
with open('Statsbomb/data/matches/11/27.json', encoding="utf8") as f:
    matches2 = json.load(f)
with open('Statsbomb/data/matches/2/27.json', encoding="utf8") as f:
    matches3 = json.load(f)
with open('Statsbomb/data/matches/7/27.json', encoding="utf8") as f:
    matches4  = json.load(f)    
matches= matches1 + matches2 + matches3 + matches4 

match_ids = []

for match in matches:
    match_id_required=str(match['match_id'])
    match_ids.append(match_id_required)
    
events = pd.DataFrame()

for match_id_required in match_ids:
    file_name=str(match_id_required)+'.json'   
    with open(r'DIRECTORY'+file_name, encoding="utf8") as data_file:
        data = json.load(data_file)
    data = json_normalize(data, sep = "_").assign(match_id = file_name[:-5])    
    events = events.append(data)
    
  
delays = 10
features_to_delay = ['shot_outcome_id']

def create_delayed_features(events, features_to_delay, delays):
    df_delays = [events[features_to_delay].shift(-step).add_suffix(f'+{step}') for step in (range(0, delays))]
    return pd.concat(df_delays, axis=1)

df_features = create_delayed_features(events, features_to_delay, delays)    

df_features['Next_Goals'] = np.where((df_features['shot_outcome_id+0']==97) | (df_features['shot_outcome_id+1']==97) | (df_features['shot_outcome_id+2']==97) | (df_features['shot_outcome_id+3']==97) | (df_features['shot_outcome_id+4']==97) | (df_features['shot_outcome_id+5']==97) | (df_features['shot_outcome_id+6']==97) | (df_features['shot_outcome_id+7']==97) | (df_features['shot_outcome_id+8']==97) | (df_features['shot_outcome_id+9']==97), 1, 0) 

delayed_events = pd.concat([events, df_features['Next_Goals']], axis=1)

corners = delayed_events[(delayed_events['pass_type_id']==61)]

corners[['x_end_location', 'y_end_location']] = pd.DataFrame(corners.pass_end_location.tolist(), index= corners.index)
corners[['x_start_location', 'y_start_location']] = pd.DataFrame(corners.location.tolist(), index= corners.index)
corners = corners.drop(['location', 'pass_end_location','related_events'], axis=1)



#CORNER ZONES
corners['zone_1'] = np.where((corners['y_end_location'] <= 22) | (corners['y_end_location'] >= 58), 1, 0 )
corners['zone_2'] = np.where((corners['x_end_location'] <= 102) & (corners['y_end_location'] < 58) & (corners['y_end_location'] > 22), 1, 0 )
corners['zone_3'] = np.where((corners['x_end_location'] > 102) & (((corners['y_end_location'] >= 22) & (corners['y_end_location'] <= 34)) | ((corners['y_end_location'] >= 46) & (corners['y_end_location'] <= 58))), 1, 0)
corners['zone_4'] = np.where((corners['x_end_location'] > 102) & (corners['x_end_location'] <= 108) & (corners['y_end_location'] > 34) & (corners['y_end_location'] < 46), 1, 0)
corners['zone_5'] = np.where((corners['x_end_location'] > 108) & (corners['x_end_location'] <= 114) & (corners['y_end_location'] > 34) & (corners['y_end_location'] < 46), 1, 0)
corners['zone_6'] = np.where((corners['x_end_location'] > 114) & (corners['y_end_location'] > 34) & (corners['y_end_location'] < 46), 1, 0)

corners_zone_1 = corners[(corners['zone_1'] == 1)]
corners_zone_2 = corners[(corners['zone_2'] == 1)]
corners_zone_3 = corners[(corners['zone_3'] == 1)]
corners_zone_4 = corners[(corners['zone_4'] == 1)]
corners_zone_5 = corners[(corners['zone_5'] == 1)]
corners_zone_6 = corners[(corners['zone_6'] == 1)]

goals_zone_1 = corners_zone_1.value_counts(corners_zone_1['Next_Goals'] == 1)
goals_zone_1 = goals_zone_1.to_frame()
goals_zone_1 = goals_zone_1.transpose()
goals_zone_1['Name'] = 'Zone 1'
goals_zone_1 = goals_zone_1.set_index('Name')

goals_zone_2 = corners_zone_2.value_counts(corners_zone_2['Next_Goals'] == 1)
goals_zone_2 = goals_zone_2.to_frame()
goals_zone_2 = goals_zone_2.transpose()
goals_zone_2['Name'] = 'Zone 2'
goals_zone_2 = goals_zone_2.set_index('Name')

goals_zone_3 = corners_zone_3.value_counts(corners_zone_3['Next_Goals'] == 1)
goals_zone_3 = goals_zone_3.to_frame()
goals_zone_3 = goals_zone_3.transpose()
goals_zone_3['Name'] = 'Zone 3'
goals_zone_3 = goals_zone_3.set_index('Name')

goals_zone_4 = corners_zone_4.value_counts(corners_zone_4['Next_Goals'] == 1)
goals_zone_4 = goals_zone_4.to_frame()
goals_zone_4 = goals_zone_4.transpose()
goals_zone_4['Name'] = 'Zone 4'
goals_zone_4 = goals_zone_4.set_index('Name')

goals_zone_5 = corners_zone_5.value_counts(corners_zone_5['Next_Goals'] == 1)
goals_zone_5 = goals_zone_5.to_frame()
goals_zone_5 = goals_zone_5.transpose()
goals_zone_5['Name'] = 'Zone 5'
goals_zone_5 = goals_zone_5.set_index('Name')

goals_zone_6 = corners_zone_6.value_counts(corners_zone_6['Next_Goals'] == 1)
goals_zone_6 = goals_zone_6.to_frame()
goals_zone_6 = goals_zone_6.transpose()
goals_zone_6['Name'] = 'Zone 6'
goals_zone_6 = goals_zone_6.set_index('Name')

goals_all_zones = pd.concat([goals_zone_1, goals_zone_2, goals_zone_3, goals_zone_4, goals_zone_5, goals_zone_6], axis=0)
goals_all_zones['Total Corners'] = goals_all_zones[True] + goals_all_zones[False]
goals_all_zones['Scoring Pct'] = (goals_all_zones[True]/goals_all_zones['Total Corners'])*100



#INSWING OR OUTSWING
corners['Inswing'] = np.where(corners['pass_technique_id'] == 104 , 1, 0)
corners['Straight'] = np.where(corners['pass_technique_id'] == 107, 1, 0)
corners['Outswing'] = np.where(corners['pass_technique_id'] == 105, 1, 0)

corners_inswing = corners[(corners["Inswing"] == 1)]
corners_straight = corners[(corners["Straight"] == 1)]
corners_outswing = corners[(corners["Outswing"] == 1)]

goals_inswing = corners_inswing.value_counts(corners_inswing['Next_Goals'] == 1)
goals_inswing = goals_inswing.to_frame()
goals_inswing = goals_inswing.transpose()
goals_inswing['Name'] = 'Inswing'
goals_inswing = goals_inswing.set_index('Name')

goals_straight = corners_straight.value_counts(corners_straight['Next_Goals'] == 1)
goals_straight = goals_straight.to_frame()
goals_straight = goals_straight.transpose()
goals_straight['Name'] = 'Straight'
goals_straight = goals_straight.set_index('Name')

goals_outswing = corners_outswing.value_counts(corners_outswing['Next_Goals'] == 1)
goals_outswing = goals_outswing.to_frame()
goals_outswing = goals_outswing.transpose()
goals_outswing['Name'] = 'Outswing'
goals_outswing = goals_outswing.set_index('Name')

goals_all_curves = pd.concat([goals_inswing, goals_straight, goals_outswing], axis=0)
goals_all_curves['Total Corners'] = goals_all_curves[True] + goals_all_curves[False]
goals_all_curves['Scoring Pct'] = (goals_all_curves[True]/goals_all_curves['Total Corners'])*100



#SHORT OR FAR POST
corners['Short_Side'] = np.where((corners['y_start_location'] == 0.1), 1, 0)
corners['Far_Side'] = np.where((corners['y_start_location'] == 80), 1, 0)
corners['Short_Half'] = np.where((corners['y_end_location'] <= 40), 1, 0)
corners['Far_Half'] = np.where((corners['y_end_location'] > 40), 1, 0)

corners_first_post_short = corners[(corners['Short_Side'] == 1) & (corners['Short_Half'] == 1)]
corners_first_post_far = corners[(corners['Far_Side'] == 1) & (corners['Far_Half'] == 1)]
corners_second_post_short = corners[(corners['Short_Side'] == 1) & (corners['Far_Half'] == 1)]
corners_second_post_far = corners[(corners['Far_Side'] == 1) & (corners['Short_Half'] == 1)]

goals_first_post_short = corners_first_post_short.value_counts(corners_first_post_short['Next_Goals'] == 1)
goals_first_post_short = goals_first_post_short.to_frame()
goals_first_post_short = goals_first_post_short.transpose()

goals_first_post_far = corners_first_post_far.value_counts(corners_first_post_far['Next_Goals'] == 1)
goals_first_post_far = goals_first_post_far.to_frame()
goals_first_post_far = goals_first_post_far.transpose()

goals_second_post_short = corners_second_post_short.value_counts(corners_second_post_short['Next_Goals'] == 1)
goals_second_post_short = goals_second_post_short.to_frame()
goals_second_post_short = goals_second_post_short.transpose()

goals_second_post_far = corners_second_post_far.value_counts(corners_second_post_far['Next_Goals'] == 1)
goals_second_post_far = goals_second_post_far.to_frame()
goals_second_post_far = goals_second_post_far.transpose()

goals_first_post = goals_first_post_short + goals_first_post_far
goals_first_post['Name'] = "Near Post"
goals_first_post = goals_first_post.set_index('Name')
goals_second_post = goals_second_post_short + goals_second_post_far
goals_second_post['Name'] = "Far Post"
goals_second_post = goals_second_post.set_index('Name')

goals_all_posts = pd.concat([goals_first_post, goals_second_post], axis=0,)
goals_all_posts['Total Corners'] = goals_all_posts[True] + goals_all_posts[False]
goals_all_posts['Scoring Pct'] = (goals_all_posts[True]/goals_all_posts['Total Corners'])*100

import matplotlib.pyplot as plt

#Barchart Curves
ax=plt.subplot()
plt.bar(goals_all_curves.index, goals_all_curves['Scoring Pct'], width=0.5, edgecolor="black")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title("Scoring Rate dependent on the curvature of Corners", y = 1.05,fontweight="bold", fontsize=14)
plt.text(0.23, 3.2, 'corners that become goals within 10 events', fontweight="medium", fontsize=10)
plt.text(1.5, -0.7, "Data:Statsbomb | @StatswithMax", fontsize=8)
ax.set_xlabel("Type of curvature", fontsize=12)
ax.set_ylabel("Scoring Rate (%)", fontsize=12)
plt.tight_layout()

plt.savefig("Graphs/CornerCurve", dpi=300)
plt.show()

#Barchart Posts
ax=plt.subplot()
plt.bar(goals_all_posts.index, goals_all_posts['Scoring Pct'], width=0.3, edgecolor="black")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title("Scoring Rate dependent on near or far Corners ", y = 1.05,fontweight="bold", fontsize=14)
plt.text(0.03, 3.2, 'corners that become goals within 10 events', fontweight="medium", fontsize=10)
plt.text(0.8, -0.7, "Data:Statsbomb | @StatswithMax", fontsize=8)
ax.set_xlabel("Post Selection", fontsize=12)
ax.set_ylabel("Scoring Rate (%)", fontsize=12)
plt.tight_layout()

plt.savefig("Graphs/PostSelection", dpi=300)
plt.show()

#Map Zones
import FCPython
import matplotlib.patches as patches
(fig,ax) = FCPython.createGoalMouth()
plt.xlim((-1,66))
plt.ylim((-3,30))
rect1 = patches.Rectangle((0,0), 12.5, 30, color = "wheat" )
rect2 = patches.Rectangle((52.2,0), 12.5, 30, color = "wheat")
rect3 = patches.Rectangle((12.5, 16.5), 40, 13.5, color="red")
rect4 = patches.Rectangle((12.5,0), 11, 16.5, color="orange")
rect5 = patches.Rectangle((41.5,0), 11, 16.5, color="orange")
rect6 = patches.Rectangle((23.5,0), 18, 5.5, color="coral")
rect7 = patches.Rectangle((23.5,5.5), 18, 5.5, color="tomato")
rect8 = patches.Rectangle((23.5,11), 18, 5.5, color= "yellow")
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
ax.add_patch(rect8)

ax.set_title("Scoring rate per Zone", y=1.05, fontweight="bold", fontsize=14)
plt.text(8, 31, "Goals scored after first pass from corner played to this zone", fontsize=8)
plt.text(40, -3, "Data:Statsbomb | @StatswithMax", fontsize=8)
plt.text(3, 15, "1,65%", fontweight="bold")
plt.text(56, 15, "1,65%", fontweight="bold")
plt.text(29, 23, "4,27%", fontweight="bold")
plt.text(15, 7.75, "2,43%", fontweight="bold")
plt.text(44, 7.75, "2,43%", fontweight="bold")
plt.text(29, 2, "3,22%", fontweight="bold")
plt.text(29, 7.75, "3,78%", fontweight="bold")
plt.text(29, 13, "2,27%", fontweight="bold")

plt.savefig("Graphs/ZoneRates", dpi=300)