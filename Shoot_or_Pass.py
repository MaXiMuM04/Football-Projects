#Download Event data:

#Create new parameters
events = events[events["location"].notna()]   
events[['starting_x', 'starting_y']] = pd.DataFrame(events.location.tolist(), index = events.index)
events['intercept'] = 1
events['calc_x'] = 120 - events['starting_x']
events['calc_x_sq'] = events['calc_x']**2
events['calc_c'] = abs(events['starting_y'] - 40)
events['calc_c_sq'] = events['calc_c']**2
events['distance_sq'] = (events['calc_x'])**2 + (events["calc_c"])**2
events['distance'] = np.sqrt(events['distance_sq'])
events['left_post_sq'] = (events['starting_y']-36.34)**2 + events['calc_x_sq']**2
events['right_post_sq'] = (events['starting_y']-43.66)**2 + events['calc_x_sq']**2
#CHECK ANGLE
events["angle"] = np.arccos((events['left_post_sq'] + events['right_post_sq'] - 7.32**2)/(2*np.sqrt(events['left_post_sq'])*np.sqrt(events['right_post_sq'])))
events['goal'] = np.where(events['shot_outcome_id'] == 97, 1, 0)
events = events[(events.type_id == 16) | (events.type_id == 30)]
events['pos_goal'] = 0
events = events.dropna(subset = 'angle')

goal_data = []
goals = events[events.shot_outcome_id == 97]

for index, row in goals.iterrows():
    data = [row['possession'], row['match_id']]
    goal_data.append(data)

events = events.reset_index()

for index,row in events.iterrows():
    data_play =  [row['possession'], row['match_id']]
    if data_play in goal_data:
        events.loc[index,'pos_goal'] = 1

    
shots = events[(events.type_id == 16)]
shots = shots[shots.shot_type_id == 87]
y_shot = shots['goal']
x_cols = ['intercept', 'calc_x', 'calc_x_sq', 'calc_c', "calc_c_sq", 'distance_sq', 'distance', 'angle']
X_shot = shots[x_cols]
probit_xG = smf.Probit(y_shot,X_shot)
result_xG = probit_xG.fit()
print(result_xG.summary2())
params_xG = pd.DataFrame(result_xG.params)


passes = events[events.type_id == 30]
final_third_passes = passes[passes.starting_x >= 80]
final_third_passes = final_third_passes[final_third_passes.pass_outcome_id != 77]
y_pass = final_third_passes['pos_goal']
X_pass = final_third_passes[x_cols]
probit_pass = smf.Probit(y_pass, X_pass)
result_pass = probit_pass.fit()
print(result_pass.summary2())
params_pass = pd.DataFrame(result_pass.params)

final_third = events[events.starting_x >= 80]
X = final_third[x_cols]
final_third['xG'] = si.norm.cdf(np.matmul(X, params_xG))
final_third['xPG'] = si.norm.cdf(np.matmul(X, params_pass))
final_third['shot_benefit'] = final_third['xG']-final_third['xPG']

#Histogram
a = final_third['starting_x']
b = final_third['starting_y']
d = final_third['shot_benefit']

y_bins = np.linspace(80, 120, 40)
x_bins = np.linspace(0, 80, 80)

H, xedges, yedges = np.histogram2d(b, a, bins=[x_bins, y_bins], weights= d)
H_counts, xedges, yedges = np.histogram2d(b, a, bins=[x_bins, y_bins])
H = H/H_counts

#Create Graph
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.imshow(H.T, origin='lower', cmap='seismic', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin = -0.2, vmax = 0.2)
#Penalty Area
plt.plot([20,60], [104,104], color = "black")
plt.plot([20,20], [104,120], color = "black")
plt.plot([60,60], [104,120], color = "black")
arc = Arc((40,104), width = 16.3, height = 6, theta1= 180, theta2= 360, color = "black", linewidth = 1.5)
ax.add_patch(arc)
#6-yard box
plt.plot([30.8,49.2], [114,114], color = "black")
plt.plot([30.8,30.8], [114,120], color = "black")
plt.plot([49.2,49.2], [114,120], color = "black")
# 
plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
plt.text(30, 122, "Should you", fontsize = 'large', color = 'black', ha = 'center')
plt.text(43, 122, "Shoot", fontsize = 'large', color = 'red', ha = 'center')
plt.text(49.5, 122, "or", fontsize = 'large', color = 'black', ha = 'center')
plt.text(55, 122, "Pass", fontsize = 'large', color = 'blue', ha = 'center')
plt.text(59, 122, "?", fontsize = 'large', color = 'black', ha = 'center')
plt.text(63, 77, "Data:Statsbomb | @StatswithMax", fontsize = 'small', color = "black", ha = 'center')
