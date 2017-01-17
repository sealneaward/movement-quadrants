import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.pyplot as plt

###########################################
# Polar Conversions
###########################################
def cart2pol(row):
    x = row[0]
    y = row[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    row = [rho,phi]
    return row

def pol2cart(row):
    rho = row[0]
    phi = row[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    row = [x,y]
    return row


#####################################################################
# Plot Shot Chart:http://savvastjortjoglou.com/nba-shot-sharts.html
#####################################################################
def plot_shot(data):
    plt.figure(figsize=(12,11))
    plt.scatter(data.LOC_X, data.LOC_Y, c=data.shot_zone_range_area, s=30)
    draw_court()
    # Adjust plot limits to just fit in half court
    plt.xlim(-250,250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    plt.ylim(422.5, -47.5)
    # get rid of axis tick labels
    # plt.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('./data/img/half/fully_converted_with_range_areas.jpg')
    plt.close()

###########################################################################
# Visualization of court: http://savvastjortjoglou.com/nba-shot-sharts.html
###########################################################################
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

def label(file_name):
    ###########################################
    # Load data and map location categories
    ###########################################
    dictionary_1 = {'Right Side(R)':1, 'Left Side(L)':2, 'Center(C)':3, 'Right Side Center(RC)':4, 'Left Side Center(LC)':5}
    dictionary_2 = {'Less Than 8 ft.':1, '8-16 ft.':2, '16-24 ft.':3, '24+ ft.':4}
    dictionary_3 = {'Restricted Area':1, 'In The Paint (Non-RA)':2, 'Mid-Range':3, 'Right Corner 3':4, 'Left Corner 3':5, 'Above the Break 3':6}

    data = pd.read_csv('./data/train/data.csv')
    data['shot_zone_area'] = data['shot_zone_area'].map(dictionary_1)
    data['shot_zone_range'] = data['shot_zone_range'].map(dictionary_2)
    data['SHOT_ZONE_BASIC'] = data['SHOT_ZONE_BASIC'].map(dictionary_3)

    data = data.dropna(subset = ['LOC_X','LOC_Y','shot_zone_area', 'shot_zone_range'])

    data['SHOT_ZONE_BASIC'] = data['SHOT_ZONE_BASIC'].astype(int)
    data['shot_zone_range'] = data['shot_zone_range'].astype(int)
    data['shot_zone_area'] = data['shot_zone_area'].astype(int)

    ###################################################
    # Combine areas and ranges
    ###################################################
    data.loc[(data.shot_zone_area == 1) & (data.shot_zone_range == 1),'shot_zone_range_area'] = 1
    data.loc[(data.shot_zone_area == 1) & (data.shot_zone_range == 2),'shot_zone_range_area'] = 2
    data.loc[(data.shot_zone_area == 1) & (data.shot_zone_range == 3),'shot_zone_range_area'] = 3
    data.loc[(data.shot_zone_area == 1) & (data.shot_zone_range == 4),'shot_zone_range_area'] = 4

    data.loc[(data.shot_zone_area == 2) & (data.shot_zone_range == 1),'shot_zone_range_area'] = 5
    data.loc[(data.shot_zone_area == 2) & (data.shot_zone_range == 2),'shot_zone_range_area'] = 6
    data.loc[(data.shot_zone_area == 2) & (data.shot_zone_range == 3),'shot_zone_range_area'] = 7
    data.loc[(data.shot_zone_area == 2) & (data.shot_zone_range == 4),'shot_zone_range_area'] = 8

    data.loc[(data.shot_zone_area == 3) & (data.shot_zone_range == 1),'shot_zone_range_area'] = 9
    data.loc[(data.shot_zone_area == 3) & (data.shot_zone_range == 2),'shot_zone_range_area'] = 10
    data.loc[(data.shot_zone_area == 3) & (data.shot_zone_range == 3),'shot_zone_range_area'] = 11
    data.loc[(data.shot_zone_area == 3) & (data.shot_zone_range == 4),'shot_zone_range_area'] = 12

    data.loc[(data.shot_zone_area == 4) & (data.shot_zone_range == 1),'shot_zone_range_area'] = 13
    data.loc[(data.shot_zone_area == 4) & (data.shot_zone_range == 2),'shot_zone_range_area'] = 14
    data.loc[(data.shot_zone_area == 4) & (data.shot_zone_range == 3),'shot_zone_range_area'] = 15
    data.loc[(data.shot_zone_area == 4) & (data.shot_zone_range == 4),'shot_zone_range_area'] = 16

    data.loc[(data.shot_zone_area == 5) & (data.shot_zone_range == 1),'shot_zone_range_area'] = 17
    data.loc[(data.shot_zone_area == 5) & (data.shot_zone_range == 2),'shot_zone_range_area'] = 18
    data.loc[(data.shot_zone_area == 5) & (data.shot_zone_range == 3),'shot_zone_range_area'] = 19
    data.loc[(data.shot_zone_area == 5) & (data.shot_zone_range == 4),'shot_zone_range_area'] = 20

    # plot_shot(data)

    data = data.drop_duplicates(subset=['GAME_EVENT_ID','GAME_ID'], inplace=False)
    # data[['LOC_RHO','LOC_PHI']] = data[['LOC_X','LOC_Y']].apply(cart2pol, axis=1)

    X = data[['LOC_X','LOC_Y']]
    Y = data[['shot_zone_range_area']]
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0123)

    ####################################################################################
    # Train, Predict, Evaluate TODO: Serialize the classifier so code is more efficient
    ####################################################################################
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(train_x, train_y)
    predictions = knn.predict(test_x)

    # KNN is 99.56% accurate with single range, 99.07% accurate with two ranges
    print str(classification_report(test_y, predictions, digits=4))

    ######################################################################
    # Use KNN Model to label full converted movement set
    ######################################################################
    # read data and rename columns
    data = pd.read_csv('./data/converted/'+file_name)
    data[['LOC_X','LOC_Y']] = data[['x_loc','y_loc']]

    # predict and label shot zones
    X = data[['LOC_X','LOC_Y']]
    zones = knn.predict(X)
    data['shot_zone_range_area'] = zones
    plot_shot(data)

    # map real labels
    # data['range_area_basic'] = data['shot_zone_range_area']
    #
    # dictionary_1 = {1:'Right Side(R)', 2:'Left Side(L)', 3:'Center(C)', 4:'Right Side Center(RC)', 5:'Left Side Center(LC)'}
    # dictionary_2 = {1:'Less Than 8 ft.', 2:'8-16 ft.', 3:'16-24 ft.', 4:'24+ ft.'}
    # dictionary_3 = {1:'Restricted Area', 2:'In The Paint (Non-RA)', 3:'Mid-Range', 4:'Right Corner 3', 5:'Left Corner 3', 6:'Above the Break 3'}
    #
    # data['range_basic'] = data['range_basic'].map(dictionary_2)
    #
    # # get rid of excess data
    # data = data.drop('LOC_X', axis=1, inplace=False)
    # data = data.drop('LOC_Y', axis=1, inplace=False)
    # data = data.drop('shot_zone_range', axis=1, inplace=False)
    #
    # # write to labelled folder
    # data.to_csv('./data/label/'+file_name, index=False)

label('0021500139.csv')
