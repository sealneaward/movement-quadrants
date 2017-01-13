import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.mixture import GaussianMixture
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

#######################################
# Individual shot plot function
#######################################
def plot_shot(x,y,quadrant, event_id):
    plt.figure(figsize=(12,11))
    plt.scatter(x, y, c=['red'], s=30)
    draw_court()
    # Adjust plot limits to just fit in half court
    plt.xlim(-250,250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    plt.ylim(422.5, -47.5)
    # get rid of axis tick labels
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('./data/img/'+str(quadrant)+'/'+str(event_id)+'.jpg')
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

###########################################
# Load data and map location categories
###########################################
dictionary = {'Restricted Area':1, 'In The Paint (Non-RA)':2, 'Mid-Range':3, 'Right Corner 3':4, 'Left Corner 3':5, 'Above the Break 3':6}
data = pd.read_csv('./data/train/data.csv')
data['SHOT_ZONE_BASIC'] = data['SHOT_ZONE_BASIC'].map(dictionary)
data = data.dropna(subset = ['LOC_X','LOC_Y','SHOT_ZONE_BASIC'])
data['SHOT_ZONE_BASIC'] = data['SHOT_ZONE_BASIC'].astype(int)
data = data.drop_duplicates(subset=['GAME_EVENT_ID','GAME_ID'], inplace=False)
data[['LOC_RHO','LOC_PHI']] = data[['LOC_X','LOC_Y']].apply(cart2pol, axis=1)

events = data['GAME_EVENT_ID'].values

# X = data[['LOC_X','LOC_Y','LOC_RHO','LOC_PHI','GAME_EVENT_ID']]
X = data[['LOC_X','LOC_Y']]
# X = data[['LOC_RHO','LOC_PHI']]

# print X.head(5)

Y = data[['SHOT_ZONE_BASIC']]
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.5, random_state=0123)

###########################################
# Train, Predict, Evaluate
###########################################
print 'X shape: ' + str(train_x.shape)
print 'Y shape: ' + str(train_y.shape)

# svm = LinearSVC()
# svm.fit(train_x, train_y)
# predictions = svm.predict(test_x)

# gmm = GaussianMixture(n_components=5)
# gmm.fit(train_x, train_y)
# predictions = gmm.predict(test_x)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x, train_y)
predictions = knn.predict(test_x)

print str(classification_report(test_y, predictions, digits=4))

#####################################################################
# Plot Shot Chart:http://savvastjortjoglou.com/nba-shot-sharts.html
#####################################################################
# for event in events:
#     event_data = data[data['GAME_EVENT_ID'] == event]
#     x = event_data['LOC_X'].values[0]
#     y = event_data['LOC_Y'].values[0]
#     quadrant = event_data['SHOT_ZONE_BASIC'].values[0]
#
#     plot_shot(x,y,quadrant,event)

# TODO: Add method to label movement data
