import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

##################################################################################################
# Plot a series of movement positions:http://savvastjortjoglou.com/nba-play-by-play-movements.html
##################################################################################################
def plot_full_court_movement(data, name):
    plt.scatter(data.x_loc, -data.y_loc, c=data.game_clock,
                cmap=plt.cm.Blues, s=250, zorder=1)

    cbar = plt.colorbar(orientation="horizontal")
    cbar.ax.invert_xaxis()
    draw_full_court()

    plt.xlim(0, 94)
    plt.ylim(-50, 0)
    plt.savefig('./data/img/full/'+name+'.jpg')

    plt.figure(figsize=(15, 11.5))
    plt.close()

def plot_half_court_movement(data):
    plt.figure(figsize=(12,11))
    plt.scatter(data.x_loc, data.y_loc, c=data.game_clock,
                cmap=plt.cm.Blues, s=250, zorder=1)

    draw_half_court()
    # Adjust plot limits to just fit in half court
    plt.xlim(-250,250)
    # Descending values along th y axis from bottom to top
    # in order to place the hoop by the top of plot
    plt.ylim(422.5, -47.5)
    # get rid of axis tick labels
    # plt.tick_params(labelbottom=False, labelleft=False)
    plt.savefig('./data/img/half/event_movement_convert_half.jpg')
    plt.close()


########################################################################
# Convert all full court coordinates in data to half court coordinates
########################################################################
def half_full_to_half(data):

    # convert to half court scale
    # note the x_loc and the y_loc are switched in shot charts from movement data (charts are perpendicular)
    data['x_loc_copy'] = data['x_loc']
    data['y_loc_copy'] = data['y_loc']

    # Range conversion formula
    # http://math.stackexchange.com/questions/43698/range-scaling-problem

    data['x_loc'] = data['y_loc_copy'].apply(lambda y: 250 * (1 - (y - 0)/(50 - 0)) + -250 * ((y - 0)/(50 - 0)))
    data['y_loc'] = data['x_loc_copy'].apply(lambda x: -47.5 * (1 - (x - 0)/(47 - 0)) + 422.5 * ((x - 0)/(47 - 0)))
    data = data.drop('x_loc_copy', axis=1, inplace=False)
    data = data.drop('y_loc_copy', axis=1, inplace=False)

    return data

##############################################################################################################
# Convert all full court coordinates that occur in other half of court to occur in one half of a full court
##############################################################################################################
def full_to_half_full(data):

    # first force all points above 47 to their half court counterparts
    data.loc[data.x_loc > 47,'y_loc'] = data.loc[data.x_loc > 47, 'y_loc'].apply(lambda y: 50 - y)
    data.loc[data.x_loc > 47,'x_loc'] = data.loc[data.x_loc > 47, 'x_loc'].apply(lambda x: 94 - x)

    return data

################################################################################
# Draw Full Court: http://savvastjortjoglou.com/nba-play-by-play-movements.html
################################################################################
def draw_full_court(ax=None, color="gray", lw=1, zorder=0):

    if ax is None:
        ax = plt.gca()

    # Creates the out of bounds lines around the court
    outer = Rectangle((0,-50), width=94, height=50, color=color,
                      zorder=zorder, fill=False, lw=lw)

    # The left and right basketball hoops
    l_hoop = Circle((5.35,-25), radius=.75, lw=lw, fill=False,
                    color=color, zorder=zorder)
    r_hoop = Circle((88.65,-25), radius=.75, lw=lw, fill=False,
                    color=color, zorder=zorder)

    # Left and right backboards
    l_backboard = Rectangle((4,-28), 0, 6, lw=lw, color=color,
                            zorder=zorder)
    r_backboard = Rectangle((90, -28), 0, 6, lw=lw,color=color,
                            zorder=zorder)

    # Left and right paint areas
    l_outer_box = Rectangle((0, -33), 19, 16, lw=lw, fill=False,
                            color=color, zorder=zorder)
    l_inner_box = Rectangle((0, -31), 19, 12, lw=lw, fill=False,
                            color=color, zorder=zorder)
    r_outer_box = Rectangle((75, -33), 19, 16, lw=lw, fill=False,
                            color=color, zorder=zorder)

    r_inner_box = Rectangle((75, -31), 19, 12, lw=lw, fill=False,
                            color=color, zorder=zorder)

    # Left and right free throw circles
    l_free_throw = Circle((19,-25), radius=6, lw=lw, fill=False,
                          color=color, zorder=zorder)
    r_free_throw = Circle((75, -25), radius=6, lw=lw, fill=False,
                          color=color, zorder=zorder)

    # Left and right corner 3-PT lines
    # a represents the top lines
    # b represents the bottom lines
    l_corner_a = Rectangle((0,-3), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    l_corner_b = Rectangle((0,-47), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    r_corner_a = Rectangle((80, -3), 14, 0, lw=lw, color=color,
                           zorder=zorder)
    r_corner_b = Rectangle((80, -47), 14, 0, lw=lw, color=color,
                           zorder=zorder)

    # Left and right 3-PT line arcs
    l_arc = Arc((5,-25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
                color=color, zorder=zorder)
    r_arc = Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248, lw=lw,
                color=color, zorder=zorder)

    # half_court
    # ax.axvline(470)
    half_court = Rectangle((47,-50), 0, 50, lw=lw, color=color,
                           zorder=zorder)

    hc_big_circle = Circle((47, -25), radius=6, lw=lw, fill=False,
                           color=color, zorder=zorder)
    hc_sm_circle = Circle((47, -25), radius=2, lw=lw, fill=False,
                          color=color, zorder=zorder)

    court_elements = [l_hoop, l_backboard, l_outer_box, outer,
                      l_inner_box, l_free_throw, l_corner_a,
                      l_corner_b, l_arc, r_hoop, r_backboard,
                      r_outer_box, r_inner_box, r_free_throw,
                      r_corner_a, r_corner_b, r_arc, half_court,
                      hc_big_circle, hc_sm_circle]

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

###########################################################################
# Visualization of court: http://savvastjortjoglou.com/nba-shot-sharts.html
###########################################################################
def draw_half_court(ax=None, color='black', lw=2, outer_lines=False):
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

def convert(file_name):

    data = pd.read_csv('./data/no_label/'+filename)
    data = full_to_half_full(data)
    data = half_full_to_half(data)
    # save data as converted
    data.to_csv('./data/converted/0021500139.csv',index=False)

# DEMO of plotting functions to demonstrate the conversion process
data = pd.read_csv('./data/no_label/0021500139.csv')
data = data[(data['event_id'] == 2) & (data['player_id'] == 101162)]

data = full_to_half_full(data)
plot_full_court_movement(data,'event_movement_convert')
data = half_full_to_half(data)
plot_half_court_movement(data)

# save data as converted
data.to_csv('./data/converted/0021500139.csv',index=False)
