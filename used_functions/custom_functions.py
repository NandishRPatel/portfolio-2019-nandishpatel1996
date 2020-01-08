import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import re
import matplotlib.dates as mdates
from tabulate import tabulate
import seaborn as sns
import matplotlib.patches as mpatch

# For slideshow purposes
GLOBAL_INDEX = 1

# Setting matplotlib parameters

## Change title size of a plot
mpl.rcParams['axes.titlesize'] = 22
## Change label size(x and y) of a plot
mpl.rcParams['axes.labelsize'] = 16
## Change xticks size of a plot
mpl.rcParams['xtick.labelsize'] = 15
## Change yticks size of a plot
mpl.rcParams['ytick.labelsize'] = 15

def clean_elevation_gain(elevation_gain):
    
    '''
    Elevation gain contained unit which was in metre. It removes that unit
    and converts it into float.
    
    Input  : elevation_gain -> list
    output : list
    '''
    
    return [ float(re.sub('\sm', '', i)) for i in elevation_gain]

def crete_correlogram(df, size = (10,10), corr_font_size = 18, xlab_rotation = 0, ylab_rotation = 0):
    
    '''
    Creates a correlogram of a dataframe.
    Input 
          : df ->  type : pandas dataframe
          : size -> size of a plot(x-axis, y-axis) -> type : tuple
          : corr_font_size -> size of text(correlation) in each box of correlogram -> type : int
          : xlab_rotation, ylab_rotation -> rotation of xlabel and ylabel -> type : int
          
    output : Returns nothing
    '''
    
    # Creating an empty figure
    fig, ax = plt.subplots(figsize = size)
    # Getting correlation values of df
    cor = df.corr().values
    # Creating correlation matrix
    im = ax.imshow(cor)                                                  
    
    # Getting columns
    df_columns = df.columns.values
    # Length of the columns (no of columns)
    len_df_columns = len(df_columns)
    
    # Stop showing grid in plots
    ax.grid(False)
    # Positions to put xticks
    ax.set_xticks(np.arange(len_df_columns))
    # Positions to put yticks
    ax.set_yticks(np.arange(len_df_columns))             
    # Setting y-axis limits
    ax.set_ylim(-0.5, len(df_columns) - 0.5)
    
    # Setting labels on x-axis
    ax.set_xticklabels(df_columns)
    # Setting labels on y-axis
    ax.set_yticklabels(df_columns)
    
    # Rotating xtick labels
    plt.setp(ax.get_xticklabels(), rotation = xlab_rotation, ha = "right",
         rotation_mode = "anchor")
    # Rotating ytick labels
    plt.setp(ax.get_yticklabels(), rotation = ylab_rotation, ha = "right",
         rotation_mode = "anchor")
    
    # Setting correlation value into each box.
    for i in range(len_df_columns):
        for j in range(len_df_columns):
            ax.text(j, i, round(cor[i,j], ndigits = 2),ha = "center", va = "center", 
                    color = "w", fontsize = corr_font_size)
    
    # Plotting a colorbar on the side of a correlogram
    plt.colorbar(im)
    plt.show()

def rand_jitter(somelist, jitter = 0.03):
    
    '''
    Function that will jitters the values of provided list.
    It will help us avoid the problem of points overlapping in scatter plot.
    Input 
           : somelist -> type : list 
           : jitter -> degree to which values are jittered -> type : float
           
    output : Returns list containing jittered values.
    '''
    
    # stdev - how much scattered
    stdev = jitter * (max(somelist) - min(somelist))
    # Adding jittered value to somelist
    return somelist + np.random.randn(len(somelist)) * stdev

def plots_with_workout(df, kind, size = (20,20)):
    
    '''
    Function that will create plot where the plots are differentiated by workout_type
    by using different colors.
    Input  : kind -> type of a plot [for now it can be scatter or box] -> type : string
    Output : doesn't return anything [just plot the graph]
    '''
    
    # No of rows and columns for subplots
    rows = 4
    columns = 4
    
    # Creating and empty figure with 4x4 = 16 subplots
    fig, ax = plt.subplots(rows, columns, figsize = (20,20))
    
    # Ravel returns flattened array of axis of subplots
    ax = ax.ravel()
    
    for i,el in enumerate(df.columns.values[:-2]):
        # Get unique values
        labels, color_list = np.unique(df["workout_type"], return_inverse = True)
        if kind == "scatter":
            # Plotting a scatter plot
            plot = ax[i].scatter(rand_jitter(df["workout_type_int"], jitter = 0.04), 
                                 df.loc[:,el], c = color_list, marker = 'o', alpha = 0.9)
            # Legends for plots
            ax[i].legend(plot.legend_elements()[0], labels)
            # Setting xticks
            ax[i].set_xticks([1,2,3])
            # Setting labels on x-axis
            ax[i].set_xticklabels(["Ride","Workout","Race"], fontsize = 12)
            # Set title for each subplot
            ax[i].set_title(el, size = 18)
        
        elif kind == "box":
            # Plotting a boxplot
            plot = df.boxplot(el, by = "workout_type", ax = ax.flatten()[i], return_type = "axes")
            # Setting super title to " "
            fig.suptitle(' ')                                                  
    
    # Removing unnecessary subplots
    for i in range(len(df.columns.values) - 2, rows * columns):
        fig.delaxes(ax[i])
    
    # Spacing b/w subplots
    plt.tight_layout()                                                         
    
    plt.show()


def activityOverGivenMonth(df, year, month):
    
    '''
    Function that will plot total distance ridden by a rider grouped by days.
    Input
           : year -> type : int
           : month -> type : int
    
    output : Returns nothing just plots the graph
    '''
    
    # Creating an empty dataframe
    temp_plot = pd.DataFrame()
    
    # Getting data for particular month and year
    temp_plot = df[np.logical_and(df["date"].dt.month == month , df["date"].dt.year == year)]
    
    # Removing the timezone details to avoid plotting problems
    temp_plot["date"] = temp_plot['date'].dt.tz_localize(None)
    
    # Setting figure size (x-axis, y-axis)
    plt.figure(figsize = (18, 18))
    
    # Plotting a line plot
    plt.plot(temp_plot["date"], temp_plot["distance"], marker = 'o')
    
    # Setting plot annotations
    plt.xlabel("Day - Month - Year", fontsize = 16, fontweight = "bold")
    plt.ylabel("Distance (km)", fontsize = 16, fontweight = "bold")
    plt.title("Distance travelled by Days")
    
    # Setting x-axis major locator to day - days on x-axis
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # Setting x-axis major formator to day - days on x-axis in dd-mm-yy format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # Setting ticks on x-axis to given date range
    plt.gca().set_xticks(pd.date_range(start = temp_plot["date"].iloc[0], end = temp_plot["date"].iloc[-1], freq = "d"))
    # Rotating the ticks on the x-axis
    plt.setp(plt.gca().get_xticklabels(), rotation = 90)
    
    # Plotting distance on a line plot dots 
    for i in range(len(temp_plot.date.values)):
        plt.text(temp_plot["date"].values[i], temp_plot["distance"].values[i], str(temp_plot["distance"].values[i]),
                 fontsize = 14, bbox = dict(facecolor = 'white', alpha = 0.6))
    
    plt.show()



def create_histogram_plus_boxplot(series, var_name, color, ylabel, xlabel, size):
    
    '''
    Function that creates combination of histogram and boxplot for pandas dataseries.
    Making it easier to observe distribution of a particular varibale
    
    Input
           : series -> type : pandas series or list
           : var_name -> name of the variable (to display in the title of a plot) -> type : string
           : color -> color for both histogram and boxplot(border color) -> type : string
           : xlabel -> name of the x axis -> type : string
           : ylabel -> name of the y axis -> type : string
    
    Output : Returns nothing
    '''
    
    # Create a figure of width 10cm and height 11cm
    plt.figure(figsize = size)
    
    # plt.subplot(row, column, index)
    plt.subplot(2, 1, 1)
    # Plotting a histogram
    plt.hist(series, edgecolor = "black", color = color)  
    # Setting the title of histogram
    plt.title('Histogram of ' + var_name)
    # xticks = [] because both histogram and boxplot share same x axis (you can use sharex argument in plt.subplots function)
    plt.xticks([])
    
    # Setting label on y-axis
    plt.ylabel(ylabel)
    
    plt.subplot(2, 1, 2)
    # Plotting a boxplot
    plt.boxplot(series, boxprops = dict(color = color), vert = False)  
    # Setting labels on x-axis
    plt.xlabel(xlabel)
    # Setting the title of boxplot
    plt.title('Boxplot of ' + var_name)
    # No need of labels on y-axis(one variable)
    plt.yticks([])
    
    plt.show()


def string_to_int(string):
    
    '''
    Function that converts string to int.
    
    Input  : string -> type : string
    Output : Returns int
    '''
    
    if string[0] == "0":
        return int(string[1])
    return int(string)
    
def create_weekdata(df, week):
    
    '''
    Function that creates a dataframe containing details about appliances based on week number.
    
    Input  
           : df -> type : dataframe
           : week -> type : int
    
    Output : Returns dataframe of particular week
    '''
    
    # Series of weeks
    data_weeks = pd.Series(map(string_to_int, df["date"].dt.strftime('%W')))
    # Choice of Week
    choice_week = min(data_weeks) + week - 1
    # Creating a boolean flag
    flag_week = (data_weeks == choice_week)
    # Filtering data based on a boolean flag
    week_data = df.loc[flag_week]
    # Grouping data based on hour and day, getting appliances 
    week_data_groupby = week_data.groupby([week_data["date"].dt.hour,
                                              week_data["date"].dt.weekday_name])['Appliances'].sum()
    
    # Creating a df from groupby object using unstack function
    week_data_df = week_data_groupby.unstack()
    # Sorting dataframe columns
    week_data_df = week_data_df[["Monday","Tuesday","Wednesday","Thursday","Friday", "Saturday","Sunday"]]
    
    return week_data_df


def create_scatter_matrix(df, size = (18,18), xlab_rotation = 0, ylab_rotation = 0, xticklab_fontsize = 10, yticklab_fontsize = 10):
    
    '''
    Function that creates custom scatter matrix.
    Input
           : df -> type : Pandas DataFrame
           : size -> size of a plot (x, y) -> type : Tuple
           : xlab_rotation, ylab_rotation -> label rotation -> type : int
           : xticklab_fontsize, yticklab_fontsize -> fontsize of tick labels -> int
    
    Output : Returns Nothing
    '''
    
    # Getting names of columns of dataframe df
    names = df.columns.values
    # Calculating correlation between all the variables in df
    correlations = df.corr().values
    # Taking out number of columns
    d = df.shape[1]
    
    # Creating subplots d x d
    fig, axes = plt.subplots(nrows = d, ncols = d, figsize = size)
    # Spacing between subplots
    fig.subplots_adjust(wspace = 0.1, hspace = 0.1)
    
    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            
            # Stop showing xaxis and yaxis grid on subplots
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            
            # Rotating the xaxis and yaxis labels 
            ax.xaxis.label.set_rotation(xlab_rotation)
            ax.yaxis.label.set_rotation(ylab_rotation)
            
            # Padding between plots and labels on y-axis
            ax.yaxis.labelpad = 30                                             
            
            # Setting labels on subplots
            ax.set_xlabel(names[j])
            ax.set_ylabel(names[i])
            
            # Plotting histogram on the diagonal
            if i == j:
                ax.hist(df.iloc[:,i], color = "blue", edgecolor='black')
            
            # Plotting text(correlation) above the diagonal
            elif i < j:
                ax.text(0.5, 0.5, round(correlations[i][j],2), fontsize = 20, 
                        horizontalalignment='center', verticalalignment='center')
                # Setting background color to white
                ax.set_facecolor('xkcd:white')
            
            # Plotting scatter plot elsewhere
            else:
                ax.scatter(df.iloc[:,j], df.iloc[:,i], s = 7)
            
            # Hiding the axis as required
            if j != 0 and i < d - 1:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if i == d - 1 and j > 0:
                ax.get_yaxis().set_visible(False)
            if j == 0 and i != d - 1:
                ax.get_xaxis().set_visible(False)
    
    # bold tick labels
    plt.setp(ax.get_xticklabels(), fontsize = xticklab_fontsize, fontweight = "bold")
    plt.setp(ax.get_yticklabels(), fontsize = yticklab_fontsize, fontweight = "bold")
    
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred):
    
    '''
    Function that calculates Mean Absolute Percentage Error(MAPE).
    Input
           : y_true -> true values of data -> Pandas Series
           : y_pred -> values obtained from model -> Pandas Series
    
    Output : Returns MAPE -> type : float
    '''
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_RMSE_VS_NoVar(df):
    
    '''
    Function that plots line plot for RMSE vs Number of variable.
    Input  : df -> type : Pandas DataFrame
    Output : Returns Nothing
    '''
    
    # Plotting the figure
    plt.figure(figsize = (12,12))

    # Plotting line plot and scatter plot for models
    sns.scatterplot(df["no_variables_chosen"], df["rmse"], marker = "o", s = 100)
    sns.lineplot(df["no_variables_chosen"], df["rmse"])

    # Setting x-axis, y-axis label and title of the plot
    plt.xlabel("#Variables taken in Linear Model")
    plt.ylabel("RMSE(Root Mean Squared Error)")
    plt.title("Model Evaluation")

    plt.show()

    
# The idea for the Code(Below two functions) was from https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
# I have made some changes to it.

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def create_lineplot_multiple_YAXIS(df, var1, var2, var3, var4, lab1, lab2, lab3, lab4, size = (18,18), 
                                   c1 = "b", c2 = "r", c3 = "g", m1 = "o", m2 = "s", m3 = "^"):
    
    '''
    Function that allows to have three y-axis.
    Input
           : df -> type : Pandas DataFrame
           : var1...var4 -> variable(column) to be fetched from dataframe -> type : string
           : lab1...lab4 -> labels to show on plot -> type : string
           : size -> size of a plot (x, y) -> type : Tuple
           : c1, c2, and c3 are colors for different variables -> type : character (r -> red, b -> blue, g -> green)
           : m1, m2, and m3 are markers for different variable -> type : character (o -> solid circle, s -> solid square, ^ -> solid     triangle)
           
    Output : Returns Nothing
    '''
    
    # Creating subplots
    fig, host = plt.subplots(figsize = size)
    # Spacing between the plots
    fig.subplots_adjust(right = 0.75)
    
    # Create a twin Axes sharing the xaxis
    par1 = host.twinx()
    par2 = host.twinx()
    
    # Offset the right spine of par2. The ticks and label have already been placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))
    
    # Having been created by twinx, par2 has its frame off, so the line of its detached spine is invisible.  First, activate the frame but     # make the patch and spines invisible.
    make_patch_spines_invisible(par2)
    
    # Showing the right spine
    par2.spines["right"].set_visible(True)

    # Plotting lines with the markers
    p1, = host.plot(df[var1], df[var2], c = c1, label = var2, marker = "o")
    p2, = par1.plot(df[var1], df[var3], c = c2, label = var3, marker = "s")
    p3, = par2.plot(df[var1], df[var4], c = c3, label = var4, marker = "^")
    
    # Setting xaxis major locator to month locator.
    host.xaxis.set_major_locator(mdates.MonthLocator())
    # Setting xaxis major formatter to year - month
    host.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Setting labels
    host.set_xlabel(var1 + " (" + lab1 + ")", fontsize = 18)
    host.set_ylabel(var2 + " (" + lab2 + ")", fontsize = 16)
    # Changing grid colors for better visibility
    host.yaxis.grid(color = c1, linewidth = 0.3)
    # Rotation the xaxis labels
    plt.setp(host.get_xticklabels(), rotation = 90)

    # Setting labels and grid color
    par1.set_ylabel(var3 + " (" + lab3 + ")", fontsize = 16)
    par1.yaxis.grid(color = c2, linewidth = 0.3)

    # Setting labels and grid color
    par2.set_ylabel(var4 + " (" + lab4 + ")", fontsize = 16)
    par2.yaxis.grid(color = c3, linewidth = 0.3)

    # Setting the different label colors for different variables
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    # Setting the different tick colors for different variables
    host.tick_params(axis = 'y', colors = p1.get_color())
    par1.tick_params(axis = 'y', colors = p2.get_color())
    par2.tick_params(axis = 'y', colors = p3.get_color())

    # Creating legend for better understanding
    lines = [p1, p2, p3]
    host.legend(lines, [l.get_label() for l in lines], fontsize = 12)
    
    # Setting title of the plot
    plt.title("Distance,TSS and Average Speed grouped by Months", fontsize = 18)
    plt.show()


def put_coordinates(df, c):
    
    '''
    Function that plots coordinates of the centroid on the scatter plot.
    Input
           : df -> type : DataFrame
           : c -> colors for centroids -> type : list
    
    Output : Returns Nothing
    '''
    
    # Font Properties
    font = {
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }
    
    # Plotting coordinates on the plot
    for index, centroid in df.iterrows():
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        plt.text(centroid_x, centroid_y, "(" +  "%.5f" % centroid_x + ", " + "%.5f" % centroid_y + ")",
            fontdict = font, bbox = dict(facecolor = c[index], alpha = 0.5))


def kmean_iteration(data, current_centroids_df, previous_centroids_df):
    
    '''
    Function that performs an iteration of the Kmeans algorithm.
    Input
           : data -> type : DataFrame
           : current_centroids_df -> type : DataFrame
           : previous_centroids_df -> type : DataFrame
    
    Output 
           : current_centroids_df -> type : DataFrame
           : previous_centroids_df -> type : DataFrame
    '''
    
    # Global index for saving plots
    global GLOBAL_INDEX
    
    # Setting variable for plot title and savefig
    variable = str(GLOBAL_INDEX % 6 if GLOBAL_INDEX % 6 != 0 else 6 if GLOBAL_INDEX == 6 else int(GLOBAL_INDEX / 6))
    
    # Colors for cluster centroid and data points
    colors_dict = {0 : "green", 1 : "blue", 2 : "yellow", 3 : "cyan"}
    colors = list(colors_dict.values())
    
    # Calculating Euclidean distance and creating new column in cluster dataframe
    cluster_count = 0
    clusters = pd.DataFrame()
    for index, centroid in current_centroids_df.iterrows():
        clusters[cluster_count] = (data[data.columns] - centroid).pow(2).sum(1).pow(0.5)
        cluster_count += 1

    # Setting the cluster which has minimum distance among the four clusters
    data['cluster'] = clusters.idxmin(axis=1)
    
    # Setting colors
    colors_list = [colors_dict[i] for i in data["cluster"]]
    
    # Setting current clusters centroids to previous cluster
    previous_centroids_df = current_centroids_df

    # Setting current cluster centroids to empty dataframe
    current_centroids_df = pd.DataFrame()

    # Creating new centroids by grouping the data by cluster and then aggregating them based on mean 
    new_centroids = data.groupby('cluster').agg(np.mean)

    # Setting current cluster centroids to new centroids after adjustments
    current_centroids_df = new_centroids

    # Printing centroids
    print("Previous centroids")
    print(tabulate(previous_centroids_df, headers = 'keys', tablefmt = 'psql'))
    print("Current centroids")
    print(tabulate(current_centroids_df, headers = 'keys', tablefmt = 'psql'))

    # Plotting an empty figure
    plt.figure(figsize = (10,7))
    
    # Plotting data points
    plt.scatter(data.iloc[:,0], data.iloc[:,1], c = colors_list, alpha = 0.5, edgecolor = "black")
    
    # Plotting centroids
    plt.scatter(current_centroids_df.loc[:,0], current_centroids_df.loc[:,1], c = colors, 
                edgecolors = "red", s = 500, linewidth = 3)
    
    # Plotting the legend
    patches = [ plt.plot([],[], marker = "o", ms = 10, ls = "", mec = None, color = colors[i], 
            label = "Cluster " + str(i))[0]  for i in range(len(colors)) ]
    
    plt.legend(handles = patches, loc = 'upper right', ncol = 1)

    
    # Put coordinates of centroid on plot
    put_coordinates(df = current_centroids_df, c = colors)
    
    # Plot title, xlabel and ylabel
    plt.title("Scatter plot of the data with centroids after #" + variable + " iteration")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Saving plot for a slideshow - Commenting it as i have already saved it.
    # plt.savefig('./html/images/' + variable + '.png', bbox_inches = 'tight')
    
    # Increamenting the variable for plot image name
    GLOBAL_INDEX += 1
    
    plt.show()
    
    return current_centroids_df, previous_centroids_df