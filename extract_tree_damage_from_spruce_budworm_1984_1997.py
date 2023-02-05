#coding: utf-8

"""
Summary
-------
Script to produce predictions of SBW outbreak extent using a random forest model
trained on 2021 data.

This is a script that uses various libraries such as geopandas, shapely, descartes, scikit-learn,
matplotlib, pyproj, fiona, statsmodels, seaborn, pygam, alphashape and osgeo. The code processes raster data, performs
data manipulations, trains machine learning models and plots the results. The code has two defined functions,
concave_hull and duplicated. The former function computes the concave hull of a set of points using the alphashape library
and the latter is a function that loads raster data and performs a variety of data manipulations and statistical
computations, including difference calculation, data flattening, and machine learning model training and evaluation.

The script was produced as part of a long term project at the MNDMNRF and the University of Toronto to improve tree
damage mapping due to gaps in the annual aerial survey maps (caused by wide flight lines, smoke from wildfires, and
weather conditions such as fog). 


"""

import geopandas as gpd
import pandas as pd 
from geopandas.tools import sjoin
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
from descartes import PolygonPatch
import time
import math
import scipy.stats as stats
import numpy as np
import os, sys
from pyproj import CRS, Transformer
import fiona
import statsmodels.api as sm
import seaborn as sns

from sklearn.model_selection import train_test_split
import sklearn
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mpl
from osgeo import ogr, gdal,osr
from math import floor

from shapely.ops import unary_union

import warnings
warnings.filterwarnings('ignore')

import alphashape

# Define the concave_hull function
def concave_hull(points, a, compar):

    alpha = a 
        
    # Calculate the hull using alphashape
    hull = alphashape.alphashape(points, a) 

    # Try converting the hull into a MultiPolygon using one method, otherwise try another
    try: 
        if a != 0:  
            hull = MultiPolygon(hull) 
        else:
            hull = MultiPolygon([hull])
    except TypeError:
        hull = MultiPolygon([hull])

    # Check if the hull exists
    if len(hull) != 0:
        # Convert the hull into a GeoDataFrame
        p1 = gpd.GeoDataFrame(index=[0], crs='ESRI:102001', geometry=[hull])
        hull_explode = p1.explode(ignore_index=True)
        
        # Store the exploded polygons into a GeoDataFrame
        polygon = gpd.GeoDataFrame(crs='ESRI:102001', geometry=list(hull_explode['geometry']))

        # Calculate the area of each polygon and store it as a new column in the GeoDataFrame
        polygon['AREA'] = (polygon['geometry'].area) * 0.000001

        # Filter the GeoDataFrame to only include polygons with area greater than or equal to 3
        polygon1 = polygon[polygon['AREA'] >= 3]

        # If no such polygon exists, filter the GeoDataFrame to only include polygons
        # with area greater than or equal to 0
        if len(polygon1) == 0:
            polygon = polygon[polygon['AREA'] >= 0]
        else:
            polygon = polygon1


        return polygon

    # Return an empty list if no hull was generated
    else:
        print('no geometry')
        return []

def duplicated(yeari,mod):

    year = str(yeari)
    yearp = str(1998)


    # Create list of file names with year and yearp as placeholders
    files = ['100m_on_ndmi_102001_'+year,'asm_'+year,'buff8_'+year,\
             '100m_on_ndmi_102001_'+yearp,'100m_on_nbr1_102001_'+year,\
             '100m_on_nbr1_102001_'+yearp,'100m_on_b4b5_102001_'+year,'100m_on_b4b5_102001_'+yearp]
    # Create list of names with year and yearp as placeholders
    names = ['ndmi_'+year,'dam','small_buff','ndmi_'+yearp,\
             'nbr1_'+year,'nbr1_'+yearp,'b4b5_'+year,'b4b5_'+yearp] 
    # Create dictionary "pred"
    pred = {}
    # Create list "transformers"
    transformers = []
    # Create list "cols_list"
    cols_list = []
    # Create list "rows_list"
    rows_list = [] 

    # Iterate through files and names
    for fi,n in zip(files,names): 
        # Print the file name
        print(fi)
        # Set the file name of the raster file
        file_name_raster = fi
        # Open the raster file using gdal
        src_ds = gdal.Open('rasters/new_res/final/'+file_name_raster+'.tif')
        # Get the first raster band
        rb1=src_ds.GetRasterBand(1)
        # Get the number of columns in the raster
        cols = src_ds.RasterXSize
        # Append the number of columns to cols_list
        cols_list.append(cols)
        # Get the number of rows in the raster
        rows = src_ds.RasterYSize
        # Append the number of rows to rows_list
        rows_list.append(rows) 
        # Read the data in the raster as an array
        data = rb1.ReadAsArray(0, 0, cols, rows)
        # Print a success message
        print('Success in reading file.........................................') 
        # Add the data as a flattened array to the "pred" dictionary with key n
        pred[n] = data.flatten()
        # Print the length of the flattened data array
        print(len(data.flatten()))
        # Get the geo-transform of the raster
        transform=src_ds.GetGeoTransform()
        # Append the geo-transform to the "transformers" list
        transformers.append(transform)
    

    # Add a keys to the "pred" dictionary with the difference between the
    # spectral indices for the two years (healthy & affected) 

    pred['diff'] = pred['ndmi_'+year] - pred['ndmi_'+yearp]
    pred['nbr1_diff'] = pred['nbr1_'+year] - pred['nbr1_'+yearp]
    pred['b4b5_diff'] = pred['b4b5_'+year] - pred['b4b5_'+yearp]

    # Calculate the number of values that should be in the latitude and longitude arrays 
    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]
    lrx = ulx + (col_num * xres)
    lry = uly + (row_num * yres)

    # Creating two arrays Xi and Yi that contain the linspace values between the
    # minimum and maximum of ulx, lrx, uly, lry.

    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    

    Xi, Yi = np.meshgrid(Xi, Yi)

    # Flattening Xi and Yi and reshaping them in a way that results in Xi and Yi
    # with row_num and col_num length.
    Xi, Yi = Xi.flatten(), Yi.flatten()

    # Reverse the arrays - necessary for plotting correctly and exporting for use in ArcGIS / QGIS.
    # This is due to how images are indexed in Python GDAL. 

    X_reshape = Xi.reshape(row_num,col_num)[::-1]
    Xi = X_reshape.flatten()
    Y_reshape = Yi.reshape(row_num,col_num)[::-1]
    Yi = Y_reshape.flatten()

    # Adding Xi and Yi values as the lon and lat columns to the pred data frame. 
  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred).dropna(how='any') 
    

    df_calc = df

    # NDMI should be between -1 and 1, otherwise value is likely due to
    # shadow or cloud mask error. 

    df = df[df['ndmi_'+year] >= -1]
    df = df[df['ndmi_'+year] <= 1]
    df = df[df['ndmi_'+yearp] >= -1]
    df = df[df['ndmi_'+yearp] <= 1]

    # Taking percentiles of the columns b4b5_yearp, b4b5_year, nbr1_yearp,
    # and nbr1_year and filtering df so that it only contains rows where the values
    # in these columns are between the 1st and 99th percentile.

    # Due this as a secondary checck to make sure cloud and shadow masks work.

    # Important for machine learning because of "garbage in, garbage out" problem. 

    p99 = np.percentile(df_calc['b4b5_'+yearp], 99)
    p1 = np.percentile(df_calc['b4b5_'+yearp], 1)
    df = df[df['b4b5_'+yearp] >= p1]
    df = df[df['b4b5_'+yearp] <= p99]

    p99 = np.percentile(df_calc['b4b5_'+year], 99)
    p1 = np.percentile(df_calc['b4b5_'+year], 1)
    df = df[df['b4b5_'+year] >= p1]
    df = df[df['b4b5_'+year] <= p99]

    p99 = np.percentile(df_calc['nbr1_'+yearp], 99)
    p1 = np.percentile(df_calc['nbr1_'+yearp], 1)
    df = df[df['nbr1_'+yearp] >= p1]
    df = df[df['nbr1_'+yearp] <= p99]

    p99 = np.percentile(df_calc['nbr1_'+year], 99)
    p1 = np.percentile(df_calc['nbr1_'+year], 1)
    df = df[df['nbr1_'+year] >= p1]
    df = df[df['nbr1_'+year] <= p99]

 
    df['dam'] = np.where(df['dam'] >= 1,1,0)

    # Sampling every 10th row from the data frame df and storing the result in df_save.
    # Need to do this to avoid memory error in concave_hull function.
    # Not necessary to predict all points - that level of detail is probably not possible
    # given the older satellite imagery and may introduce more uncertainty. 
    df = df.iloc[::10, :]
    df_save = df
    
    lengths = []

    trainer = []

    # Obtain 4000 training points.
    # 4000 points will be used for cross-validation.
    # 4000 is enough points that the distribution of data matches that of the entire dataset. 
    for cl in [1,0]: 
        df_f = df[df['dam'] == cl].dropna(how='any')
        if cl != 0:

            if len(df_f) >= 2000: 
                num = 2000
                trainer.append(df_f.sample(n=num,random_state=1)) 
                lengths.append(num)
            else:
                trainer.append(df_f)
        else:
            # Negative points cannot be within an 8km buffer of the aerial survey to avoid bias due to
            # gaps in that data due to flight lines, weather conditions, and smoke from wildfires. 
            df_f = df_f[df_f['small_buff'] != 1]
            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1))
            

    df2 = pd.concat(trainer) # Concatenate all the dataframes in the list 'trainer' and assign to df2
    # Reset the index of the resulting dataframe and drop all rows with missing values
    df2 = df2.reset_index(drop=True).dropna(how='any')

    # Select the columns specified as input features for the model and assign to df_trainX
    df_trainX = df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']] 
    X = np.array(df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]) 
    # Select the column 'dam' and convert it to a numpy array with shape (n,1) where n is the
    # number of samples, and assign to df_trainY
    # Convert the 'dam' column to a 1-dimensional numpy array and assign to Y
    df_trainY = np.array(df2[['dam']]).reshape(-1, 1)
    Y = np.array(df2['dam'])
    
    # Split the data into 5 folds, with each fold having 30% of the samples for testing and the rest for training
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1) 

    # Initialize a list 'mattc' to store the results of the evaluation metric 'Matthews Correlation Coefficient'
    mattc = [] 

    for train_index, test_index in sss.split(X, Y): # For each split of the data into training and testing data

        X_train, X_test = X[train_index], X[test_index] # Split X into training and testing data
        y_train, y_test = Y[train_index], Y[test_index] # Split Y into training and testing data

        bestF = CV_rfc.fit(X, Y) # Fit the model 'CV_rfc' to the entire dataset (X and Y)
        
        Ztest = mod.predict(X_test) # Use the trained model to make predictions on the testing data
        print(matthews_corrcoef(y_test, Ztest)) # Print the value of the evaluation metric 'Matthews Correlation Coefficient'
        mattc.append(matthews_corrcoef(y_test, Ztest)) # Append the value of the evaluation metric to the list 'mattc'

    df_save['tracker'] = list(range(0,len(df_save))) # Add a new column 'tracker' with values ranging from 0 to the number of rows in the dataframe 'df_save'
    rem_track = df_save.dropna(how='any') # Drop all rows with missing values in the dataframe 'df_save' and assign the resulting dataframe to 'rem_track'

    # Predict the 'Zi' values using the given inputs to the model

    Zi = mod.predict(np.array(rem_track[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]))

    # Add the predicted 'Zi' values to the 'rem_track' dataframe
    rem_track['pred'] = Zi

    # Create an empty list to store the Matthews correlation coefficients
    matt = []

    # Convert the 'dam' column of 'rem_track' to a numpy array

    rep = np.array(rem_track['dam'])

    Zn = np.array(rem_track['pred'])

    # Print the confusion matrix and Matthews correlation coefficient between the
    # actual ('dam') and predicted ('pred') values
        
    print(confusion_matrix(rep, Zn))
    print(matthews_corrcoef(rep, Zn))

    # Append the Matthews correlation coefficient to the 'matt' list
    matt.append(matthews_corrcoef(rep, Zn))

    # Add a 'pred' column with default value -9999 to the 'add_track' dataframe

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    # Concatenate the 'rem_track' and 'add_track' dataframes
    total = pd.concat([rem_track,add_track])
    total = rem_track

    # Read the 'na_map' data from the specified shapefile

    na_map = gpd.read_file('rasters/temp/'+year+'_proj_clip_dam.shp')
    na_map = na_map[na_map['DAM'] == 1]

    # Filter the predicted values to only those that have positive values (experiencing damage)

    mort = rem_track[rem_track['pred'] == 1]

    # Create a list of points from the 'lon' and 'lat' columns of 'mort'
    
    points = [(x,y,) for x, y in zip(mort['lon'],mort['lat'])]
    na_map3 = na_map[na_map['DAM'] == 1]

    # Create a concave hull from the given points 
    
    ch2 = concave_hull(points,0.0036,na_map3)

    #Create a figure showing the concave hull results and how they compare to the aerial survey maps 
    fig, ax = plt.subplots(1,2)
    if len(ch2) > 0: 
        ch2.plot(ax=ax[1],facecolor='None',edgecolor='k')
    na_map3.plot(ax=ax[0],facecolor='red',edgecolor='None',alpha=0.5)
    na_map3.plot(ax=ax[1],facecolor='red',edgecolor='None',alpha=0.5)
    plt.show()
    if len(ch2) > 0:
        # Write the resulting geodataframe to file in shapefile format
        ch2.to_file('rasters/concave_hull/'+str(yeari)+'_test_y2013_feb3.shp', driver='ESRI Shapefile')
        
if __name__ == "__main__":

    # Code to create, fit, and cross-validate Random Forest model using the 2021 aerial survey map
    # and the median values of all satellite images between June 1 and October 1 for 2013,
    # the "healthy" year, with little to no damage recorded. 

    year = str(2021)
    yearp = str(2013)


    files = ['100m_on_ndmi_102001_'+year,'asm_'+year,'combo','buff8_'+year,'age','elev','soil_text',\
             '100m_on_ndmi_102001_'+yearp,'100m_on_nbr1_102001_'+year,\
             '100m_on_nbr1_102001_'+yearp,'100m_on_b4b5_102001_'+year,'100m_on_b4b5_102001_'+yearp,]
    names = ['ndmi_'+year,'dam','combo','small_buff','age','elev', 'soil_text','ndmi_'+yearp,\
             'nbr1_'+year,'nbr1_'+yearp,'b4b5_'+year,'b4b5_'+yearp] 
    pred = {}
    transformers = []
    cols_list = []
    rows_list = [] 

    for fi,n in zip(files,names): 
        print(fi)
        file_name_raster = fi
        src_ds = gdal.Open('rasters/new_res/final/'+file_name_raster+'.tif')
        rb1=src_ds.GetRasterBand(1)
        cols = src_ds.RasterXSize
        cols_list.append(cols)
        rows = src_ds.RasterYSize
        rows_list.append(rows) 
        data = rb1.ReadAsArray(0, 0, cols, rows)
        print('Success in reading file.........................................') 
        pred[n] = data.flatten()
        print(len(data.flatten()))
        transform=src_ds.GetGeoTransform()
        transformers.append(transform)
    
    pred['age'] = pred['age'] + (int(year)-2011)

    pred['diff'] = pred['ndmi_'+yearp] - pred['ndmi_'+year]
    pred['nbr1_diff'] = pred['nbr1_'+yearp] - pred['nbr1_'+year]
    pred['b4b5_diff'] = pred['b4b5_'+yearp] - pred['b4b5_'+year]


    col_num = cols_list[0]
    row_num = rows_list[0]
    ulx, xres, xskew, uly, yskew, yres  = transformers[0]
    lrx = ulx + (col_num * xres)
    lry = uly + (row_num * yres)


    Yi = np.linspace(np.min([uly,lry]), np.max([uly,lry]), row_num)
    Xi = np.linspace(np.min([ulx,lrx]), np.max([ulx,lrx]), col_num)
    

    Xi, Yi = np.meshgrid(Xi, Yi)
    Xi, Yi = Xi.flatten(), Yi.flatten()

    X_reshape = Xi.reshape(row_num,col_num)[::-1]
    Xi = X_reshape.flatten()
    Y_reshape = Yi.reshape(row_num,col_num)[::-1]
    Yi = Y_reshape.flatten()

  
    pred['lon'] = Xi
    pred['lat'] = Yi

    
    df = pd.DataFrame(pred).dropna(how='any') 

    df_calc = df 

    df = df[df['ndmi_'+year] >= -1]
    df = df[df['ndmi_'+year] <= 1]
    df = df[df['ndmi_'+yearp] >= -1]
    df = df[df['ndmi_'+yearp] <= 1]

    p99 = np.percentile(df_calc['b4b5_'+yearp], 99)
    p1 = np.percentile(df_calc['b4b5_'+yearp], 1)
    df = df[df['b4b5_'+yearp] >= p1]
    df = df[df['b4b5_'+yearp] <= p99]

    p99 = np.percentile(df_calc['b4b5_'+year], 99)
    p1 = np.percentile(df_calc['b4b5_'+year], 1)
    df = df[df['b4b5_'+year] >= p1]
    df = df[df['b4b5_'+year] <= p99]

    p99 = np.percentile(df_calc['nbr1_'+yearp], 99)
    p1 = np.percentile(df_calc['nbr1_'+yearp], 1)
    df = df[df['nbr1_'+yearp] >= p1]
    df = df[df['nbr1_'+yearp] <= p99]

    p99 = np.percentile(df_calc['nbr1_'+year], 99)
    p1 = np.percentile(df_calc['nbr1_'+year], 1)
    df = df[df['nbr1_'+year] >= p1]
    df = df[df['nbr1_'+year] <= p99]

 
    df['dam'] = np.where(df['dam'] >= 2,1,0)
    df_save = df
    
    lengths = []

    trainer = [] 
    for cl in [1,0]: 
        df_f = df[df['dam'] == cl].dropna(how='any')
        if cl != 0:

            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1)) 
            lengths.append(num)
        else:
            df_f = df_f[df_f['small_buff'] != 1]
            num = 2000
            trainer.append(df_f.sample(n=num,random_state=1))
            

    df2 = pd.concat(trainer)
    df2 = df2.reset_index(drop=True).dropna(how='any')

    df_trainX = df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']] 
    X = np.array(df2[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]) 

    df_trainY = np.array(df2[['dam']]).reshape(-1, 1)
    Y = np.array(df2['dam'])

    count = 0 

    rfc = RandomForestClassifier(random_state=1)
    
##    param_grid = { 
##    'max_depth': [5, 10, 30],
##    'max_features': ['sqrt'],
##    'min_samples_leaf': [1,3,5],
##    'min_samples_split': [2,20,40,60]
##    }

    param_grid = { 
    'max_depth': [30],
    'max_features': ['sqrt'],
    'min_samples_leaf': [5],
    'min_samples_split': [20]
    }
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5) #5
    bestF = CV_rfc.fit(X, Y)
    print(CV_rfc.best_params_)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
    
    mattc = []
    from sklearn.metrics import matthews_corrcoef
    for train_index, test_index in sss.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        print(len(X_train))

        bestF = CV_rfc.fit(X, Y)
        
        Ztest = bestF.predict(X_test)
        print(matthews_corrcoef(y_test, Ztest))
        mattc.append(matthews_corrcoef(y_test, Ztest))
        

    df_save['tracker'] = list(range(0,len(df_save))) 
    rem_track = df_save.dropna(how='any')

    Zi = bestF.predict(np.array(rem_track[['ndmi_'+year,'nbr1_'+year,'b4b5_'+year,\
                     'diff','nbr1_diff','b4b5_diff']]))

    rem_track['pred'] = Zi

    matt = []

    rep = np.array(rem_track['dam'])

    Zn = np.array(rem_track['pred'])
        
    print(confusion_matrix(rep, Zn))

    print(matthews_corrcoef(rep, Zn))
    matt.append(matthews_corrcoef(rep, Zn))

    add_track = df_save[pd.isnull(df_save).any(axis=1)]
    add_track['pred'] = -9999

    total = pd.concat([rem_track,add_track])
    total = rem_track


    yfor = list(range(1984,1998))
    for y in yfor: 

        duplicated(y,bestF)

        
