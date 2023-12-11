#Script for ATS780 Machine Learning for Atmospheric Sciences HW3


#%%
from graphviz import Source # To plot trees  "conda install graphviz" & conda install python-graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz
import glob
import xarray as xr
import pandas as pd
import random
import tensorflow as tf

#import for scikit-learn PCA class
from sklearn.decomposition import PCA

#Setting visible GPU devices (only using device 0)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use one GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True) #Only use as much memory as needed instead of maxing at first call of tf
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=40000)])
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# %%

# Specify the local directory where the interpolated GFS data resides
processed_GFS_directory = '/mnt/data1/mking/ATS780/processed_GFS_files/'

# Specify the local directory where the interpolated CLAVRx data resides
clavrx_directory = '/mnt/data1/mking/ATS780/CLAVRX_data/'

# Specify the local direcotry where the persistance interpolated CLAVRx data resides
persistance_directory = '/mnt/data1/mking/ATS780/Persist_CLAVRX_data_/'

#Get the sorted file list in each directory
clavrx_flist_val = sorted(glob.glob(clavrx_directory + 'clavrx_2018*'))
clavrx_flist_trng = sorted(glob.glob(clavrx_directory + 'clavrx_2019*'))
clavrx_flist_test = sorted(glob.glob(clavrx_directory + 'clavrx_2022*'))

GFS_flist_val = sorted(glob.glob(processed_GFS_directory + 'GFS_2018*'))
GFS_flist_trng = sorted(glob.glob(processed_GFS_directory + 'GFS_2019*'))
GFS_flist_test = sorted(glob.glob(processed_GFS_directory + 'GFS_2022*'))

persist_flist_val = sorted(glob.glob(persistance_directory + 'clavrx_2018*'))
persist_flist_trng = sorted(glob.glob(persistance_directory + 'clavrx_2019*'))
persist_flist_test = sorted(glob.glob(persistance_directory + 'clavrx_2022*'))

print(len(clavrx_flist_val), len(clavrx_flist_trng), len(clavrx_flist_test))
print(len(GFS_flist_val), len(GFS_flist_trng), len(GFS_flist_test))
print(len(persist_flist_val), len(persist_flist_trng), len(persist_flist_test))

#%%
#Select a number of random latitude/longitude values to use on each file

#Define the lat/lon values used from CLAVR_x_organizer
res = 0.02
left_lon = -100
right_lon = -65
top_lat = 50
bottom_lat = 25

#One dimensional arrays defining longitude and latitude
len_lon = np.round(np.arange(left_lon,right_lon, res),2)
len_lat = np.round(np.arange(bottom_lat, top_lat, res),2)

#Us numpy meshgrid function to create 2d coordinates using lat/lon values
meshlon, meshlat = np.meshgrid(len_lon, len_lat)

#Set random seed for reproducibility
random.seed(42)

#Generate random lat/lon pairs
num_of_pairs_each_time = 200
random_lat_lon_idx_pairs_val = np.empty((len(clavrx_flist_val)*num_of_pairs_each_time, 2)).astype(int)
random_lat_lon_idx_pairs_trng = np.empty((len(clavrx_flist_trng)*num_of_pairs_each_time, 2)).astype(int)
random_lat_lon_idx_pairs_test = np.empty((len(clavrx_flist_test)*num_of_pairs_each_time, 2)).astype(int)

for idx in range(len(clavrx_flist_val)*num_of_pairs_each_time):
    lat_idx = random.randint(0, np.shape(len_lat)[0] - 1)
    lon_idx = random.randint(0, np.shape(len_lon)[0] - 1)
    random_lat_lon_idx_pairs_val[idx,0] = lat_idx
    random_lat_lon_idx_pairs_val[idx,1] = lon_idx

for idx in range(len(clavrx_flist_trng)*num_of_pairs_each_time):
    lat_idx = random.randint(0, np.shape(len_lat)[0] - 1)
    lon_idx = random.randint(0, np.shape(len_lon)[0] - 1)
    random_lat_lon_idx_pairs_trng[idx,0] = lat_idx
    random_lat_lon_idx_pairs_trng[idx,1] = lon_idx

for idx in range(len(clavrx_flist_test)*num_of_pairs_each_time):
    lat_idx = random.randint(0, np.shape(len_lat)[0] - 1)
    lon_idx = random.randint(0, np.shape(len_lon)[0] - 1)
    random_lat_lon_idx_pairs_test[idx,0] = lat_idx
    random_lat_lon_idx_pairs_test[idx,1] = lon_idx

# %%

#Loop through files, pull out data based on selected lat/lon indexes and place into dataframe

#Validation Data
for idx in range(len(clavrx_flist_val)):

    #Load clavrx data and update values to 0 and 1
    clavrx_load = xr.open_dataset(clavrx_flist_val[idx])
    cloud_mask_data = np.squeeze(clavrx_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    cloud_mask = np.empty(cloud_mask_data.shape)
    cloud_mask[(cloud_mask_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    cloud_mask[(cloud_mask_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load clavrx data and update values to 0 and 1
    persist_load = xr.open_dataset(persist_flist_val[idx])
    persist_data = np.squeeze(persist_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    persist_mask = np.empty(persist_data.shape)
    persist_mask[(persist_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    persist_mask[(persist_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load GFS data 
    GFS_load = xr.open_dataset(GFS_flist_val[idx])
    isobaric = GFS_load['isobaric'].data
    relative_humidity_data = np.squeeze(GFS_load['relative_humidity'].data)
    vertical_velocity_data = np.squeeze(GFS_load['vertical_velocity'].data)
    temperature_data = np.squeeze(GFS_load['temperature'].data)
    absolute_vorticity_data = np.squeeze(GFS_load['absolute vorticity'].data)
    cloud_mixing_ratio_data = np.squeeze(GFS_load['cloud_mixing_ratio'].data)

    # Initialize an empty dictionary to store the data for each variable
    data_dict = {}

    # Variable names
    variable_names = ['Cld_Msk', 'Cld_Msk_Persist','RH', 'VV', 'Temp', 'AbsVort', 'Cld_Mix_Ratio']  

    #Current lat/lon index values
    pair_idx_1 = idx * num_of_pairs_each_time
    pair_idx_2 = (idx * num_of_pairs_each_time) + num_of_pairs_each_time

    # Loop through variable names
    for variable in variable_names:

        #Add Cld_Msk values        
        if variable == 'Cld_Msk':
            
            data = cloud_mask[random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data
        
        #Add Cld_Msk values        
        if variable == 'Cld_Msk_Persist':
            
            data = persist_mask[random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data

        # Loop through pressure levels
        for pressure_level in isobaric:
            # Create column name
            column_name = f'{variable}_{pressure_level}mb'
            
            # Extract data for the current variable and pressure level
            if variable == 'RH':
                data = relative_humidity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'VV':
                data = vertical_velocity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Temp':
                data = temperature_data[isobaric == pressure_level, random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'AbsVort':
                data = absolute_vorticity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Cld_Mix_Ratio':
                data = cloud_mixing_ratio_data[isobaric == pressure_level, random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_val[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

    if idx == 0: #If first file...create dataframe
        df_val = pd.DataFrame(data_dict)
    else: #If any other...append dataframe to first
        next_df_val = pd.DataFrame(data_dict)
        df_val = pd.concat([df_val, next_df_val], ignore_index=True)

    print(f'{idx + 1}/{len(clavrx_flist_val)} completed', end='\r')

#Trng Data
for idx in range(len(clavrx_flist_trng)):
    #Load clavrx data and update values to 0 and 1
    clavrx_load = xr.open_dataset(clavrx_flist_trng[idx])
    cloud_mask_data = np.squeeze(clavrx_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    cloud_mask = np.empty(cloud_mask_data.shape)
    cloud_mask[(cloud_mask_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    cloud_mask[(cloud_mask_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load clavrx data and update values to 0 and 1
    persist_load = xr.open_dataset(persist_flist_trng[idx])
    persist_data = np.squeeze(persist_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    persist_mask = np.empty(persist_data.shape)
    persist_mask[(persist_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    persist_mask[(persist_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load GFS data 
    GFS_load = xr.open_dataset(GFS_flist_trng[idx])
    isobaric = GFS_load['isobaric'].data
    relative_humidity_data = np.squeeze(GFS_load['relative_humidity'].data)
    vertical_velocity_data = np.squeeze(GFS_load['vertical_velocity'].data)
    temperature_data = np.squeeze(GFS_load['temperature'].data)
    absolute_vorticity_data = np.squeeze(GFS_load['absolute vorticity'].data)
    cloud_mixing_ratio_data = np.squeeze(GFS_load['cloud_mixing_ratio'].data)

    # Initialize an empty dictionary to store the data for each variable
    data_dict = {}

    # Variable names
    variable_names = ['Cld_Msk', 'Cld_Msk_Persist','RH', 'VV', 'Temp', 'AbsVort', 'Cld_Mix_Ratio']  

    #Current lat/lon index values
    pair_idx_1 = idx * num_of_pairs_each_time
    pair_idx_2 = (idx * num_of_pairs_each_time) + num_of_pairs_each_time

    # Loop through variable names
    for variable in variable_names:

        #Add Cld_Msk values        
        if variable == 'Cld_Msk':
            
            data = cloud_mask[random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data
        
        #Add Cld_Msk values        
        if variable == 'Cld_Msk_Persist':
            
            data = persist_mask[random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data

        # Loop through pressure levels
        for pressure_level in isobaric:
            # Create column name
            column_name = f'{variable}_{pressure_level}mb'
            
            # Extract data for the current variable and pressure level
            if variable == 'RH':
                data = relative_humidity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'VV':
                data = vertical_velocity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Temp':
                data = temperature_data[isobaric == pressure_level, random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'AbsVort':
                data = absolute_vorticity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Cld_Mix_Ratio':
                data = cloud_mixing_ratio_data[isobaric == pressure_level, random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_trng[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

    if idx == 0: #If first file...create dataframe
        df_trng = pd.DataFrame(data_dict)
    else: #If any other...append dataframe to first
        next_df_trng = pd.DataFrame(data_dict)
        df_trng = pd.concat([df_trng, next_df_trng], ignore_index=True)
    print(f'{idx + 1}/{len(clavrx_flist_trng)} completed',end='\r')

#Test Data
for idx in range(len(clavrx_flist_test)):

    #Load clavrx data and update values to 0 and 1
    clavrx_load = xr.open_dataset(clavrx_flist_test[idx])
    cloud_mask_data = np.squeeze(clavrx_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    cloud_mask = np.empty(cloud_mask_data.shape)
    cloud_mask[(cloud_mask_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    cloud_mask[(cloud_mask_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load clavrx data and update values to 0 and 1
    persist_load = xr.open_dataset(persist_flist_test[idx])
    persist_data = np.squeeze(persist_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    persist_mask = np.empty(persist_data.shape)
    persist_mask[(persist_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    persist_mask[(persist_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load GFS data 
    GFS_load = xr.open_dataset(GFS_flist_test[idx])
    isobaric = GFS_load['isobaric'].data
    relative_humidity_data = np.squeeze(GFS_load['relative_humidity'].data)
    vertical_velocity_data = np.squeeze(GFS_load['vertical_velocity'].data)
    temperature_data = np.squeeze(GFS_load['temperature'].data)
    absolute_vorticity_data = np.squeeze(GFS_load['absolute vorticity'].data)
    cloud_mixing_ratio_data = np.squeeze(GFS_load['cloud_mixing_ratio'].data)

    # Initialize an empty dictionary to store the data for each variable
    data_dict = {}

    # Variable names
    variable_names = ['Cld_Msk', 'Cld_Msk_Persist','RH', 'VV', 'Temp', 'AbsVort', 'Cld_Mix_Ratio']  

    #Current lat/lon index values
    pair_idx_1 = idx * num_of_pairs_each_time
    pair_idx_2 = (idx * num_of_pairs_each_time) + num_of_pairs_each_time

    # Loop through variable names
    for variable in variable_names:

        #Add Cld_Msk values        
        if variable == 'Cld_Msk':
            
            data = cloud_mask[random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data
        
        #Add Cld_Msk values        
        if variable == 'Cld_Msk_Persist':
            
            data = persist_mask[random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data

        # Loop through pressure levels
        for pressure_level in isobaric:
            # Create column name
            column_name = f'{variable}_{pressure_level}mb'
            
            # Extract data for the current variable and pressure level
            if variable == 'RH':
                data = relative_humidity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'VV':
                data = vertical_velocity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Temp':
                data = temperature_data[isobaric == pressure_level, random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'AbsVort':
                data = absolute_vorticity_data[isobaric == pressure_level, random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Cld_Mix_Ratio':
                data = cloud_mixing_ratio_data[isobaric == pressure_level, random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs_test[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

    if idx == 0: #If first file...create dataframe
        df_test = pd.DataFrame(data_dict)
    else: #If any other...append dataframe to first
        next_df_test = pd.DataFrame(data_dict)
        df_test = pd.concat([df_test, next_df_test], ignore_index=True)
        print(len(df_test))

    print(f'{idx + 1}/{len(clavrx_flist_test)} completed', end='\r')
#%%

# #Save Dataframe
# df_val.to_csv('HW3_data_val.csv', index=False)
# df_trng.to_csv('HW3_data_trng.csv', index=False)
# df_test.to_csv('HW3_data_test.csv', index=False)

#Load a DataFrame from a CSV file...run if needed
df_val = pd.read_csv('HW3_data_val.csv')
df_trng = pd.read_csv('HW3_data_trng.csv')
df_test = pd.read_csv('HW3_data_test.csv')

#%%

# Split for Trng
X_trng = df_trng.drop(columns=['Cld_Msk','Cld_Msk_Persist'])
y_trng = df_trng[['Cld_Msk']]
y_trng_baseline = df_trng['Cld_Msk_Persist']

# Split for Val
X_val = df_val.drop(columns=['Cld_Msk','Cld_Msk_Persist'])
y_val = df_val['Cld_Msk']
y_val_baseline = df_val['Cld_Msk_Persist']

# Split for Test
X_test = df_test.drop(columns=['Cld_Msk','Cld_Msk_Persist'])
y_test = df_test['Cld_Msk']
y_test_baseline = df_test['Cld_Msk_Persist']

#%%
#Section to apply PCA to reduce dimensions in data

# #Merge the X value dataframes before applying PCA
# merged_df = pd.merge(X_trng, )

pca = PCA(n_components=.95)
pca.fit(X_trng)
X_reduced_trng = pca.transform(X_trng)
X_reduced_val = pca.transform(X_val)
X_reduced_test = pca.transform(X_test)

# %%
#Define random forest and train model

#Define Hyperparameters
fd = {
    "tree_number": 15,    # number of trees to "average" together to create a random forest
    "tree_depth": 8,      # maximum depth allowed for each tree
    "node_split": 50,     # minimum number of training samples needed to split a node
    "leaf_samples": 50,    # minimum number of training samples required to make a leaf node
    "criterion": 'gini',  # information gain metric, 'gini' or 'entropy'
    "bootstrap": False,   # whether to perform "bagging=bootstrap aggregating" or not
    "max_samples": None,  # number of samples to grab when training each tree IF bootstrap=True, otherwise None 
    "random_state": 13    # set random state for reproducibility
}

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators = fd["tree_number"],
                           random_state = fd["random_state"],
                           min_samples_split = fd["node_split"],
                           min_samples_leaf = fd["leaf_samples"],
                           criterion = fd["criterion"],
                           max_depth = fd["tree_depth"],
                           bootstrap = fd["bootstrap"],
                           max_samples = fd["max_samples"])

#Train random forest
rf_classifier.fit(X_reduced_trng, y_trng)

#Make prediction on all training data
y_pred_trng = rf_classifier.predict(X_reduced_trng)

#%%
#Confusion Matrix on training data

acc = metrics.accuracy_score(y_trng, y_pred_trng)
print("training accuracy: ", np.around(acc*100), '%')

confusion = confusion_matrix(y_trng, y_pred_trng)

print(confusion)

pred_classes = ['Pred No Cloud', 'Pred Cloud']
true_classes = ['True No Cloud', 'True Cloud']

def confusion_matrix_plot(predclasses, targclasses, pred_classes, true_classes):
  class_names = np.unique(targclasses)
  table = []
  for pred_class in class_names:
    row = []
    for true_class in class_names:
        row.append(100 * np.mean(predclasses[targclasses == true_class] == pred_class))
    table.append(row)
  class_titles_t = true_classes
  class_titles_p = pred_classes
  conf_matrix = pd.DataFrame(table, index=class_titles_t, columns=class_titles_p)
  display(conf_matrix.style.background_gradient(cmap='Greens').format("{:.1f}"))

#Plot Confusion Matrix
confusion_matrix_plot(y_pred_trng, y_trng['Cld_Msk'], pred_classes, true_classes)

#Plot Confusion Matrix for baseline
confusion_matrix_plot(y_trng_baseline, y_trng['Cld_Msk'], pred_classes, true_classes)

acc = metrics.accuracy_score(y_trng, y_trng_baseline)
print("baseline training accuracy: ", np.around(acc*100), '%')

#Confusion numbers for baseline
confusion = confusion_matrix(y_trng, y_trng_baseline)
print(confusion)

#%%
#Confusion Matrix on validation data

#Make prediction on validation data
y_pred_val = rf_classifier.predict(X_reduced_val)

acc = metrics.accuracy_score(y_val, y_pred_val)
print("validation accuracy: ", np.around(acc*100), '%')

confusion_validation = confusion_matrix(y_val, y_pred_val)

print(confusion_validation)

#Plot Confusion Matrix
confusion_matrix_plot(y_pred_val, y_val, pred_classes, true_classes)

#Plot Confusion Matrix for baseline
confusion_matrix_plot(y_val_baseline, y_val, pred_classes, true_classes)

acc = metrics.accuracy_score(y_val, y_val_baseline)
print("baseline validation accuracy: ", np.around(acc*100), '%')

#Confusion numbers for baseline
confusion = confusion_matrix(y_val, y_val_baseline)
print(confusion)

#%%
# #Look at individual tree
# local_path = '/home/mking/ATS780_HW2/'
# fig_savename = 'rf_cloud_tree'
# tree_to_plot = 0 # Enter the value of the tree that you want to see!

# #Get predictor feature names
# column_names = X_trng.columns
# column_names = column_names.tolist()

# tree = rf_classifier[tree_to_plot] # Obtain the tree to plot
# tree_numstr = str(tree_to_plot) # Adds the tree number to filename

# complete_savename = fig_savename + '_' + tree_numstr + '.dot'
# export_graphviz(tree,
#                 out_file=local_path + '/' + complete_savename,
#                 filled=True,
#                 proportion=False,
#                 leaves_parallel=False,
#                 feature_names=column_names)

# Source.from_file(local_path + complete_savename)

#%%
#Feature importance

def calc_importances(rf, feature_list):
    ''' Calculate feature importance '''
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Print out the feature and importances 
    print('')
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    print('')

    return importances

def plot_feat_importances(importances, feature_list):
    ''' Plot the feature importance calculated by calc_importances ''' 
    plt.figure(figsize=(19,35))
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.barh(x_values, importances)
    # Tick labels for x axis
    plt.yticks(x_values, feature_list)
    # Axis labels and title
    plt.xlabel('Importance'); plt.ylabel('Variable'); plt.title('Variable Importances')
    
    
# plot_feat_importances(calc_importances(rf_classifier, column_names),  column_names)


# %% 
# Evaluate the model on test data
y_pred = rf_classifier.predict(X_reduced_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

#Confusion numbers for baseline
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

#Plot Confusion Matrix
confusion_matrix_plot(y_pred, y_test, pred_classes, true_classes)

#Plot Confusion Matrix for baseline
confusion_matrix_plot(y_test_baseline, y_test, pred_classes, true_classes)

acc = metrics.accuracy_score(y_test, y_test_baseline)
print("baseline validation accuracy: ", np.around(acc*100), '%')

#Confusion numbers for baseline
confusion = confusion_matrix(y_test, y_test_baseline)
print(confusion)

#%%

#Build Neural Network
settings = {
    "hiddens": [50, 50, 50],
    "activations": ["relu", "relu", "relu"],
    "learning_rate": 0.0001,
    "random_seed": 33,
    "max_epochs": 100,
    "batch_size": 64,
    "patience": 20,
    "dropout_rate": 0.,
}

def build_model(x_train, y_train, settings):
    # create input layer
    input_layer = tf.keras.layers.Input(shape=x_train.shape[1:])

    # create a normalization layer if you would like
    normalizer = tf.keras.layers.Normalization(axis=(1,))
    normalizer.adapt(x_train)
    layers = normalizer(input_layer)

    # create hidden layers each with specific number of nodes
    assert len(settings["hiddens"]) == len(
        settings["activations"]
    ), "hiddens and activations settings must be the same length."

    # add dropout layer
    layers = tf.keras.layers.Dropout(rate=settings["dropout_rate"])(layers)

    for hidden, activation in zip(settings["hiddens"], settings["activations"]):
        layers = tf.keras.layers.Dense(
            units=hidden,
            activation=activation,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"]),
        )(layers)

    # # create output layer
    # output_layer = tf.keras.layers.Dense(
    #     units=y_train.shape[-1],
    #     activation="linear",
    #     use_bias=True,
    #     bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 1),
    #     kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 2),
    # )(layers)

    # create output layer
    output_layer = tf.keras.layers.Dense(
        units=y_train.shape[-1],
        activation="sigmoid",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 1),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 2),
    )(layers)    

    # construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model


def compile_model(model, settings):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
        ],
    )
    return model

# def compile_model(model, settings):
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
#         loss='mse',
#         metrics='mae')
    
#     return model

#%%

#Converting pandas dataframes to numpy arrays
# X_trng_np = X_trng.to_numpy()
y_trng_np = y_trng.to_numpy()
y_trng_np = (y_trng_np).astype(int)
y_trng_hot = np.eye(2)[y_trng_np.flatten()]
# X_val_np = X_val.to_numpy()
y_val_np = y_val.to_numpy()
y_val_np = (y_val_np).astype(int)
y_val_hot = np.eye(2)[y_val_np.flatten()]
# X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy()

# X_reduced_trng = pca.transform(X_trng)
# X_reduced_val = pca.transform(X_val)
# X_reduced_test = pca.transform(X_test)

Cloud_Mask_Model = build_model(X_reduced_trng, y_trng_hot, settings)

Cloud_Mask_Model = compile_model(Cloud_Mask_Model, settings)

# define the early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=settings["patience"], restore_best_weights=True, mode="auto")

# # define the class weights
# class_weights = {
#     0: 1 / np.mean(Ttrain[:, 0] == 1),
#     1: 1 / np.mean(Ttrain[:, 1] == 1),
# }

history = Cloud_Mask_Model.fit(X_reduced_trng,y_trng_hot, 
                         epochs = settings["max_epochs"], 
                         batch_size=settings["batch_size"], 
                         shuffle=True,
                         validation_data=[X_reduced_val,y_val_hot],
                         callbacks=[early_stopping_callback],
                        #  class_weight = class_weights,
                         verbose = 1)

#Print names of items stored in history...then use to plot loss history
print(history.history.keys())

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
# Print evolution of loss - separately for training and validation data
axs[0].plot(history.epoch, history.history['loss'], label='Trng Loss', color='blue')
axs[0].plot(history.epoch, history.history['val_loss'], label='Val Loss', color='red')
axs[0].set_title(f'Training/Validation Loss History: {settings["hiddens"][0]} nodes per hidden layer')
axs[0].legend()
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('Categorical Crossentropy')
# Plot Mean absolute error of training & validation data
axs[1].plot(history.epoch, history.history['categorical_accuracy'], label='Trng Cat Accuracy', color='blue')
axs[1].plot(history.epoch, history.history['val_categorical_accuracy'], label='Val Cat Accuracy', color='red')
axs[1].set_title(f'Training/Validation Accuracy History: {settings["hiddens"][0]} nodes per hidden layer')
axs[1].legend()
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('Categorical Accuracy')

#Save figure
#Before saving figure...check for file path
path_test = os.path.exists('Plots/')
if (path_test == False): #If path doesn't exist...create folder
    os.mkdir('Plots/')
plt.savefig(f'Plots/Trng_Loss_Hiddens_{settings["hiddens"][0]}.png')
plt.show()
plt.close()


# %%
NN_pred_trng_hot = Cloud_Mask_Model.predict(X_reduced_trng)
NN_pred_val_hot = Cloud_Mask_Model.predict(X_reduced_val)

#Convert predictions to binary arrays
NN_pred_trng_hot = np.round(NN_pred_trng_hot)
NN_pred_trng = np.argmax(NN_pred_trng_hot, axis=1).reshape(-1, 1)
NN_pred_val_hot = np.round(NN_pred_val_hot)
NN_pred_val = np.argmax(NN_pred_val_hot, axis=1).reshape(-1, 1)

#Confusion Matrix on training data
acc = metrics.accuracy_score(y_trng, NN_pred_trng)
print("training accuracy: ", np.around(acc*100), '%')
confusion = confusion_matrix(y_trng, NN_pred_trng)
print(confusion)

#Plot Confusion Matrix
confusion_matrix_plot(NN_pred_trng, y_trng, pred_classes, true_classes)

#Confusion Matrix on training data
acc = metrics.accuracy_score(y_val, NN_pred_val)
print("validation accuracy: ", np.around(acc*100), '%')
confusion = confusion_matrix(y_val_baseline, NN_pred_val)
print(confusion)

#Plot Confusion Matrix
confusion_matrix_plot(NN_pred_val, y_val, pred_classes, true_classes)

# %%

#Test data
NN_pred_test_hot = Cloud_Mask_Model.predict(X_reduced_test)
NN_pred_test_hot = np.round(NN_pred_test_hot)
NN_pred_test = np.argmax(NN_pred_test_hot, axis=1).reshape(-1, 1)

#Confusion Matrix on training data
acc = metrics.accuracy_score(y_test, NN_pred_test)
print("testing accuracy: ", np.around(acc*100), '%')
confusion = confusion_matrix(y_test, NN_pred_test)
print(confusion)

#Plot Confusion Matrix
confusion_matrix_plot(NN_pred_test, y_test, pred_classes, true_classes)


#%%
#Confusion Matrix for test data for random forest baseline

#Make prediction on validation data
y_pred_test = rf_classifier.predict(X_reduced_test)

acc = metrics.accuracy_score(y_test, y_pred_test)
print("validation accuracy: ", np.around(acc*100), '%')

confusion_validation = confusion_matrix(y_test, y_pred_test)

print(confusion_validation)

#Plot Confusion Matrix
confusion_matrix_plot(y_pred_test, y_test, pred_classes, true_classes)
# %%
