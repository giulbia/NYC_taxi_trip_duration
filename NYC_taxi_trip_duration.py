import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
# import math

from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Functions for feature engineering
def haversine_(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)


def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick_drop"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    """ function was taken from beluga's notebook as this function works on array
    while my function used to work on individual elements and was noticably slow"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def assign_cluster(df, k):
    """function to assign clusters """
    df_pick = df[['pickup_longitude','pickup_latitude']]
    df_drop = df[['dropoff_longitude','dropoff_latitude']]
    """I am using initialization as from the output of
    k-means from my local machine to save time in this kernel"""
    init = np.array([[ -73.98737616,   40.72981533],
                     [-121.93328857,   37.38933945],
                     [ -73.78423222,   40.64711269],
                     [ -73.9546417 ,   40.77377538],
                     [ -66.84140269,   36.64537175],
                     [ -73.87040541,   40.77016484],
                     [ -73.97316185,   40.75814346],
                     [ -73.98861094,   40.7527791 ],
                     [ -72.80966949,   51.88108444],
                     [ -76.99779701,   38.47370625],
                     [ -73.96975298,   40.69089596],
                     [ -74.00816622,   40.71414939],
                     [ -66.97216034,   44.37194443],
                     [ -61.33552933,   37.85105133],
                     [ -73.98001393,   40.7783577 ],
                     [ -72.00626526,   43.20296402],
                     [ -73.07618713,   35.03469086],
                     [ -73.95759366,   40.80316361],
                     [ -79.20167796,   41.04752096],
                     [ -74.00106031,   40.73867723]])
    k_means_pick = KMeans(n_clusters=k, init=init, n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick = k_means_pick.labels_
    df['label_pick'] = clust_pick.tolist()
    df['label_drop'] = k_means_pick.predict(df_drop)
    return df, k_means_pick


# Function for model evaluation
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# Main
def my_training_function(xgb_pars=None):

    # Read data
    train_df = pd.read_csv("data/train.csv", sep=",", skipinitialspace=True)
    test_df = pd.read_csv('data/test.csv', sep=",", skipinitialspace=True)



    # Feature engineering
    train_data = train_df.copy()
    train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)
    train_data.loc[:, 'pick_month'] = train_data['pickup_datetime'].dt.month
    train_data.loc[:, 'hour'] = train_data['pickup_datetime'].dt.hour
    train_data.loc[:, 'week_of_year'] = train_data['pickup_datetime'].dt.weekofyear
    train_data.loc[:, 'day_of_year'] = train_data['pickup_datetime'].dt.dayofyear
    train_data.loc[:, 'day_of_week'] = train_data['pickup_datetime'].dt.dayofweek
    train_data.loc[:,'hvsine_pick_drop'] = haversine_(train_data['pickup_latitude'].values,
                                                      train_data['pickup_longitude'].values,
                                                      train_data['dropoff_latitude'].values,
                                                      train_data['dropoff_longitude'].values)
    train_data.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(train_data['pickup_latitude'].values,
                                                                 train_data['pickup_longitude'].values,
                                                                 train_data['dropoff_latitude'].values,
                                                                 train_data['dropoff_longitude'].values)
    train_data.loc[:,'bearing'] = bearing_array(train_data['pickup_latitude'].values,
                                                train_data['pickup_longitude'].values,
                                                train_data['dropoff_latitude'].values,
                                                train_data['dropoff_longitude'].values)

    train_cl, k_means = assign_cluster(train_data, 20)
    centroid_pickups = pd.DataFrame(k_means.cluster_centers_, columns=['centroid_pick_long', 'centroid_pick_lat'])
    centroid_dropoff = pd.DataFrame(k_means.cluster_centers_, columns=['centroid_drop_long', 'centroid_drop_lat'])
    centroid_pickups['label_pick'] = centroid_pickups.index
    centroid_dropoff['label_drop'] = centroid_dropoff.index
    train_cl = pd.merge(train_cl, centroid_pickups, how='left', on=['label_pick'])
    train_cl = pd.merge(train_cl, centroid_dropoff, how='left', on=['label_drop'])

    train_cl.loc[:,'hvsine_pick_cent_p'] = haversine_(train_cl['pickup_latitude'].values,
                                                      train_cl['pickup_longitude'].values,
                                                      train_cl['centroid_pick_lat'].values,
                                                      train_cl['centroid_pick_long'].values)
    train_cl.loc[:,'hvsine_drop_cent_d'] = haversine_(train_cl['dropoff_latitude'].values,
                                                      train_cl['dropoff_longitude'].values,
                                                      train_cl['centroid_drop_lat'].values,
                                                      train_cl['centroid_drop_long'].values)
    train_cl.loc[:,'hvsine_cent_p_cent_d'] = haversine_(train_cl['centroid_pick_lat'].values,
                                                        train_cl['centroid_pick_long'].values,
                                                        train_cl['centroid_drop_lat'].values,
                                                        train_cl['centroid_drop_long'].values)
    train_cl.loc[:,'manhtn_pick_cent_p'] = manhattan_distance_pd(train_cl['pickup_latitude'].values,
                                                                 train_cl['pickup_longitude'].values,
                                                                 train_cl['centroid_pick_lat'].values,
                                                                 train_cl['centroid_pick_long'].values)
    train_cl.loc[:,'manhtn_drop_cent_d'] = manhattan_distance_pd(train_cl['dropoff_latitude'].values,
                                                                 train_cl['dropoff_longitude'].values,
                                                                 train_cl['centroid_drop_lat'].values,
                                                                 train_cl['centroid_drop_long'].values)
    train_cl.loc[:,'manhtn_cent_p_cent_d'] = manhattan_distance_pd(train_cl['centroid_pick_lat'].values,
                                                                   train_cl['centroid_pick_long'].values,
                                                                   train_cl['centroid_drop_lat'].values,
                                                                   train_cl['centroid_drop_long'].values)

    train_cl.loc[:,'bearing_pick_cent_p'] = bearing_array(train_cl['pickup_latitude'].values,
                                                          train_cl['pickup_longitude'].values,
                                                          train_cl['centroid_pick_lat'].values,
                                                          train_cl['centroid_pick_long'].values)
    train_cl.loc[:,'bearing_drop_cent_p'] = bearing_array(train_cl['dropoff_latitude'].values,
                                                          train_cl['dropoff_longitude'].values,
                                                          train_cl['centroid_drop_lat'].values,
                                                          train_cl['centroid_drop_long'].values)
    train_cl.loc[:,'bearing_cent_p_cent_d'] = bearing_array(train_cl['centroid_pick_lat'].values,
                                                            train_cl['centroid_pick_long'].values,
                                                            train_cl['centroid_drop_lat'].values,
                                                            train_cl['centroid_drop_long'].values)

    # Feature engineering on test set
    test_data = test_df.copy()
    test_data['pickup_datetime'] = pd.to_datetime(test_data.pickup_datetime)
    test_data.loc[:, 'pick_month'] = test_data['pickup_datetime'].dt.month
    test_data.loc[:, 'hour'] = test_data['pickup_datetime'].dt.hour
    test_data.loc[:, 'week_of_year'] = test_data['pickup_datetime'].dt.weekofyear
    test_data.loc[:, 'day_of_year'] = test_data['pickup_datetime'].dt.dayofyear
    test_data.loc[:, 'day_of_week'] = test_data['pickup_datetime'].dt.dayofweek

    test_data.loc[:,'hvsine_pick_drop'] = haversine_(test_data['pickup_latitude'].values,
                                                     test_data['pickup_longitude'].values,
                                                     test_data['dropoff_latitude'].values,
                                                     test_data['dropoff_longitude'].values)
    test_data.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(test_data['pickup_latitude'].values,
                                                                test_data['pickup_longitude'].values,
                                                                test_data['dropoff_latitude'].values,
                                                                test_data['dropoff_longitude'].values)
    test_data.loc[:,'bearing'] = bearing_array(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values,
                                               test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)

    test_data['label_pick'] = k_means.predict(test_data[['pickup_longitude','pickup_latitude']])
    test_data['label_drop'] = k_means.predict(test_data[['dropoff_longitude','dropoff_latitude']])
    test_cl = pd.merge(test_data, centroid_pickups, how='left', on=['label_pick'])
    test_cl = pd.merge(test_cl, centroid_dropoff, how='left', on=['label_drop'])

    test_cl.loc[:,'hvsine_pick_cent_p'] = haversine_(test_cl['pickup_latitude'].values,
                                                     test_cl['pickup_longitude'].values,
                                                     test_cl['centroid_pick_lat'].values,
                                                     test_cl['centroid_pick_long'].values)
    test_cl.loc[:,'hvsine_drop_cent_d'] = haversine_(test_cl['dropoff_latitude'].values,
                                                     test_cl['dropoff_longitude'].values,
                                                     test_cl['centroid_drop_lat'].values,
                                                     test_cl['centroid_drop_long'].values)
    test_cl.loc[:,'hvsine_cent_p_cent_d'] = haversine_(test_cl['centroid_pick_lat'].values,
                                                       test_cl['centroid_pick_long'].values,
                                                       test_cl['centroid_drop_lat'].values,
                                                       test_cl['centroid_drop_long'].values)
    test_cl.loc[:,'manhtn_pick_cent_p'] = manhattan_distance_pd(test_cl['pickup_latitude'].values,
                                                                test_cl['pickup_longitude'].values,
                                                                test_cl['centroid_pick_lat'].values,
                                                                test_cl['centroid_pick_long'].values)
    test_cl.loc[:,'manhtn_drop_cent_d'] = manhattan_distance_pd(test_cl['dropoff_latitude'].values,
                                                                test_cl['dropoff_longitude'].values,
                                                                test_cl['centroid_drop_lat'].values,
                                                                test_cl['centroid_drop_long'].values)
    test_cl.loc[:,'manhtn_cent_p_cent_d'] = manhattan_distance_pd(test_cl['centroid_pick_lat'].values,
                                                                  test_cl['centroid_pick_long'].values,
                                                                  test_cl['centroid_drop_lat'].values,
                                                                  test_cl['centroid_drop_long'].values)

    test_cl.loc[:,'bearing_pick_cent_p'] = bearing_array(test_cl['pickup_latitude'].values,
                                                         test_cl['pickup_longitude'].values,
                                                         test_cl['centroid_pick_lat'].values,
                                                         test_cl['centroid_pick_long'].values)
    test_cl.loc[:,'bearing_drop_cent_p'] = bearing_array(test_cl['dropoff_latitude'].values,
                                                         test_cl['dropoff_longitude'].values,
                                                         test_cl['centroid_drop_lat'].values,
                                                         test_cl['centroid_drop_long'].values)
    test_cl.loc[:,'bearing_cent_p_cent_d'] = bearing_array(test_cl['centroid_pick_lat'].values,
                                                           test_cl['centroid_pick_long'].values,
                                                           test_cl['centroid_drop_lat'].values,
                                                           test_cl['centroid_drop_long'].values)


    # ### Model

    train = train_cl
    test = test_cl

    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values,
                        test[['pickup_latitude', 'pickup_longitude']].values,
                        test[['dropoff_latitude', 'dropoff_longitude']].values))

    pca = PCA().fit(coords)

    train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
    train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
    train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
    test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
    test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    train['store_and_fwd_flag_int'] = np.where(train['store_and_fwd_flag']=='N', 0, 1)
    test['store_and_fwd_flag_int'] = np.where(test['store_and_fwd_flag']=='N', 0, 1)

    y = np.log(train['trip_duration'].values + 1)

    feature_names = list(train.columns)
    do_not_use_for_training = ['pick_date','id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag']
    feature_names = [f for f in train.columns if f not in do_not_use_for_training]

    Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test[feature_names].values)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # Set default parameters
    if xgb_pars is None:
        xgb_pars = {'min_child_weight': 50, 'eta': 0.4, 'colsample_bytree': 0.3, 'max_depth': 15,
                    'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 0,
                    'eval_metric': 'rmse', 'objective': 'reg:linear'}

    # MLflow
    with mlflow.start_run():
        
        # Train model, watch on dval
        model = xgb.train(xgb_pars, dtrain, 15, watchlist, early_stopping_rounds=2, maximize=False, verbose_eval=1)
        
        print('Modeling RMSLE %.5f' % model.best_score)
        
        # Evaluate metrics
        val_pred = model.predict(dvalid)
                
        (rmse, mae, r2) = eval_metrics(val_pred, dvalid.get_label())
        
        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("max_depth", xgb_pars["max_depth"])
        mlflow.log_param("lambda", xgb_pars["lambda"])
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")


if __name__ == "_main__":
    my_training_function()
