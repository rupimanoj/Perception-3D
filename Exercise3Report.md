[//]: # (Image References)

[cluster_3_msg]: ./misc_images/cluster_3.PNG
[flat_top_msg]: ./misc_images/flat_top.PNG
[object_iden_msg]: ./misc_images/object_identification.PNG
[object_reco_msg]: ./misc_images/object_reco_3.PNG
[CM_1]: ./misc_images/figure_1.png
[CM_2]: ./misc_images/figure_2.png

## Object Recognition Pipeline:

As part of this project, we will identify each object placed on table top.
![alt text][object_reco_msg]

### Collecting training data:

To collect the training data [capture_features.py](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-3/sensor_stick/scripts/capture_features.py) script is used, which in turn uses functions `compute_color_histograms` and `compute_normal_histograms` to capture features.



##### Color histograms features:

To capture histogram features, by default `nbins=32` ( number of bins the color pixel values are divided into). As there are three channels  (R,G,B, /H,S,V) total number of color histogram features captured per each point cloud image will be 96. TO remove lighting effects of captured data, it is preferable to use HSV features.

``` python
def compute_color_histograms(cloud, using_hsv=True):

    point_colors_list = []

    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    h_hist = np.histogram(channel_1_vals,bins = nbins , range = bins_range)
    s_hist = np.histogram(channel_2_vals, bins= nbins, range = bins_range)
    v_hist = np.histogram(channel_3_vals, bins = nbins , range = bins_range)

    features_hist = np.concatenate((h_hist[0],s_hist[0],v_hist[0])).astype(np.float64)
    norm_features = features_hist/(np.sum(features_hist))
    return norm_features

```

##### Noramals histogram features:

Similar to color histograms, normal histogram features are captured. Instead of 3 color channels, here X,Y,Z normal components features are captured. Total number of features will be 96.

``` python
def compute_normal_histograms(normal_cloud):
    
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    x_hist = np.histogram(norm_x_vals,bins = nbins , range = bins_range)
    y_hist = np.histogram(norm_y_vals, bins= nbins, range = bins_range)
    z_hist = np.histogram(norm_z_vals, bins = nbins , range = bins_range)

    features_hist = np.concatenate((x_hist[0],y_hist[0],z_hist[0])).astype(np.float64)
    norm_features = features_hist/(np.sum(features_hist))

    return norm_features

```

To capture features, training environment is launched in Gazebo and `capture_features` rosnode is invoked.

`roslaunch sensor_stick training.launch` <br/>
`rosrun sensor_stick capture_features.py`

Each object staed in array `models` is captured for 50 times by keeping objects in different orientations. Features are captured in `training_set.sav file`.

``` python

    models = [\
       'beer',
       'bowl',
       'create',
       'disk_part',
       'hammer',
       'plastic_cup',
       'soda_can']

```

### Training classifier using SVM:

Using SVM model and traing data from 'training_set.sav' file, classifier is trained.

For SVM classifier in this exercise, linear kernel is used and cross validation for 5 folds is used.

TO launch SVM classifier trainer rosnode is launched using below command.

`rosrun sensor_stick train_svm.py`

After classifier is learned, on training data below confusion matrix is obtained.

![alt text][CM_1]
![alt text][CM_2]

SVM Model is saved in model.sav file.

### Classifying objects and Tagging with appropriate names:


Trained model is loaded using python pickle module.

``` python

model = pickle.load(open('model.sav', 'rb'))
clf = model['classifier']
encoder = LabelEncoder()
encoder.classes_ = model['classes']
scaler = model['scaler']

```

By following steps in [Exercise-1](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-1/report.md)  and [Exercise-2](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-2/report.md) , individual objects can be extracted using Elucidean clustering techniques.
USing individual objects, features are calculated using `compute_color_histograms` and `compute_normal_histograms`.
Once features are obtained, object can be predicted using `clf.predict` 
To transform the predicted target into label , `inverse_transform' API from LabelEncoder class is used.
For further processing of object labels and point clouds are stored in 'detected_objects'.
Detected objects are published using `detected_objects_pub` publisher. <br/>

![alt text][cluster_3_msg]
![alt text][object_iden_msg]

`detected_objects_pub.publish(detected_objects)` <br/>

```python

detected_objects_labels = []
detected_objects = []
# Classify the clusters! (loop through each detected cluster one at a time)
for index, pts_list in enumerate(cluster_indices):
	 # Grab the points for the cluster       
	pcl_cluster = obj_msg.extract(pts_list)
	pcl_cluster_ros = pcl_to_ros(pcl_cluster)
	# Compute the associated feature vector
	chists = compute_color_histograms(pcl_cluster_ros)
	normals = get_normals(pcl_cluster_ros)
	nhists = compute_normal_histograms(normals)
	feature = np.concatenate((chists, nhists))
	# Make the prediction
	prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
	label = encoder.inverse_transform(prediction)[0]
	detected_objects_labels.append(label)
	# Publish a label into RViz
	label_pos = list(white_cloud[pts_list[0]])
	label_pos[2] += .4
	object_markers_pub.publish(make_label(label,label_pos, index))

	# Add the detected object to the list of detected objects.
	do = DetectedObject()
	do.label = label
	do.cloud = pcl_cluster_ros
	detected_objects.append(do)

# Publish the list of detected objects
rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
detected_objects_pub.publish(detected_objects)

```



