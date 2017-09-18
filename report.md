[World1_UN]: ./misc_images/World1-UN.png
[World2_UN]: ./misc_images/World2-UN.png
[World3_UN]: ./misc_images/World3-UN.png
[World1_N]: ./misc_images/World1-N.png
[World2_N]: ./misc_images/World2-N.png
[World3_N]: ./misc_images/World3-N.png
[World1]: ./misc_images/world1.PNG
[World2]: ./misc_images/world_2.PNG
[World3]: ./misc_images/world_3.PNG
[table_top]: ./misc_images/pr2_world.PNG
[noise_removed]: ./misc_images/noise_removed.PNG

Following steps in [Exercise-1](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-1/report.md), [Exercise-2](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-2/report.md), [Exercise-3](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-3/report.md) we should be able to detect and label objects placed on table top.

However, this time the world we are operatig is different and objects are different. 

### Key differences from previous project

RGB-D camera is modelled with noise this time. The point cloud data obtained from camera includes noise along with the ground truth. Using statistical filtering techniques noise is removed.

#### Noise image
![alt text][table_top] <br/>

#### Image after statistical filtering is applied
![alt text][noise_removed] <br/>

``` python
outlier_filter = cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50)
x = 1
outlier_filter.set_std_dev_mul_thresh(x)
noise_filtered = outlier_filter.filter()
```

Different world dimensions and environment. This will affect the pass through filter parameters.
#### Filtering along Z-direction to remove table stand.

``` python
pass_thorugh_z = down_sampled.make_passthrough_filter()
filter_axis = 'z'
pass_thorugh_z.set_filter_field_name(filter_axis)
axis_min = 0.61
axis_max = 1.5
pass_thorugh_z.set_filter_limits(axis_min,axis_max)
cloud_filtered_z = pass_thorugh_z.filter()
```

#### Filtering along Y-direction to remove table sides and unwanted area where objects are not placed.

``` python
pass_thorugh_y = cloud_filtered_z.make_passthrough_filter()
filter_axis = 'y'
pass_thorugh_y.set_filter_field_name(filter_axis)
axis_min = -0.5
axis_max = 0.5
pass_thorugh_y.set_filter_limits(axis_min,axis_max)
```

Keeping track of above difference and following previous steps from [Exercise-1](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-1/report.md), [Exercise-2](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-2/report.md), [Exercise-3](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-3/report.md)  we should be able to detect and label objects from different worlds.

To train the data from different worlds 'models` array in [capture_features.py](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-3/sensor_stick/scripts/capture_features.py)  is changed to different objects stated in [pick_list_1.yaml](https://github.com/rupimanoj/Perception-3D/blob/master/pr2_robot/config/pick_list_1.yaml) , [pick_list_2.yaml](https://github.com/rupimanoj/Perception-3D/blob/master/pr2_robot/config/pick_list_2.yaml),  [pick_list_3.yaml](https://github.com/rupimanoj/Perception-3D/blob/master/pr2_robot/config/pick_list_3.yaml) files.

### Capturing training data for differnt worlds and training SVM classifier.

To launch training environment <br/>
`roslaunch sensor_stick training.launch`<br/><br/>

To capture training data <br/>
`rosrun sensor_stick capture_features.py ` <br/><br/>

TO train model using SVM <br/>
`rosrun sensor_stick train_svm.py` <br/><br/>

<b> In this world environment 'sigmoid' kernel is used for SVM model. </b>

On training data with learned SVM model, we get below results on different worlds.

## World1 Results

```
Features in Training Set: 150 <br/>
Invalid Features in Training set: 0 <br/>
Scores: [ 0.93333333  1.          0.96666667  0.9         0.93333333] <br/><br/>
Accuracy: 0.95 (+/- 0.07) <br/>
accuracy score: 0.946666666667  <br/>
```

![alt text][World1_UN] <br/>
![alt text][World1_N] <br/>
![alt text][World1] <br/>

## World2 Results

```
Features in Training Set: 250 <br/>
Invalid Features in Training set: 1 <br/>
Scores: [ 0.98        0.94        0.98        0.96        0.95918367] <br/>
Accuracy: 0.96 (+/- 0.03) <br/>
accuracy score: 0.963855421687 <br/>
```

![alt text][World2_UN] <br/>
![alt text][World2_N] <br/>
![alt text][World2] <br/>

## World3 Results

``` 
Features in Training Set: 400 <br/>
Invalid Features in Training set: 0 <br/>
Scores: [ 0.975   0.9125  0.9625  0.9875  0.9625] <br/>
Accuracy: 0.96 (+/- 0.05) <br/>
accuracy score: 0.96 <br/>
```

![alt text][World3_UN] <br/>
![alt text][World3_N] <br/>
![alt text][World3] <br/>

In world3, glue objec is not getting detected as clustering logic is not able to capture it when it is hiding behind other objects. The pass scenario in this case is 7/8 . <br/>

With the above trained models, we should be able to detect and label objects in different worlds. Rest of the report discusses about how to save the output details of objects to be picked.

## Writing pick and place locations to output files

Once indvidual object labels and point clouds are stored in detected, individual objects centroid is calculated so that pr2_robot can pick up object succesfully.

``` python
points_arr = ros_to_pcl(pcl_cluster_ros).to_array()
center = np.mean(points_arr, axis=0)[:3]
center_float = []
for item in center:
	center_float.append(np.asscalar(item))
centroids.append(center_float)
```

To output the details into yaml file, `publish_to_yaml` function is used.

``` python

def publish_to_yaml(labels,centroids):

	world_no = 1  ## change according to the world environment
	out_file_name = 'output_1.yaml'

	object_list_param = rospy.get_param('/object_list')
	dropbox_params = rospy.get_param('/dropbox')
	left_drop_position = []
	right_drop_position = []

	for k in range(0, len(dropbox_params)):
		if(dropbox_params[k]['group'] == 'red'):
			left_drop_position = dropbox_params[k]['position']
		if(dropbox_params[k]['group'] == 'green'):

			right_drop_position = dropbox_params[k]['position']
			
	test_scene_num = Int32()
	test_scene_num.data = world_no
	object_name = String()
	arm_name = String()
	pick_pose = Pose()
	place_pose = Pose()

	dict_list = []

	for item in object_list_param:
		for index,label in enumerate(labels):
			if(item['name'] == label):
				
				object_name.data = item['name']
				if(item['group'] == 'red'):
					arm_name.data = 'left'
					place_pose.position.x = left_drop_position[0]
					place_pose.position.y = left_drop_position[1]
					place_pose.position.z = left_drop_position[2]
				if(item['group'] == 'green'):
					arm_name.data = 'right'
					place_pose.position.x = right_drop_position[0]
					place_pose.position.y = right_drop_position[1]
					place_pose.position.z = right_drop_position[2]

				pick_pose.position.x = centroids[index][0]
				pick_pose.position.y = centroids[index][1]
				pick_pose.position.z = centroids[index][2]
				yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
				dict_list.append(yaml_dict)
	
	send_to_yaml(out_file_name,dict_list)

```

Objects to be picked by pr2_robot are obtained by reading parameters `object_list` from ros server. 
If `detected_objects` contain any object from `objects_list` we fill relevant details.

	`object_list_param = rospy.get_param('/object_list')` <br/>

By traversing through above parameter list, we can get `object_name`, `arm_name`. <br/>

Objects place location is found by reading parametes from `dropbox` ros server. By reading parameters `place_pose` can be filled.
	`dropbox_params = rospy.get_param('/dropbox')` <br/>

'pick_pose' is filled using previously filled centroid location of objects.

Once all the required details are collected, details are written to output file using artifact function `send_to_yaml`.