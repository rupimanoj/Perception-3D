#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
#from sensor_stick.pcl_helper import *
import pdb
import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String, Empty
from pr2_robot.srv import *
import std_srvs.srv
from rospy_message_converter import message_converter
import yaml
import time 
from pcl_helper import *
global pcl_sub
global global_cloud
global count
global_cloud = PointCloud2()

facing_left = False
facing_right = False
obstacle_map_right_drawn = False
obstacle_map_left_drawn = False


# Helper function to get surface normals
def get_normals(cloud):
	get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
	return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
	yaml_dict = {}
	yaml_dict["test_scene_num"] = test_scene_num.data
	yaml_dict["arm_name"]  = arm_name.data
	yaml_dict["object_name"] = object_name.data
	yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
	yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
	return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
	data_dict = {"object_list": dict_list}
	with open(yaml_filename, 'w') as outfile:
		yaml.dump(data_dict, outfile, default_flow_style=False)


def publish_to_yaml(labels,centroids):

	world_no = 1
	out_file_name = 'output_1.yaml'
	print("sending data to required file")
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
	
	send_to_yaml('output_1.yaml',dict_list)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
	global facing_left
	global facing_right
	global obstacle_map_left_drawn
	global obstacle_map_right_drawn
	print("invoked")
	global_cloud = pcl_msg
	
	""" TO draw the collision avoidance map of surrounding drop tables
	if(not facing_left and not obstacle_map_left_drawn):
		print("wait_for_40_seconds")
		pub_j1.publish(-1.57)
		time.sleep(40)
		facing_left = True
		return
	if(facing_left and not obstacle_map_left_drawn):
		print("will update collision left map here")
		obstacle_map_left_drawn = True
		pub_j1.publish(0)
		time.sleep(40)
		facing_left = False
		return
	if(not facing_right and not obstacle_map_right_drawn):
		print("wait_for_40_seconds")
		pub_j1.publish(1.57)
		time.sleep(40)
		facing_right = True
		return
	if(facing_right and not obstacle_map_right_drawn):
		print("will update collision right map here")
		obstacle_map_right_drawn = True
		pub_j1.publish(0)
		time.sleep(40)
		facing_right = False
		return
	"""

# Exercise-2 TODOs:
	print("invoked")
	time.sleep(60)
	# TODO: Convert ROS msg to PCL data
	cloud = ros_to_pcl(pcl_msg)
	# TODO: Statistical Outlier Filtering

	outlier_filter = cloud.make_statistical_outlier_filter()
	outlier_filter.set_mean_k(15)
	x = 1
	outlier_filter.set_std_dev_mul_thresh(x)
	noise_filtered = outlier_filter.filter()

	vox = noise_filtered.make_voxel_grid_filter()
	LEAF_SIZE = 0.01
	vox.set_leaf_size(LEAF_SIZE,LEAF_SIZE,LEAF_SIZE)
	down_sampled = vox.filter()

	pass_thorugh_z = down_sampled.make_passthrough_filter()
	filter_axis = 'z'
	pass_thorugh_z.set_filter_field_name(filter_axis)
	axis_min = 0.61
	axis_max = 1.5
	pass_thorugh_z.set_filter_limits(axis_min,axis_max)
	cloud_filtered_z = pass_thorugh_z.filter()


	pass_thorugh_y = cloud_filtered_z.make_passthrough_filter()
	filter_axis = 'y'
	pass_thorugh_y.set_filter_field_name(filter_axis)
	axis_min = -0.5
	axis_max = 0.5
	pass_thorugh_y.set_filter_limits(axis_min,axis_max)
	cloud_filtered_y = pass_thorugh_y.filter()

	seg = cloud_filtered_y.make_segmenter()
	seg.set_model_type(pcl.SACMODEL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)
	max_distance = 0.001
	seg.set_distance_threshold(max_distance)
	inliers,coefficients = seg.segment()

	extracted_inliners = cloud_filtered_y.extract(inliers,negative=False)
	extracted_outliers = cloud_filtered_y.extract(inliers,negative=True)

	table_msg = extracted_inliners
	obj_msg = extracted_outliers

	obj_msg_ros = pcl_to_ros(obj_msg)
	table_msg_ros = pcl_to_ros(table_msg)
	pcl_objects_pub.publish(obj_msg_ros)
	print("published")

	white_cloud =  XYZRGB_to_XYZ(obj_msg)
	tree = white_cloud.make_kdtree()
	ec = white_cloud.make_EuclideanClusterExtraction()
	ec.set_ClusterTolerance(0.015)
	ec.set_MinClusterSize(100)
	ec.set_MaxClusterSize(3000)
	ec.set_SearchMethod(tree)
	cluster_indices = ec.Extract()


	print(len(cluster_indices))
	cluster_color = get_color_list(len(cluster_indices))
	color_cluster_point_list = []

	for j,indices in enumerate(cluster_indices):
			for i,indice in enumerate(indices):
				color_cluster_point_list.append([white_cloud[indice][0],
												white_cloud[indice][1],
												white_cloud[indice][2],
											rgb_to_float(cluster_color[j])])

	cluster_cloud = pcl.PointCloud_PointXYZRGB()
	cluster_cloud.from_list(color_cluster_point_list)
	ros_cluster_cloud = pcl_to_ros(cluster_cloud)
	pcl_clusters_pub.publish(ros_cluster_cloud)


	noise_filtered_ros = pcl_to_ros(noise_filtered)
	cloud_filtered_z_ros = pcl_to_ros(cloud_filtered_z)
	cloud_filtered_y_ros = pcl_to_ros(cloud_filtered_y)



	# TODO: Convert PCL data to ROS messages

	# TODO: Publish ROS messages

# Exercise-3 TODOs:
	map_pcl = {}
	detected_objects_labels = []
	detected_objects = []
	labels = []
	centroids = []
	for index, pts_list in enumerate(cluster_indices):
		
		pcl_cluster = obj_msg.extract(pts_list)
		pcl_cluster_ros = pcl_to_ros(pcl_cluster)
		chists = compute_color_histograms(pcl_cluster_ros,True)
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
		#object_markers_pub.publish(make_label(label,label_pos, index))

		do = DetectedObject()
		do.label = label
		do.cloud = pcl_cluster_ros
		detected_objects.append(do)

		map_pcl.update({label:pcl_cluster})

		#collide_objects_pub.publish(pcl_cluster_ros)
		labels.append(label)
		points_arr = ros_to_pcl(pcl_cluster_ros).to_array()
		center = np.mean(points_arr, axis=0)[:3]
		center_float = []
		for item in center:
			center_float.append(np.asscalar(item))
		centroids.append(center_float)
		

		# Add the detected object to the list of detected objects.
	rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
	

   	object_list_param = rospy.get_param('/object_list')
	pick_list_objects = []
	for i in range(len(object_list_param)):
		pick_list_objects.append(object_list_param[i]['name'])

	print "\n"  
	print "Pick List includes: "
	print pick_list_objects
	print "\n"
	pick_set_objects = set(pick_list_objects)
	detected_set_objects = set(detected_objects_labels)


	
	#if detected_set_objects <> pick_set_objects:
	try:
		pr2_mover(detected_objects, table_msg_ros,map_pcl)
	except rospy.ROSInterruptException:
		pass
	

	#publish_to_yaml(labels,centroids)
	# Suggested location for where to invoke your pr2_mover() function within pcl_callback()
	# Could add some logic to determine whether or not your object detections are robust
	# before calling pr2_mover()
	#try:
	#	#pr2_mover(detected_objects_list)
	#except rospy.ROSInterruptException:
	#	pass

# function to load parameters and request PickPlace service

# function to load parameters and request PickPlace service
def pr2_mover(object_list, table_msg_ros, map_pcl):

	test_scene_num = Int32()
	test_scene_num.data = 3
	outputFileName = "output_3.yaml"

	object_collison_cloud = []

	# Lets start by getting the pick list!
	# Retrieve the picklist from the parameter server (YAML files)
	object_list_param = rospy.get_param('/object_list')

	# Let's determine the location of the red and green boxes
	box_param = rospy.get_param('/dropbox')

	box_name = []
	box_group = []
	box_position = []
	# We'll loop through the two boxes
	for i in range(0, len(box_param)):
		box_name.append(box_param[i]['name'])
		box_group.append(box_param[i]['group'])
		box_position.append(box_param[i]['position'])



	picked_list = []
	dict_list = []
	j = 0
	for i in range(0, len(object_list_param)):
		
		labels = []
		centroids = [] # to be list of tuples (x, y, z)
		for object in object_list:
			labels.append(object.label)
			points_arr = ros_to_pcl(object.cloud).to_array()
			temp = np.mean(points_arr, axis=0)[:3]
			centroids.append(temp)
		object_name = String()
		# This is getting the first object name and box from the picklist

		object_name.data = object_list_param[i]['name']
		object_group = object_list_param[i]['group']

		arm_name = String()
		place_pose = Pose()

		# arm_name - Right for Green Box, Left for Red Box
		#-0.1, 0.71, 0.605
		if object_group == 'red':
			arm_name.data = 'left'
			place_pose.position.x = box_position[0][0] - 0.1
			place_pose.position.y = box_position[0][1] 
			place_pose.position.z = box_position[0][2] 
		else:
			arm_name.data = 'right'
			place_pose.position.x = box_position[1][0] - 0.1
			place_pose.position.y = box_position[1][1] 
			place_pose.position.z = box_position[1][2] 

		print box_position[0][0] ,box_position[0][1] ,box_position[0][2]
		print box_position[1][0] ,box_position[1][1] ,box_position[1][2]  
		# pick_pose
		pick_pose = Pose()
		desired_object = object_list_param[i]['name']
		print "\n\nPicking up ", desired_object, "\n\n"
		print "\n\nPutting in ", object_group, "\n\n"




		#collide_objects_pub.publish(table_msg_ros)

		temp_cloud = PointCloud2()

		rospy.wait_for_service('clear_octomap')
		clear_octomap = rospy.ServiceProxy('clear_octomap', std_srvs.srv.Empty)
		resp_clear = clear_octomap()
		print("cleared..?")
		time.sleep(5)

		for object in object_list:
			if object.label not in picked_list and (not(object.label == object_name.data)):
				temp_cloud = ros_to_pcl2(temp_cloud, pcl_to_ros(map_pcl[object.label]))
			else:
				print("skipped")
				print(object.label)
		collide_objects_pub.publish(temp_cloud)

		# Match desired object with the centroid list/labels
		try:
			labelPosition = labels.index(desired_object)
			pick_pose.position.x = np.asscalar(centroids[labelPosition][0])
			pick_pose.position.y = np.asscalar(centroids[labelPosition][1])
			pick_pose.position.z = np.asscalar(centroids[labelPosition][2])
		except ValueError:
			continue

		# Populate various ROS messages
		yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
		dict_list.append(yaml_dict)
		# Wait for 'pick_place_routine' service to come up
		#rospy.wait_for_service('pick_place_routine')
			
		try:
			pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
			# Insert your message variables to be sent as a service request
			resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
			print ("Response: ",resp.success)
			picked_list.append(object_name.data)	
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
		

	# Output your request parameters into output yaml file
	send_to_yaml(outputFileName, dict_list)



# function to load parameters and request PickPlace service

if __name__ == '__main__':


	model = pickle.load(open('model3.sav', 'rb'))
	clf = model['classifier']
	encoder = LabelEncoder()
	encoder.classes_ = model['classes']
	scaler = model['scaler']
	rospy.init_node("clustering", anonymous=True)
	collide_objects_pub = rospy.Publisher("/pr2/3d_map/points",PointCloud2,queue_size=1)
	pub_j1 = rospy.Publisher('/pr2/world_joint_controller/command',Float64, queue_size=10)
	pcl_sub = rospy.Subscriber("/pr2/world/points",PointCloud2,pcl_callback,queue_size=1)	
	pcl_objects_pub = rospy.Publisher("/pcl_objects",PointCloud2,queue_size=1)
	#pcl_objects_pub_filtered = rospy.Publisher("/pcl_objects_filtered",PointCloud2,queue_size=1)
	pcl_clusters_pub = rospy.Publisher("/pcl_clusters",PointCloud2,queue_size=1)
	object_markers_pub  = rospy.Publisher("/object_markers",Marker,queue_size=1)
	detected_objects_pub = rospy.Publisher("/detected_objects",DetectedObjectsArray,queue_size=1)

	while not rospy.is_shutdown():
		rospy.spin()	

	# TODO: Spin while node is not shutdown
