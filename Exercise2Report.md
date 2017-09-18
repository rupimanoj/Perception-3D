[//]: # (Image References)

[obj_msg]: ./misc_images/Objects.PNG
[table_msg]: ./misc_images/table.PNG
[cluster_msg]: ./misc_images/object_clusters.PNG


## Clustering Pipline

### Objects extraction from table top pointcloud

As Explained in [Exercise-1](https://github.com/rupimanoj/Perception-Exercises/blob/master/Exercise-1/report.md), using RANSAC Plane segmentation technique, objects and table base were seperated.

![alt text][obj_msg]
![alt text][table_msg]



### Elucidian clustering

By constructing kdtree from XYZ data of objects pointcloud and setting minimum and maximum points in a custer, object clustering is done as below.

``` python
white_cloud =  XYZRGB_to_XYZ(obj_msg)
tree = white_cloud.make_kdtree()
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.015)
ec.set_MinClusterSize(200)   
ec.set_MaxClusterSize(3000)	 
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()
```
After this `cluster_indices` stores list of elements where all the points belonging to one cluster will be grouped together in one list element. To visualize or get an idea, `cluster_indices` looks like below lisy <br/>

``` python
cluster_indices = [
			[points indices from white cloud belonging to cluster1],
			[points indices from white cloud belonging to cluster2],
			[points indices from white cloud belonging to cluster3].....]
```

Once cluster cloud points are grouped together as in the above `cluster_indices`, we traverse through each cloud indices and create `color_cluster_point_list` where each cluster group is marked with different color.  


``` python
# Create Cluster-Mask Point Cloud to visualize each cluster separately
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []

for j,indices in enumerate(cluster_indices):
		for i,indice in enumerate(indices):
			color_cluster_point_list.append([white_cloud[indice][0],white_cloud[indice][1],white_cloud[indice][2],
										rgb_to_float(cluster_color[j])])

cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)

#  Convert PCL data to ROS messages
obj_msg_ros = pcl_to_ros(obj_msg) 
table_msg_ros = pcl_to_ros(table_msg)
cluster_cloud_ros = pcl_to_ros(cluster_cloud)

# Publish ROS messages
pcl_clusters_pub.publish(cluster_cloud_ros)  #publish cluster cloud
pcl_objects_pub.publish(obj_msg_ros)		#publish segmented objects
pcl_table_pub.publish(table_msg_ros)		#publish table plane point cloud
```

Once clusters PointCloud data is created, we publish the data through `pcl_clusters_pub' publisher.

``` python
cluster_cloud_ros = pcl_to_ros(cluster_cloud)
pcl_table_pub.publish(table_msg_ros)
```
![alt text][cluster_msg]