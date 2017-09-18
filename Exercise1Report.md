# Tabletop Segmentation
[//]: # (Image References)

[table_image]: ./misc_images/inliners_pcd.PNG
[pass_through_image]: ./misc_images/pass_through.PNG
[objects_image]: ./misc_images/objects_pcd.PNG
[VOX_image]: ./misc_images/vox.PNG
[table_top]: ./misc_images/table_top.PNG

## RANSAC Pipeline steps:

Initial table top scene point cloud.
![alt text][VOX_image]

### Down sampling:

With leaf size of 0.01 block (1cm), downsampling was done after converting PointCloud2 data to pcl datatype.

![alt text][VOX_image]

``` python
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.01
vox.set_leaf_size(LEAF_SIZE,LEAF_SIZE,LEAF_SIZE)
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd' #to store downsampled point cloud
pcl.save(cloud_filtered,filename)

```

#### Pass through filter:

Pass though filtering was done along Z-direction to remove table base. axis_min and axis_max distances were obtained by looking the model in Gazebo simulator. axis_min was fine tuned using trial and error method to completely remove table edge.

![alt text][pass_through_image]

``` python
pass_thorugh = cloud_filtered.make_passthrough_filter()
filter_axis = 'z'
pass_thorugh.set_filter_field_name(filter_axis)
axis_min = 0.774	#achieved this value by trial and error methid..SLight change in this value will also give table edge 					#as part of objects..which are treated as outliers..Hence fine tuning is required here.
axis_max = 1.5
pass_thorugh.set_filter_limits(axis_min,axis_max)
cloud_filtered = pass_thorugh.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered,filename)

```

### RANSAC Plane segmentation:

To extract the table base, plane segmentation technique was used. AS we want to extract plsne, model type is set to `SACMODEL_PLANE`.
`max_distance` => Max distance for a point to be considered as part of Plane model.

``` python
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.001
seg.set_distance_threshold(max_distance)
inliers,coefficients = seg.segment()
```

By extracting outliers from the RANSAC model, we get interested objects point cloud. Inliers of model will be table base point cloud as shown in the figure.

![alt text][table_image]

``` python
extracted_inliners = cloud_filtered.extract(inliers,negative=False)
filename = "extracted_inliners.pcd"  #to store table top plane
pcl.save(extracted_inliners,filename)
```

![alt text][objects_image]

``` python
extracted_outliers = cloud_filtered.extract(inliers,negative=True)
filename = 'extracted_outliers.pcd'	#to store objects other than table plane
pcl.save(extracted_outliers,filename)
```