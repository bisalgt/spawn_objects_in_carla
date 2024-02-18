import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d


original_source_file = "../../interactive_pcd_lib/feb18/source_cloud.ply"
raycasted_source_file = "../../interactive_pcd_lib/feb18/raycasted_source_cloud.ply"


pcd_source = o3d.io.read_point_cloud(original_source_file)  # Replace with your actual point cloud

# Convert the point cloud to a numpy array
points_source = np.asarray(pcd_source.points)

# Separate the x, y, and z coordinates
x_source = points_source[:, 0]
y_source = points_source[:, 1]
z_source = points_source[:, 2]



# Raycasted
pcd_raycasted = o3d.io.read_point_cloud(raycasted_source_file)  # Replace with your actual point cloud

# Convert the point cloud to a numpy array
points_raycasted = np.asarray(pcd_raycasted.points)

# Separate the x, y, and z coordinates
x_raycasted = points_raycasted[:, 0]
y_raycasted = points_raycasted[:, 1]
z_raycasted = points_raycasted[:, 2]



# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud
ax.scatter(x_source, y_source, z_source, color='black')
ax.scatter(x_raycasted, y_raycasted, z_raycasted, color="red",s=6)
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")

plt.show()