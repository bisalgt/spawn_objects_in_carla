import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon
import pandas as pd

# from figures import SIZE, BLUE, GRAY, set_limits

fig = plt.figure(1, dpi=90)

groundtruth_shadow_points = pd.read_csv("../../../Desktop/ground_truth_shadow_region.csv", sep=",", header=None)
predicted_shadow_points = pd.read_csv("../../../Desktop/predicted_shadow_region.csv", sep=",", header=None)


groundtruth_shadow_polygon = Polygon(groundtruth_shadow_points.to_numpy()[:, :2])
predicted_shadow_polygon = Polygon(predicted_shadow_points.to_numpy()[:, :2])


a = groundtruth_shadow_polygon
b = predicted_shadow_polygon

# 1
ax = fig.add_subplot(121)

plot_polygon(a, ax=ax, add_points=False, color=[0,0,0], alpha=0.2)
# plot_polygon(b, ax=ax, add_points=False, color=[1,0,0], alpha=0.2)

# c = a.intersection(b)
# plot_polygon(c, ax=ax, add_points=False, color=[0,1,0], alpha=0.5)

ax.set_title('Ground Truth Shadow')

# set_limits(ax, -1, 4, -1, 3)

#2
ax = fig.add_subplot(122)

# plot_polygon(a, ax=ax, add_points=False, color=[0, 0, 0], alpha=0.2)
plot_polygon(b, ax=ax, add_points=False, color=[1,0,0], alpha=0.2)

# c = a.union(b)
# plot_polygon(c, ax=ax, add_points=False, color=[0,0,1], alpha=0.5)

ax.set_title('Predicted Shadow')

# set_limits(ax, -1, 4, -1, 3)

plt.show()