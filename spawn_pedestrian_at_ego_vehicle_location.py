import carla
import random


### Better running these steps in python interactive console from terminal step by step

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()
# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

# 1. spawn ego vehicle using carla ros bridge from terminal
# 2. Select the vehicle and get the transform of the ego_vehicle

### After finding the ego_vehicle id use world.get_actor(vehicle_id)
### vehicle.get_transform() to get the transform of the vehicle for spawning pedestrian

### Move the vehicle at a different location slightly
### Use the earlier transform value to spawn_pedestrian
ego_car = world.get_actor(198)
# Getting the transformation of lidar sensor might be more helpful instead of the car
ego_car_transf = ego_car.get_transform()

# rand_location = world.get_random_location_from_navigation()


# Get the pedestrian blueprint and spawn it
pedestrian_bp = random.choice(world.get_blueprint_library().filter('*walker.pedestrian*'))
# transform = carla.Transform(carla.Location(x=-134,y=78.1,z=1.18))
# transform = carla.Transform(rand_location)
pedestrian = world.try_spawn_actor(pedestrian_bp, ego_car_transf)

# changes the viewpoint to the person spawn point in carla
spectator.set_transform(ego_car_transf)