import carla
import random


# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()
# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

rand_location = world.get_random_location_from_navigation()


# Get the pedestrian blueprint and spawn it
pedestrian_bp = random.choice(world.get_blueprint_library().filter('*walker.pedestrian*'))
# transform = carla.Transform(carla.Location(x=-134,y=78.1,z=1.18))
transform = carla.Transform(rand_location)
pedestrian = world.try_spawn_actor(pedestrian_bp, transform)

# changes the viewpoint to the person spawn point in carla
spectator.set_transform(transform)