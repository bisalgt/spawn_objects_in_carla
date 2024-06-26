import carla
import random


client = carla.Client('localhost', 2000)
world = client.get_world()

spectator = world.get_spectator()

sensor_actor_id = 197

# sensor_bp = world.get_blueprint_library().filter("sensor.lidar.ray_cast")[0]

# actor_sensor = world.get_actors().filter("sensor.lidar.*")

# print(actor_sensor)

lidar_sensor = world.get_actor(sensor_actor_id)

lidar_sensor_location = lidar_sensor.get_location()

shift_xyz = [2, -1, 0]

new_location_array = lidar_sensor_location.x + shift_xyz[0], lidar_sensor_location.y + shift_xyz[1] , lidar_sensor_location.z + shift_xyz[2]

new_location = carla.Location(*new_location_array)

rotation_by = (0, 180, 0)

new_rotation = carla.Rotation(*rotation_by)


new_transf = carla.Transform(new_location, new_rotation)


# Get the pedestrian blueprint and spawn it
pedestrian_bp = random.choice(world.get_blueprint_library().filter('*walker.pedestrian*'))
# transform = carla.Transform(carla.Location(x=-134,y=78.1,z=1.18))
# transform = carla.Transform(rand_location)
pedestrian = world.try_spawn_actor(pedestrian_bp, new_transf)

# changes the viewpoint to the person spawn point in carla
spectator.set_transform(new_transf)

rotations = [i*20 for i in range(10)]
rotation_by = (0, random.choice(rotations), 0)



# pedestrian.destroy()
# shift_xyz = [15, 2, 0]
# new_location_array = lidar_sensor_location.x + shift_xyz[0], lidar_sensor_location.y + shift_xyz[1] , lidar_sensor_location.z + shift_xyz[2]
# new_location = carla.Location(*new_location_array)
# rotation_by = (0, random.choice(rotations), 0)
# new_rotation = carla.Rotation(*rotation_by)
# new_transf = carla.Transform(new_location, new_rotation)
# pedestrian_bp = random.choice(world.get_blueprint_library().filter('*walker.pedestrian*'))
# pedestrian = world.try_spawn_actor(pedestrian_bp, new_transf)

person.destroy()
shift_xyz = [5,5,0]
new_location_array = car_location.x + shift_xyz[0], car_location.y + shift_xyz[1], car_location.z + shift_xyz[2]
new_location = carla.Location(*new_location_array)
rotation_by = (0, 180, 0)
new_rotation = carla.Rotation(*rotation_by)
new_transform = carla.Transform(new_location, new_rotation)
person = world.try_spawn_actor(person_bp, new_transform)



# shift_xyz = [7,4,0]
# new_location_array = car_location.x + shift_xyz[0], car_location.y + shift_xyz[1], car_location.z + shift_xyz[2]
# new_location = carla.Location(*new_location_array)
# rotation_by = (0, 180, 0)
# new_rotation = carla.Rotation(*rotation_by)
# new_transform = carla.Transform(new_location, new_rotation)
# person = world.try_spawn_actor(person_bp, new_transform)

pedestrian.destroy()