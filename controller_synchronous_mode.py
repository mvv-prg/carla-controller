#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import csv
import glob
import math
import os
import sys
import argparse

# Append carla egg to sys.path
sys.path.append(
    glob.glob(
        './PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        )
    )[0]
)
import carla
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pyautogui
from matplotlib.patches import Arrow
from screeninfo import get_monitors

# Python 2/3 compatibility for queue
try:
    import queue
except ImportError:
    import Queue as queue





"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 2.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 200.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 0      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer

PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)
PLAYER_VEHICLE_BLUEPRINT_INDEX = 13 # index for player vehicle (keep to 13 (Mini))
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends
                                       
# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

VIDEO_NAME = 'simulation_video_MPC.avi'  # video name


def parse_arguments():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Choose a controller.")
    parser.add_argument('--controller', type=str, choices=['MPC', 'Stanley'], required=True, help="Choose the type of controller: MPC or Stanley.")
    return parser.parse_args()


def pad_image_to_match_height(image, target_height):
    """
    Pad the image with black border to match the target height.
    """
    pad_top = (target_height - image.shape[0]) // 2
    pad_bottom = target_height - image.shape[0] - pad_top
    return cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def capture_screenshots(idx=1, image_folder='screenshots', monitor_id=0):
    """
    Capture a single screenshot and save it to the specified folder.

    Parameters:
    - idx: The index of the screenshot.
    - image_folder: Folder to save the screenshots.
    """
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    all_screenshots = []
    max_height = max([monitor.height for monitor in get_monitors()])

    for i, monitor in enumerate(get_monitors()):
        if i != monitor_id:
            continue
        screenshot = pyautogui.screenshot(region=(
            monitor.x, monitor.y, monitor.width, monitor.height))
        screenshot_np = np.array(screenshot)

        # if screenshot_np.shape[0] < max_height:
        #     screenshot_np = pad_image_to_match_height(screenshot_np, max_height)

        all_screenshots.append(screenshot_np)

    # Concatenate screenshots horizontally
    combined_screenshot = np.hstack(all_screenshots)
    cv2.imwrite(os.path.join(image_folder, f'screenshot_{idx}.png'), cv2.cvtColor(combined_screenshot, cv2.COLOR_RGB2BGR))

def images_to_video(video_name, image_folder='screenshots', fps=30):
    """
    Convert series of images in a folder to a video.

    Parameters:
    - image_folder: Path to the folder containing images.
    - video_name: Name of the output video file (e.g., "output.avi").
    - fps: Frames per second for the output video.
    """
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure images are in the correct order

    # Get the dimensions of the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, layers = frame.shape
    size = (w, h)

    # Create a video writer object
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        out.write(img)

    out.release()
    

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def send_control_command(vehicle, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client.
    
    Send control command to CARLA client.

    Args:
        client: The CARLA client object
        throttle: Throttle command for the sim car [0, 1]
        steer: Steer command for the sim car [-1, 1]
        brake: Brake command for the sim car [0, 1]
        hand_brake: Whether the hand brake is engaged
        reverse: Whether the sim car is in the reverse gear
    """
    if isinstance(vehicle, carla.Vehicle):
        control = carla.VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    vehicle.apply_control(control)


def get_current_pose(world_snapshot, player_id):
    """
    Get the current pose (x, y, yaw) of the player's vehicle from the world snapshot.

    Args:
    - world_snapshot: The world snapshot returned by wait_for_tick().
    - player_id: The ID of the player's vehicle.

    Returns:
    - x, y, yaw: The x and y coordinates and the yaw angle of the player's vehicle.
    """
    # Get the actor's snapshot using the player_id
    actor_snapshot = world_snapshot.find(player_id)

    # Extract the location and rotation
    location = actor_snapshot.get_transform().location
    rotation = actor_snapshot.get_transform().rotation

    x = location.x
    y = location.y
    yaw = rotation.yaw

    return x, y, yaw



def main():
    args = parse_arguments()

    if args.controller == 'MPC':
        from controller2d_MPC import Controller2D
    elif args.controller == 'Stanley':
        from controller2d_Stanley import Controller2D

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(60.0)

    
    plt.ion()  # enable interactive drawing


    world = client.get_world()
    map_name = world.get_map().name
    world = client.load_world(map_name)
    print('Starting new episode at %r...' % map_name)

    try:
        carla_map = world.get_map()
        
        #############################################
        # Load Waypoints
        #############################################
        # Opens the waypoint file and stores it to "waypoints"
        waypoints_np   = None
        with open(WAYPOINTS_FILENAME) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle, 
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)


        # Because the waypoints are discrete and our controller performs better
        # with a continuous path, here we will send a subset of the waypoints
        # within some lookahead distance from the closest point to the vehicle.
        # Interpolating between each waypoint will provide a finer resolution
        # path and make it more "continuous". A simple linear interpolation
        # is used as a preliminary method to address this issue, though it is
        # better addressed with better interpolation methods (spline 
        # interpolation, for example). 
        # More appropriate interpolation methods will not be used here for the
        # sake of demonstration on what effects discrete paths can have on
        # the controller. It is made much more obvious with linear
        # interpolation, because in a way part of the path will be continuous
        # while the discontinuous parts (which happens at the waypoints) will 
        # show just what sort of effects these points have on the controller.
        # Can you spot these during the simulation? If so, how can you further
        # reduce these effects?
        
        # Linear interpolation computations
        # Compute a list of distances between waypoints
        wp_distance = []   # distance array
        for i in range(1, waypoints_np.shape[0]):
            wp_distance.append(
                    np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                            (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
                               # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list
        wp_interp      = []    # interpolated values 
                               # (rows = waypoints, columns = [x, y, v])
        wp_interp_hash = []    # hash table which indexes waypoints_np
                               # to the index of the waypoint in wp_interp
        interp_counter = 0     # counter for current interpolated point index
        for i in range(waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(waypoints_np[i]))
            wp_interp_hash.append(interp_counter)   
            interp_counter+=1
            
            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                         float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter+=1
        # add last waypoint at the end
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)   
        interp_counter+=1

        #############################################
        # Controller 2D Class Declaration
        #############################################
        # This is where we take the controller2d.py class
        # and apply it to the simulator
        controller = Controller2D(waypoints)
        

        #############################################
        # Create Vehicle and Camera
        #############################################
        
        spawn_points = carla_map.get_spawn_points()
        start_pose = spawn_points[PLAYER_START_INDEX]
        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(
            # blueprint_library.filter('vehicle.*')[13], 
            blueprint_library.filter('vehicle.*')[PLAYER_VEHICLE_BLUEPRINT_INDEX], 
            start_pose)
        actor_list.append(vehicle)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        for waypoint in waypoints:
            world.debug.draw_point(carla.Location(waypoint[0], waypoint[1]), size=0.1, color=carla.Color(255,0,0), life_time=120)


        #############################################
        # SYNC MODE START
        ############################################# 
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            #############################################
            # Determine simulation average timestep (and total frames)
            #############################################
            # Ensure at least one frame is used to compute average timestep
            num_iterations = ITER_FOR_SIM_TIMESTEP
            if (ITER_FOR_SIM_TIMESTEP < 1):
                num_iterations = 1

            # Gather current data from the CARLA server. This is used to get the
            # simulator starting game time. Note that we also need to
            # send a command back to the CARLA server because synchronous mode
            # is enabled.
            snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
            # Gather current data from the CARLA server
            measurement_data = snapshot
            sim_start_stamp = measurement_data.timestamp.elapsed_seconds 
            # Send a control command to proceed to next iteration.
            # This mainly applies for simulations that are in synchronous mode.
            send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
            # Computes the average timestep based on several initial iterations
            sim_duration = 0
            for i in range(num_iterations):
                # Gather current data
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                # Gather current data from the CARLA server
                measurement_data = snapshot
                # Send a control command to proceed to next iteration
                send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
                # Last stamp
                if i == num_iterations - 1:
                    sim_duration = measurement_data.timestamp.elapsed_seconds  -\
                                   sim_start_stamp  

            # Outputs average simulation timestep and computes how many frames
            # will elapse before the simulation should end based on various
            # parameters that we set in the beginning.
            SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
            print("SERVER SIMULATION STEP APPROXIMATION: " + \
                  str(SIMULATION_TIME_STEP))
            TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
                                   SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

            #############################################
            # Frame-by-Frame Iteration and Initialization
            #############################################
            # Store pose history starting from the start position
            snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
            # Gather current data from the CARLA server
            measurement_data = snapshot
            start_x, start_y, start_yaw = get_current_pose(measurement_data,  vehicle.id)
            send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
            x_history     = [start_x]
            y_history     = [start_y]
            yaw_history   = [start_yaw]
            time_history  = [0]
            speed_history = [0]
            #############################################
            # Vehicle Trajectory Live Plotting Setup
            #############################################
            fig_traj, ax_traj = plt.subplots(figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES))
            plt.title("Vehicle Trajectory")
            
            # Add waypoint markers
            ax_traj.plot(waypoints_np[:, 0], waypoints_np[:, 1], '-g', label='waypoints')
            ax_traj.plot([start_x]*TOTAL_EPISODE_FRAMES, [start_y]*TOTAL_EPISODE_FRAMES, color=[1, 0.5, 0], label='trajectory')
            ax_traj.plot([start_x]*INTERP_MAX_POINTS_PLOT, [start_y]*INTERP_MAX_POINTS_PLOT, color=[0, 0.7, 0.7], linewidth=4, label='lookahead_path')
            ax_traj.scatter([start_x], [start_y], marker='o', color=[1, 0.5, 0], label='Start')
            ax_traj.scatter([waypoints_np[-1, 0]], [waypoints_np[-1, 1]], marker='D', color='r', label='End')
            car_marker, = ax_traj.plot([start_x], [start_y], 's', color='b', label='Car') 
            
            dx_heading = math.cos(math.radians(start_yaw))
            dy_heading = math.sin(math.radians(start_yaw))
            dx_steer = math.cos(math.radians(start_yaw))
            dy_steer = math.sin(math.radians(start_yaw))
            arrow_length = 10.0
            ax_traj._heading_arrow = ax_traj.add_patch(Arrow(start_x, start_y, dx_heading * arrow_length, dy_heading * arrow_length, color='b'))
            ax_traj._steer_arrow = ax_traj.add_patch(Arrow(start_x, start_y, dx_steer * arrow_length, dy_steer * arrow_length, width=0.2, color='r'))


            ax_traj.legend()
            ax_traj.grid(True)
            fig_aux, (ax_speed, ax_throttle, ax_brake, ax_steer) = plt.subplots(4, 1, figsize=(10, 15))

            # Graph for speed
            ax_speed.set_title('Forward Speed (m/s)')
            ax_speed.plot([0], [0], label='forward_speed')
            ax_speed.plot([0], [0], label='reference_signal')
            ax_speed.legend()
            ax_speed.grid(True)
            # Graph for throttle
            ax_throttle.set_title('Throttle')
            ax_throttle.plot([0], [0], label='throttle')
            ax_throttle.legend()
            ax_throttle.grid(True)
            # Graph for brake
            ax_brake.set_title('Brake')
            ax_brake.plot([0], [0], label='brake')
            ax_brake.legend()
            ax_brake.grid(True)
            # Graph for steer
            ax_steer.set_title('Steer')
            ax_steer.plot([0], [0], label='steer')
            ax_steer.legend()
            ax_steer.grid(True)

            ax_throttle.set_ylim(0, 1)
            ax_brake.set_ylim(0, 1)
            ax_steer.set_ylim(-1, 1)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

            # Iterate the frames until the end of the waypoints is reached or
            # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
            # ouptuts the results to the controller output directory.
            reached_the_end = False
            skip_first_frame = True
            controller_is_updated = False
            closest_index    = 0  # Index of waypoint that is currently closest to
                                  # the car (assumed to be the first index)
            closest_distance = 0  # Closest distance of closest waypoint to car
            for frame in range(TOTAL_EPISODE_FRAMES):
                if should_quit():
                    return
                clock.tick()
                plt.pause(0.001)

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)

                 # Gather current data from the CARLA server
                measurement_data = snapshot
                # Get the velocity of the player's vehicle
                velocity = vehicle.get_velocity()
                current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                # Update pose, timestamp
                current_x, current_y, current_yaw = get_current_pose(measurement_data, vehicle.id)
                current_timestamp = measurement_data.timestamp.elapsed_seconds 
                print(current_timestamp)
                if current_timestamp <= WAIT_TIME_BEFORE_START:
                    send_control_command(vehicle, throttle=0.0, steer=0, brake=1.0)
                    continue
                else:
                    current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START

                # Store history
                x_history.append(current_x)
                y_history.append(current_y)
                yaw_history.append(current_yaw)
                speed_history.append(current_speed)
                time_history.append(current_timestamp) 

                ###
                # Setup sprectator position
                ###
                spectator = world.get_spectator()
                new_location = carla.Location(x=current_x, y=current_y, z=20)
                new_rotation = carla.Rotation(pitch=-90, yaw=current_yaw, roll=0)
                spectator.set_transform(carla.Transform(new_location, new_rotation))
                ###
                # Controller update (this uses the controller2d.py implementation)
                ###

                # To reduce the amount of waypoints sent to the controller,
                # provide a subset of waypoints that are within some 
                # lookahead distance from the closest point to the car. Provide
                # a set of waypoints behind the car as well.

                # Find closest waypoint index to car. First increment the index
                # from the previous index until the new distance calculations
                # are increasing. Apply the same rule decrementing the index.
                # The final index should be the closest point (it is assumed that
                # the car will always break out of instability points where there
                # are two indices with the same minimum distance, as in the
                # center of a circle)
                closest_distance = np.linalg.norm(np.array([
                        waypoints_np[closest_index, 0] - current_x,
                        waypoints_np[closest_index, 1] - current_y]))
                new_distance = closest_distance
                new_index = closest_index
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_index = new_index
                    new_index += 1
                    if new_index >= waypoints_np.shape[0]:  # End of path
                        break
                    new_distance = np.linalg.norm(np.array([
                            waypoints_np[new_index, 0] - current_x,
                            waypoints_np[new_index, 1] - current_y]))
                new_distance = closest_distance
                new_index = closest_index
                while new_distance <= closest_distance:
                    closest_distance = new_distance
                    closest_index = new_index
                    new_index -= 1
                    if new_index < 0:  # Beginning of path
                        break
                    new_distance = np.linalg.norm(np.array([
                            waypoints_np[new_index, 0] - current_x,
                            waypoints_np[new_index, 1] - current_y]))

                world.debug.draw_point(carla.Location(x=waypoints_np[new_index, 0], y=waypoints_np[new_index, 1]), size=0.3, color=carla.Color(255,255,0), life_time=1)
                # Once the closest index is found, return the path that has 1
                # waypoint behind and X waypoints ahead, where X is the index
                # that has a lookahead distance specified by 
                # INTERP_LOOKAHEAD_DISTANCE
                waypoint_subset_first_index = closest_index - 1
                if waypoint_subset_first_index < 0:
                    waypoint_subset_first_index = 0

                waypoint_subset_last_index = closest_index
                total_distance_ahead = 0
                while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                    total_distance_ahead += wp_distance[waypoint_subset_last_index]
                    waypoint_subset_last_index += 1
                    if waypoint_subset_last_index >= waypoints_np.shape[0]:
                        waypoint_subset_last_index = waypoints_np.shape[0] - 1
                        break

                # Use the first and last waypoint subset indices into the hash
                # table to obtain the first and last indicies for the interpolated
                # list. Update the interpolated waypoints to the controller
                # for the next controller update.
                new_waypoints = \
                        wp_interp[wp_interp_hash[waypoint_subset_first_index]:\
                                  wp_interp_hash[waypoint_subset_last_index] + 1]
                controller.update_waypoints(new_waypoints)

                # Update the other controller values and controls
                controller.update_values(current_x, current_y, math.radians(current_yaw), 
                                         current_speed,
                                         current_timestamp, frame)
                if controller_is_updated:
                    controller.update_controls()
                    cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
                    if skip_first_frame and frame == 0:
                        pass
                    else:
                        # update the graph for trajectory
                        ax_traj.lines[1].set_xdata(np.append(ax_traj.lines[1].get_xdata(), current_x))
                        ax_traj.lines[1].set_ydata(np.append(ax_traj.lines[1].get_ydata(), current_y))
                        
                        # update car marker
                        car_marker.set_data(current_x, current_y)
                        
                                    
                        # update the graph for lookahead path
                        new_waypoints_np = np.array(new_waypoints)
                        path_indices = np.floor(np.linspace(0, new_waypoints_np.shape[0]-1, INTERP_MAX_POINTS_PLOT))
                        ax_traj.lines[2].set_xdata(new_waypoints_np[path_indices.astype(int), 0])
                        ax_traj.lines[2].set_ydata(new_waypoints_np[path_indices.astype(int), 1])

                        dx_heading = math.cos(math.radians(current_yaw))
                        dy_heading = math.sin(math.radians(current_yaw))

                        # Animation of heading and steer angles
                        dx_steer = math.cos(math.radians(current_yaw) + cmd_steer/180.0*70.0*np.pi)
                        dy_steer = math.sin(math.radians(current_yaw) + cmd_steer/180.0*70.0*np.pi)
                        if hasattr(ax_traj, '_heading_arrow'):
                            ax_traj._heading_arrow.remove()
                        if hasattr(ax_traj, '_steer_arrow'):
                            ax_traj._steer_arrow.remove()
                        ax_traj._heading_arrow = ax_traj.add_patch(Arrow(current_x, current_y, dx_heading * arrow_length, dy_heading * arrow_length, color='blue'))
                        ax_traj._steer_arrow = ax_traj.add_patch(Arrow(current_x, current_y, dx_steer * arrow_length, dy_steer * arrow_length, color='red'))


                        # Обновление графиков скорости, газа, тормоза и руля
                        ax_speed.lines[0].set_xdata(np.append(ax_speed.lines[0].get_xdata(), current_timestamp))
                        ax_speed.lines[0].set_ydata(np.append(ax_speed.lines[0].get_ydata(), current_speed))

                        ax_speed.lines[1].set_xdata(np.append(ax_speed.lines[1].get_xdata(), current_timestamp))
                        ax_speed.lines[1].set_ydata(np.append(ax_speed.lines[1].get_ydata(), controller._desired_speed))

                        ax_speed.set_xlim(min(ax_speed.lines[0].get_xdata()), max(ax_speed.lines[0].get_xdata()))
                        ax_speed.set_ylim(np.min(np.concatenate([ax_speed.lines[0].get_ydata(), ax_speed.lines[1].get_ydata()])),
                                          np.max(np.concatenate([ax_speed.lines[0].get_ydata(), ax_speed.lines[1].get_ydata()]))) 
                        ax_throttle.lines[0].set_xdata(np.append(ax_throttle.lines[0].get_xdata(), current_timestamp))
                        ax_throttle.lines[0].set_ydata(np.append(ax_throttle.lines[0].get_ydata(), cmd_throttle))

                        ax_brake.lines[0].set_xdata(np.append(ax_brake.lines[0].get_xdata(), current_timestamp))
                        ax_brake.lines[0].set_ydata(np.append(ax_brake.lines[0].get_ydata(), cmd_brake))

                        ax_steer.lines[0].set_xdata(np.append(ax_steer.lines[0].get_xdata(), current_timestamp))
                        ax_steer.lines[0].set_ydata(np.append(ax_steer.lines[0].get_ydata(), cmd_steer))
                        for ax in [ax_throttle, ax_brake, ax_steer]:
                            ax.set_xlim(min(ax.lines[0].get_xdata()), max(ax.lines[0].get_xdata()))
                        # Обновление графиков
                        fig_traj.canvas.flush_events()
                        fig_aux.canvas.flush_events()
                        capture_screenshots(frame)
                    # Output controller command to CARLA server
                    send_control_command(vehicle,
                                         throttle=cmd_throttle,
                                         steer=cmd_steer,
                                         brake=cmd_brake)
                    # Find if reached the end of waypoint. If the car is within
                    # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
                    # the simulation will end.
                    dist_to_last_waypoint = np.linalg.norm(np.array([
                        waypoints[-1][0] - current_x,
                        waypoints[-1][1] - current_y]))
                    if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                        reached_the_end = True
                    if reached_the_end:
                        break
                else:
                    controller_is_updated = True
                    cmd_throttle, cmd_steer, cmd_brake = 0, 0, 1



                #Plot frame in pygame
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                # draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                #blit current_x, current_y, current_yaw and current_speed
                display.blit(
                    font.render('x: ' + str(round(current_x, 3)), True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('y: ' + str(round(current_y, 3)), True, (255, 255, 255)),
                    (8, 64))
                display.blit(
                    font.render('yaw: ' + str(round(current_yaw, 3)), True, (255, 255, 255)),
                    (8, 82))
                display.blit(
                    font.render('speed: ' + str(round(current_speed, 3)), True, (255, 255, 255)),
                    (8, 100))
                display.blit(
                    font.render('throttle: ' + str(round(cmd_throttle, 3)), True, (255, 255, 255)),
                    (8, 118))
                display.blit(
                    font.render('steer: ' + str(round(cmd_steer, 3)), True, (255, 255, 255)),
                    (8, 136))
                display.blit(
                    font.render('brake: ' + str(round(cmd_brake, 3)), True, (255, 255, 255)),
                    (8, 154))
                display.blit(
                    font.render('frame: ' + str(frame), True, (255, 255, 255)),
                    (8, 172))
                display.blit(
                    font.render('closest_index: ' + str(closest_index), True, (255, 255, 255)),
                    (8, 190))
                display.blit(
                    font.render('closest_distance: ' + str(round(closest_distance, 3)), True, (255, 255, 255)),
                    (8, 208))

                pygame.display.flip()
            # End of demo - Stop vehicle and Store outputs to the controller output
            # directory.
            if reached_the_end:
                print("Reached the end of path. Writing to controller_output...")
            else:
                print("Exceeded assessment time. Writing to controller_output...")
            # Stop the car
            send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
    except Exception as e:
        print(e)
    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        plt.ioff() # disable interactive display
        plt.show() # keep windows open at the end of the script
        #images_to_video(VIDEO_NAME)
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
