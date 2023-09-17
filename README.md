# CARLA Controller in Synchronous Mode

This script provides a controller in synchronous mode for the CARLA simulator.

## Prerequisites

- Python 3.x
- CARLA simulator installed and set up
- Required Python libraries: `csv`, `glob`, `math`, `os`, `sys`, `argparse`, `carla`, `controller2d`, `cv2`, `matplotlib`, `numpy`, `pygame`, `pyautogui`, `screeninfo`

## Configuration

The script allows for various configuration parameters such as:

- `ITER_FOR_SIM_TIMESTEP`: Number of iterations to compute approx sim timestep.
- `WAIT_TIME_BEFORE_START`: Game seconds (time before controller start).
- `TOTAL_RUN_TIME`: Game seconds (total runtime before sim end).
- `TOTAL_FRAME_BUFFER`: Number of frames to buffer after total runtime.
- `NUM_PEDESTRIANS`: Total number of pedestrians to spawn.
- `NUM_VEHICLES`: Total number of vehicles to spawn.
- `SEED_PEDESTRIANS`: Seed for pedestrian spawn randomizer.
- `SEED_VEHICLES`: Seed for vehicle spawn randomizer.
- `PLAYER_START_INDEX`: Spawn index for player.
- `PLAYER_VEHICLE_BLUEPRINT_INDEX`: Index for player vehicle.
- `WAYPOINTS_FILENAME`: Waypoint file to load.
- `DIST_THRESHOLD_TO_LAST_WAYPOINT`: Some distance from last position before simulation ends.
- `INTERP_MAX_POINTS_PLOT`: Number of points used for displaying lookahead path.
- `INTERP_LOOKAHEAD_DISTANCE`: Lookahead in meters.
- `INTERP_DISTANCE_RES`: Distance between interpolated points.
- `VIDEO_NAME`: Video name.

## Usage

To use the script, you can run it with the desired controller type:

