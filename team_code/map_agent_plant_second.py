"""
Agent file that runs the evaluations for map track models.
The predicted path in TF++ gets replaced by the centerline from the HD-MAP.
Run it by giving it as the agent option to the
leaderboard/leaderboard/leaderboard_evaluator.py file
"""

import os
from copy import deepcopy

import cv2
import carla
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from model import LidarCenterNet
from config import GlobalConfig
from data import CARLA_Data
from nav_planner import RoutePlanner
from nav_planner import extrapolate_waypoint_route

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

from scenario_logger import ScenarioLogger
import transfuser_utils as t_u

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import xml.etree.ElementTree as ET

import pathlib
import pickle
import ujson  # Like json but faster
import gzip


# New imports and constants:
from plant import PlanT
import sys


OPENPCDET = 1 # TODO: can be removed when cleaning up
USE_PERC_PLANT = 1 # TODO: can be removed when cleaning up

if OPENPCDET:
  from OpenPCDet.transform_data_new_dataset_intensity import read_bounding_box, transform_bb_to_unified, \
    PointCloud, transform_pc_to_unified
  from OpenPCDet.pcdet.models import load_data_to_gpu
  # from OpenPCDet.transform_data_new_dataset_intensity import read_bbs_from_results_frame
  
  from pathlib import Path as path_l
  from OpenPCDet.pcdet.utils import common_utils
  from OpenPCDet.pcdet.models import build_network
  from OpenPCDet.pcdet.datasets import build_dataloader
  from OpenPCDet.tools.test import parse_config as pcd_parser
  from munkres import Munkres
  from shapely.geometry import Polygon

# PCD_PATH = "/home/luis/Desktop/HIWI/carla_garage"
# PCD_PATH = "/mnt/qb/work/geiger/gwb710/carla_garage"
# PCD_CFG_PATH = f"{PCD_PATH}/pretrained_models/SECOND/config/second_new.yaml"
# PCD_CKPT_PATH = f"{PCD_PATH}/pretrained_models/SECOND/LR_0.003/WEIGHT_DECAY_0.001/GRAD_NORM_CLIP_10/ckpt/checkpoint_epoch_80.pth"

# PATH_TO_PLANNING_FILE = f"{PCD_PATH}/pretrained_models/longest6/plant_all_1"
# Cloud - downloaded
# PATH_TO_PLANNING_FILE = f"{PCD_PATH}/pretrained_models/longest6/plant_all_1"
# Cloud - LAV
# PATH_TO_PLANNING_FILE = f"/mnt/qb/work/geiger/gwb710/carla_garage/training_logdir/plant_v08_only_vehicle_lav"
# Cloud - L6
# PATH_TO_PLANNING_FILE = f"/mnt/qb/work/geiger/gwb710/carla_garage/training_logdir/plant_v08_only_vehicle"

PCD_CFG_PATH = os.environ.get('PCD_CFG_PATH',"") # f"/mnt/qb/work/geiger/gwb710/OpenPCDet/tools/cfgs/custom_models/second_new.yaml")
PCD_CKPT_PATH = os.environ.get('PCD_CKPT_PATH',"") # f"/mnt/qb/work/geiger/gwb710/OpenPCDet/output/custom_models/second_new/TMP_TEST_subsampled_data/LR_0.003/WEIGHT_DECAY_0.001/GRAD_NORM_CLIP_10/ckpt/checkpoint_epoch_80.pth")
PATH_TO_PLANNING_FILE = os.environ.get('PATH_TO_PLANNING_FILE', "")


DET_TH = float(os.environ.get('PCD_DETECTION_THRESHOLD',  0.4))
PCD_DETECTION_THRESHOLD = float(os.environ.get('PCD_DETECTION_THRESHOLD',  0.2)) 
ONLY_VEHICLE_BB = int(os.environ.get('ONLY_VEHICLE_BB', 0)) == 1
TRACKING = int(os.environ.get('TRACKING', 1)) == 1
HALF_EXTENTS = int(os.environ.get('HALF_EXTENTS', 1)) == 1

# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'MapAgent'


def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0, max_len=100):
  """
  Given some raw keypoints interpolate a full dense trajectory to be used by the user.
  returns the full interpolated route both in GPS coordinates and also in its original form.
  Args:
      - world: an reference to the CARLA world so we can use the planner
      - waypoints_trajectory: the current coarse trajectory
      - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
  """

  dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
  grp = GlobalRoutePlanner(dao)
  grp.setup()
  # Obtain route plan
  route = []
  for i in range(len(waypoints_trajectory) - 1):  # Goes until the one before the last.
    waypoint = waypoints_trajectory[i]
    waypoint_next = waypoints_trajectory[i + 1]
    if waypoint.x != waypoint_next.x or waypoint.y != waypoint_next.y:
      interpolated_trace = grp.trace_route(waypoint, waypoint_next)
      if len(interpolated_trace) > max_len:
        waypoints_trajectory[i + 1] = waypoints_trajectory[i]
      else:
        for wp_tuple in interpolated_trace:
          route.append((wp_tuple[0].transform, wp_tuple[1]))

  lat_ref, lon_ref = _get_latlon_ref(world_map)

  return location_route_to_gps(route, lat_ref, lon_ref), route


def location_route_to_gps(route, lat_ref, lon_ref):
  """
      Locate each waypoint of the route into gps, (lat long ) representations.
  :param route:
  :param lat_ref:
  :param lon_ref:
  :return:
  """
  gps_route = []

  for transform, connection in route:
    gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
    gps_route.append((gps_point, connection))

  return gps_route


def _get_latlon_ref(world_map):
  """
  Convert from waypoints world coordinates to CARLA GPS coordinates
  :return: tuple with lat and lon coordinates
  """
  xodr = world_map.to_opendrive()
  tree = ET.ElementTree(ET.fromstring(xodr))

  # default reference
  lat_ref = 42.0
  lon_ref = 2.0

  for opendrive in tree.iter('OpenDRIVE'):
    for header in opendrive.iter('header'):
      for georef in header.iter('geoReference'):
        if georef.text:
          str_list = georef.text.split(' ')
          for item in str_list:
            if '+lat_0' in item:
              lat_ref = float(item.split('=')[1])
            if '+lon_0' in item:
              lon_ref = float(item.split('=')[1])
  return lat_ref, lon_ref


def _location_to_gps(lat_ref, lon_ref, location):
  """
  Convert from world coordinates to GPS coordinates
  :param lat_ref: latitude reference for the current map
  :param lon_ref: longitude reference for the current map
  :param location: location to translate
  :return: dictionary with lat, lon and height
  """

  EARTH_RADIUS_EQUA = 6378137.0  # pylint: disable=invalid-name
  scale = math.cos(lat_ref * math.pi / 180.0)
  mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
  my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
  mx += location.x
  my -= location.y

  lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
  lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
  z = location.z

  return {'lat': lat, 'lon': lon, 'z': z}


class MapAgent(autonomous_agent.AutonomousAgent):
  """
    Main class that runs the agents with the run_step function
    """

  def setup(self, path_to_conf_file, route_index=None):
    """Sets up the agent. route_index is for logging purposes"""

    torch.cuda.empty_cache()
    self.track = autonomous_agent.Track.MAP
    self.config_path = path_to_conf_file
    self.step = -1
    self.initialized = False
    self.device = torch.device('cuda:0')
    # PERCEPTION PLANT SETTINGS:
    self.use_perc_plant = USE_PERC_PLANT
    self.det_th = DET_TH

    # Load the config saved during training
    with open(os.path.join(path_to_conf_file, 'config.pickle'), 'rb') as args_file:
      loaded_config = pickle.load(args_file)

    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the saved config.
    self.config.__dict__.update(loaded_config.__dict__)

    # For models supporting different output modalities we select which one to use here.
    # 0: Waypoints
    # 1: Path + Target Speed
    direct = os.environ.get('DIRECT', 1)
    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    if direct is not None:
      self.config.inference_direct_controller = int(direct)
      print('Direct control prediction?: ', direct)

    # If set to true, will generate visualizations at SAVE_PATH
    self.config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0)) == 1

    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))

    # Classification networks are known to be overconfident which leads to them braking a bit too late in our case.
    # Reducing the driving speed slightly counteracts that.
    if int(os.environ.get('SLOWER', 1)):
      print('Reduce target speed value by one.')
      self.config.target_speeds[2] = self.config.target_speeds[2] - 1.0
      self.config.target_speeds[3] = self.config.target_speeds[3] - 1.0

    if self.config.tp_attention:
      self.tp_attention_buffer = []

    # Stop signs can be occluded with our camera setup. This buffer remembers them until cleared.
    # TODO: Activate STOP_SIGN_CONTROLLER
    # Very useful on the LAV benchmark
    self.stop_sign_controller = int(os.environ.get('STOP_CONTROL', 0))
    print('Use stop sign controller:', self.stop_sign_controller)
    if self.stop_sign_controller:
      # There can be max 1 stop sign affecting the ego
      self.stop_sign_buffer = deque(maxlen=1)
      self.clear_stop_sign = 0  # Counter if we recently cleared a stop sign

    # Load model files
    if OPENPCDET:
      # TRACKING
      if TRACKING:
        self.lidar_freq = 1.0 / 10.0  # In seconds
        # self.simulator_time_step = (1.0 / 20.0)
        self.max_num_bb_forecast = 4  # Number of consecutive bb detection needed for a forecast
        self.min_num_bb_forecast = 4  # Minimum number of consecutive bb detection needed for a forecast
        self.bb_buffer_tracking = deque(maxlen=self.max_num_bb_forecast)
        for i in range(self.max_num_bb_forecast - self.min_num_bb_forecast):
            self.bb_buffer_tracking.append([])  # Fill in empty bounding boxes for the optional timesteps

      # Copied from tim for now
      original_args = sys.argv
      self.pcd_cfg_path = PCD_CFG_PATH 
      self.pcd_ckpt_path = PCD_CKPT_PATH 
      sys.argv = ['first_arg_is_skipped', '--cfg_file', self.pcd_cfg_path, '--ckpt', self.pcd_ckpt_path,
                  '--batch_size', '1']
      # this line needs line 14. in openpcdet.tools.test.py to look like from tools.eval_utils import eval_utils

      cwd = os.getcwd()
      PCDirectory = cwd + "/OpenPCDet/tools"
      os.chdir(PCDirectory)

      pcd_args, pcd_cfg = pcd_parser()
      os.chdir(cwd)
      sys.argv = original_args

      pcd_args.save_to_file = 0
      dist_test = False

      log_dir = path_l(cwd + "/evaluation/results/")
      # final_output_dir.mkdir(parents=True, exist_ok=True)
      log_file = log_dir / "pcd_log.txt"
      logger = common_utils.create_logger(log_file)

      test_set, test_loader, sampler = build_dataloader(
          dataset_cfg=pcd_cfg.DATA_CONFIG,
          class_names=pcd_cfg.CLASS_NAMES,
          batch_size=pcd_args.batch_size,
          dist=dist_test, workers=pcd_args.workers, logger=logger, training=False
      )
      print("Loading SECOND:")
      print(PCD_CKPT_PATH)
      pcd_model = build_network(model_cfg=pcd_cfg.MODEL, num_class=len(pcd_cfg.CLASS_NAMES), dataset=test_set)
      pcd_model.load_params_from_file(filename=pcd_args.ckpt, to_cpu=dist_test, logger=logger,
                                      pre_trained_path=pcd_args.pretrained_model)
      pcd_model.cuda()
      pcd_model.eval()

      self.pcd_model = pcd_model
      self.pcd_test_loader = test_loader
      self.pcd_detection_threshold = PCD_DETECTION_THRESHOLD # float(os.environ.get('DETECTION_THRESHOLD', 0))
      # self.cfg_agent.model.training.input_ego = False  # does not input ego, but does not skip first detection
    #else:

    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(path_to_conf_file):
      if file.endswith('.pth'):
        self.model_count += 1
        print("Loading TransFuser:")
        print(os.path.join(path_to_conf_file, file))
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location=self.device)

        net.load_state_dict(state_dict, strict=False)
        net.cuda(device=self.device)
        net.eval()
        self.nets.append(net)

    ####### Start of new Setup:
    planning_path_to_conf_file = PATH_TO_PLANNING_FILE
    with open(os.path.join(planning_path_to_conf_file, "config.pickle"), "rb") as args_file:
      loaded_config = pickle.load(args_file)
    self.planning_config = GlobalConfig()
    self.planning_config.__dict__.update(loaded_config.__dict__)
    
    self.planning_nets = []
    self.planning_model_count = 0
    # planning_path_to_conf_file = ""
    for file in os.listdir(planning_path_to_conf_file):
      if file.endswith('model_0046.pth'):
        self.planning_model_count += 1
        print("Loading PlanT:")
        print(os.path.join(planning_path_to_conf_file, file))
        net = PlanT(self.planning_config)
        if self.planning_config.sync_batch_norm:
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(planning_path_to_conf_file, file), map_location=self.device)

        net.load_state_dict(state_dict, strict=False)
        net.cuda()
        net.eval()
        self.planning_nets.append(net)

    assert self.config.lidar_seq_len == self.planning_config.lidar_seq_len
    assert self.config.data_save_freq == self.planning_config.data_save_freq
    
    
    print("\n\n")
    print("CUSTOM FLAGS:")
    print(f"\nTRACKING: {TRACKING} ")
    print(f"DET_TH_SECOND: {PCD_DETECTION_THRESHOLD}")
    print(f"DET_TH_TF: {DET_TH}")
    print(f"HALF_EXTENTS: {HALF_EXTENTS}")
    
    print(f"\nPCD_CFG_PATH: {PCD_CFG_PATH}")
    print(f"PCD_CKPT_PATH: {PCD_CKPT_PATH}")
    print(f"PATH_TO_PLANNING_FILE: {PATH_TO_PLANNING_FILE}")
    
    print(f"\nUSE_PERC_PLANT: {USE_PERC_PLANT}")
    print(f"OPENPCDET: {OPENPCDET}")
    print(f"ONLY_VEHICLE_BB: {ONLY_VEHICLE_BB}")
    print("\n\n")

    ####### End of new Setup

    self.stuck_detector = 0
    self.force_move = 0

    self.bb_buffer = deque(maxlen=1)
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.target_point_prev = [1e5, 1e5]

    # Filtering
    self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
    self.ukf = UKF(dim_x=4,
                   dim_z=4,
                   fx=bicycle_model_forward,
                   hx=measurement_function_hx,
                   dt=self.config.carla_frame_rate,
                   points=self.points,
                   x_mean_fn=state_mean,
                   z_mean_fn=measurement_mean,
                   residual_x=residual_state_x,
                   residual_z=residual_measurement_h)

    # State noise, same as measurement because we
    # initialize with the first measurement later
    self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
    # Measurement noise
    self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
    self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
    # Used to set the filter state equal the first measurement
    self.filter_initialized = False
    # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
    self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))

    #Temporal LiDAR
    self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)
    self.lidar_buffer_with_intensity = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)

    self.lidar_last = None
    self.lidar_last_with_intensity = None

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    # Path to where visualizations and other debug output gets stored
    self.save_path = os.environ.get('SAVE_PATH')

    # Logger that generates logs used for infraction replay in the results_parser.
    if self.save_path is not None and route_index is not None:
      self.save_path = pathlib.Path(self.save_path) / route_index
      pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

      self.lon_logger = ScenarioLogger(
          save_path=self.save_path,
          route_index=route_index,
          logging_freq=self.config.logging_freq,
          log_only=True,
          route_only=False,  # with vehicles
          roi=self.config.logger_region_of_interest,
      )
    else:
      self.save_path = None

  def _init(self, input_data):
    self.hd_map = carla.Map('RouteMap', input_data[1]['opendrive'])

    global_plan_world_coord_positions = []
    for point in self._global_plan_world_coord:
      global_plan_world_coord_positions.append(point[0].location)

    new_trajectory = interpolate_trajectory(self.hd_map, global_plan_world_coord_positions)

    self.hd_map_planner = RoutePlanner(self.config.dense_route_planner_min_distance,
                                       self.config.dense_route_planner_max_distance)
    self.hd_map_planner.set_route(new_trajectory[0], True)

    # During setup() not everything is available yet, so this _init is a second setup in run_step()
    # Privileged map access for logging and visualizations. Turned off during normal evaluation.
    if self.save_path is not None and self.nets:
      from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=locally-disabled, import-outside-toplevel
      from nav_planner import interpolate_trajectory as i_t  # pylint: disable=locally-disabled, import-outside-toplevel
      self.world_map = CarlaDataProvider.get_map()
      trajectory = [item[0].location for item in self._global_plan_world_coord]
      self.dense_route, _ = i_t(self.world_map, trajectory)  # privileged

      self._waypoint_planner = RoutePlanner(self.config.log_route_planner_min_distance,
                                            self.config.route_planner_max_distance)
      self._waypoint_planner.set_route(self.dense_route, True)

      vehicle = CarlaDataProvider.get_hero_actor()
      self.lon_logger.ego_vehicle = vehicle
      self.lon_logger.world = vehicle.get_world()

      self.nets[0].init_visualization()

    self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance)
    self._route_planner.set_route(self._global_plan, True)
    self.initialized = True

  def sensors(self):
    sensors = [{
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb_front'
    }, {
        'type': 'sensor.other.imu',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': self.config.carla_frame_rate,
        'id': 'imu'
    }, {
        'type': 'sensor.other.gnss',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': 0.01,
        'id': 'gps'
    }, {
        'type': 'sensor.speedometer',
        'reading_frequency': self.config.carla_fps,
        'id': 'speed'
    }, {
        'type': 'sensor.opendrive_map',
        'reading_frequency': 1e-6,
        'id': 'hd_map'
    }]
    # Don't set up LiDAR for camera only approaches
    if self.config.backbone not in ('aim'):
      sensors.append({
          'type': 'sensor.lidar.ray_cast',
          'x': self.config.lidar_pos[0],
          'y': self.config.lidar_pos[1],
          'z': self.config.lidar_pos[2],
          'roll': self.config.lidar_rot[0],
          'pitch': self.config.lidar_rot[1],
          'yaw': self.config.lidar_rot[2],
          'id': 'lidar'
      })

    return sensors

  @torch.inference_mode()  # Turns off gradient computation
  def tick(self, input_data):
    """Pre-processes sensor data and runs the Unscented Kalman Filter"""
    rgb = []
    for camera_pos in ['front']:
      rgb_cam = 'rgb_' + camera_pos
      camera = input_data[rgb_cam][1][:, :, :3]

      # Also add jpg artifacts at test time, because the training data was saved as jpg.
      _, compressed_image_i = cv2.imencode('.jpg', camera)
      camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)

      rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
      # Switch to pytorch channel first order
      rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
      rgb.append(rgb_pos)
    rgb = np.concatenate(rgb, axis=1)
    rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

    gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1][:2])
    speed = input_data['speed'][1]['speed']
    compass = t_u.preprocess_compass(input_data['imu'][1][-1])

    result = {
        'rgb': rgb,
        'compass': compass,
    }

    if self.config.backbone not in ('aim'):
      result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])
      result['lidar_intensity'] = t_u.lidar_to_ego_coordinate_with_intensity(self.config, input_data['lidar'])

    if not self.filter_initialized:
      self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
      self.filter_initialized = True

    self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
    self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
    filtered_state = self.ukf.x
    self.state_log.append(filtered_state)

    result['gps'] = filtered_state[0:2]

    waypoint_route = self._route_planner.run_step(filtered_state[0:2])
  
    if len(waypoint_route) > 2:
      target_point, far_command = waypoint_route[1]
    elif len(waypoint_route) > 1:
      target_point, far_command = waypoint_route[1]
    else:
      target_point, far_command = waypoint_route[0]

    if (target_point != self.target_point_prev).all():
      self.target_point_prev = target_point
      self.commands.append(far_command.value)

    one_hot_command = t_u.command_to_one_hot(self.commands[-2])
    result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

    ego_target_point = t_u.inverse_conversion_2d(target_point, result['gps'], result['compass'])
    ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

    result['target_point'] = ego_target_point

    waypoints_hd = self.hd_map_planner.run_step(filtered_state[0:2])
    if len(waypoints_hd) > 1:
      local_point, _ = waypoints_hd[1]
    else:
      local_point, _ = waypoints_hd[0]
    ego_local_point = t_u.inverse_conversion_2d(local_point, result['gps'], result['compass'])

    result['angle'] = -math.degrees(math.atan2(-ego_local_point[1], ego_local_point[0])) / 90.0
    result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

    # Preprocess route the same way as during training of PlanT:
    if USE_PERC_PLANT:

      route = [list(t_u.inverse_conversion_2d(i[0], result['gps'], result['compass'])) for i in waypoints_hd] # list of list for comperability to the plant_agent
      if len(route) < self.planning_config.num_route_points:
        num_missing = self.planning_config.num_route_points - len(route)
        route = np.array(route) # Shape: (20, 2)
        # Fill the empty spots by repeating the last point.
        route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
      else:
        route = np.array(route[:self.planning_config.num_route_points])

      if self.planning_config.smooth_route:
        route = self.data.smooth_path(route)
      route = torch.tensor(route, dtype=torch.float32)[:self.planning_config.num_route_points].to(self.device).unsqueeze(0)
      result["route"] = route

    if self.save_path is not None:
      waypoint_route = self._waypoint_planner.run_step(result['gps'])
      waypoint_route = extrapolate_waypoint_route(waypoint_route, self.config.route_points)
      route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])[:self.config.route_points]
      self.lon_logger.log_step(route)

    return result

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1

    if not self.initialized:
      if 'hd_map' in input_data.keys():
        self._init(input_data['hd_map'])
      else:
        return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.control = control
      tick_data = self.tick(input_data)
      # ['rgb', 'compass', 'lidar', 'gps', 'command', 'target_point', 'angle', 'speed']
      # PlanT input:
      # bounding_boxes, route, target_point, light_hazard, stop_hazard, junction, velocity
      if self.config.backbone not in ('aim'):
        self.lidar_last = deepcopy(tick_data['lidar'])
        self.lidar_last_with_intensity = deepcopy(tick_data['lidar_intensity'])
      return control

    # Need to run this every step for GPS filtering
    tick_data = self.tick(input_data)

    lidar_indices = []
    for i in range(self.config.lidar_seq_len):
      lidar_indices.append(i * self.config.data_save_freq)

    #Current position of the car
    ego_x = self.state_log[-1][0]
    ego_y = self.state_log[-1][1]
    
    if USE_PERC_PLANT:
      ego_z = 0
      ego_location = carla.Location(ego_x,ego_y,ego_z)
      ego_vehicle_waypoint = self.hd_map.get_waypoint(ego_location)
      self.is_junction = ego_vehicle_waypoint.is_junction

    ego_theta = self.state_log[-1][2]

    ego_x_last = self.state_log[-2][0]
    ego_y_last = self.state_log[-2][1]
    ego_theta_last = self.state_log[-2][2]

    # We only get half a LiDAR at every time step. Aligns the last half into the current coordinate frame.
    if self.config.backbone not in ('aim'):
      lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)
      lidar_last_with_intensity = self.align_lidar_with_intensity(self.lidar_last_with_intensity, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    # Updates stop boxes by vehicle movement converting past predictions into the current frame.
    if self.stop_sign_controller:
      self.update_stop_box(self.stop_sign_buffer, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    if self.config.backbone not in ('aim'):
      lidar_current = deepcopy(tick_data['lidar'])
      lidar_current_with_intensity = deepcopy(tick_data['lidar_intensity'])
      lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)
      lidar_full_with_intensity = np.concatenate((lidar_current_with_intensity, lidar_last_with_intensity), axis=0)
      # print(lidar_current.shape,lidar_last.shape,lidar_full.shape)
      self.lidar_buffer.append(lidar_full)
      self.lidar_buffer_with_intensity.append(lidar_full_with_intensity)

    if self.config.backbone not in ('aim'):
      # We wait until we have sufficient LiDARs
      if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
        self.lidar_last = deepcopy(tick_data['lidar'])
        self.lidar_last_with_intensity = deepcopy(tick_data['lidar_intensity'])
        tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
        self.control = tmp_control

        return tmp_control

    # Possible action repeat configuration
    if self.step % self.config.action_repeat == 1:
      self.lidar_last = deepcopy(tick_data['lidar'])
      self.lidar_last_with_intensity = deepcopy(tick_data['lidar_intensity'])

      return self.control

    if self.config.backbone in ('aim'):  # Image only method
      # Dummy data
      lidar_bev = torch.zeros((1, 1 + int(self.config.use_ground_plane), self.config.lidar_resolution_height,
                               self.config.lidar_resolution_width)).to(self.device, dtype=torch.float32)
    else:
      # Voxelize LiDAR and stack temporal frames
      lidar_bev = []
      # prepare LiDAR input
      for i in lidar_indices:
        lidar_point_cloud = deepcopy(self.lidar_buffer[i])

        # For single frame there is no point in realignment. The state_log index will also differ.
        if self.config.realign_lidar and self.config.lidar_seq_len > 1:
          # Position of the car when the LiDAR was collected
          curr_x = self.state_log[i][0]
          curr_y = self.state_log[i][1]
          curr_theta = self.state_log[i][2]

          # Voxelize to BEV for NN to process
          lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

        lidar_histogram = torch.from_numpy(
            self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                  use_ground_plane=self.config.use_ground_plane)).unsqueeze(0)

        lidar_histogram = lidar_histogram.to(self.device, dtype=torch.float32)
        lidar_bev.append(lidar_histogram)

        lidar_bev = torch.cat(lidar_bev, dim=1)

    if self.config.backbone not in ('aim'):
      self.lidar_last = deepcopy(tick_data['lidar'])
      self.lidar_last_with_intensity = deepcopy(tick_data['lidar_intensity'])

    # prepare velocity input
    gt_velocity = tick_data['speed']
    velocity = gt_velocity.reshape(1, 1)  # used by transfuser

    compute_debug_output = self.config.debug and (not self.save_path is None)

    # forward pass
    pred_wps = []
    pred_target_speeds = []
    pred_checkpoints = []
    bounding_boxes = []
    wp_selected = None
    for i in range(self.model_count):
      if self.config.backbone in ('transFuser', 'aim', 'bev_encoder'):
        pred_wp, \
        pred_target_speed, \
        pred_checkpoint, \
        pred_semantic, \
        pred_bev_semantic, \
        pred_depth, \
        pred_bb_features,\
        attention_weights,\
        pred_wp_1,\
        selected_path = self.nets[i].forward(
          rgb=tick_data['rgb'],
          lidar_bev=lidar_bev,
          target_point=tick_data['target_point'],
          ego_vel=velocity,
          command=tick_data['command'])
        
        # Only convert bounding boxes when they are used.
        if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                         self.stop_sign_controller):
          pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
        if self.use_perc_plant:
          pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
          # Filter bounding boxes -> filtering out light and stop boxes
          normal_pred_bounding_box = [box[:-1] for box in pred_bounding_box if box[-1] >= self.det_th ]
          removed_pred_bounding_box = [box[:-1] for box in pred_bounding_box if box[-1] >= self.det_th and box[-2] not in [2,3]]
          
          if ONLY_VEHICLE_BB: 
            pred_bounding_box = removed_pred_bounding_box
          else:
            pred_bounding_box = normal_pred_bounding_box
          
          # pred_bounding_box -> List of arrays (each entry is one prediction)
          # pred_bounding_box[0] -> BB: Array of Length 8: x,y, extent_x, extent_y, yaw, speed, brake, class
          # class: 0 -> car, 1 -> walker, 2 -> traffic light, 3 -> stop_sign

          pred_bounding_box_padded = torch.zeros((self.planning_config.max_num_bbs, 8), dtype=torch.float32).to(self.device)

          if len(pred_bounding_box) > 0:
            # Pad bounding boxes to a fixed number
            pred_bounding_box = np.stack(pred_bounding_box)
            pred_bounding_box = torch.tensor(pred_bounding_box, dtype=torch.float32).to(self.device)

            if pred_bounding_box.shape[0] <= self.planning_config.max_num_bbs:
              pred_bounding_box_padded[:pred_bounding_box.shape[0], :] = pred_bounding_box
            else:
              pred_bounding_box_padded[:self.planning_config.max_num_bbs, :] = pred_bounding_box[:self.planning_config.max_num_bbs]

          pred_bounding_box_padded = pred_bounding_box_padded.unsqueeze(0)
        
      else:
        raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

      if not self.use_perc_plant:
        if self.config.use_wp_gru:
          if self.config.multi_wp_output:
            wp_selected = 0
            if F.sigmoid(selected_path)[0].item() > 0.5:
              wp_selected = 1
              pred_wps.append(pred_wp_1)
            else:
              pred_wps.append(pred_wp)
          else:
            pred_wps.append(pred_wp)
        if self.config.use_controller_input_prediction:
          pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
          pred_checkpoints.append(pred_checkpoint[0][1])

        bounding_boxes.append(pred_bounding_box)

    # Average the predictions from ensembles
    # if True:
    if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                      self.stop_sign_controller):
      # We average bounding boxes by using non-maximum suppression on the set of all detected boxes.
      bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)

      self.bb_buffer.append(bbs_vehicle_coordinate_system)
    else:
      bbs_vehicle_coordinate_system = None

    if self.stop_sign_controller:
      stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

    if self.config.tp_attention:
      self.tp_attention_buffer.append(attention_weights[2])

    # Visualize the output of the last model
    if compute_debug_output and self.nets:
      if self.config.use_controller_input_prediction:
        prob_target_speed = F.softmax(pred_target_speed, dim=1)
      else:
        prob_target_speed = pred_target_speed

      self.nets[0].visualize_model(self.save_path,
                                   self.step,
                                   tick_data['rgb'],
                                   lidar_bev,
                                   tick_data['target_point'],
                                   pred_wp,
                                   pred_semantic=pred_semantic,
                                   pred_bev_semantic=pred_bev_semantic,
                                   pred_depth=pred_depth,
                                   pred_checkpoint=pred_checkpoint,
                                   pred_speed=prob_target_speed,
                                   pred_bb=bbs_vehicle_coordinate_system,
                                   gt_speed=gt_velocity,
                                   gt_wp=pred_wp_1,
                                   wp_selected=wp_selected)
    if OPENPCDET:
      # copied from Tim
      lidar_pc = PointCloud(self.lidar_buffer_with_intensity[-1])
      lidar_unified = transform_pc_to_unified(lidar_pc)
      pc = lidar_unified.pointcloud
      # if self.cfg.EXCLUDE_LIDAR_BACK: # remove all with negative x coordinates (since coordinates in uniformed coordinates)
      #     pc = pc[pc[:,0] > 0] 
      # create batch
      input_dict = {'frame_id': 000000, 'points': pc}
      test_set = self.pcd_test_loader.dataset
      batch_dict = [test_set.prepare_data(input_dict)]
      batch_dict = test_set.collate_batch(batch_dict)
      load_data_to_gpu(batch_dict)
      # predict
      with torch.no_grad():
          pred_dicts, ret_dict = self.pcd_model(batch_dict)

      annos = test_set.generate_prediction_dicts(
          batch_dict, pred_dicts, test_set.class_names,
          output_path=None
      )
      pcd_detection_threshold = self.pcd_detection_threshold
      pred_bounding_box_second = []
      frame = annos[0]
      if 'name' in frame:
        preds_in_frame = len(frame["name"])
      for i in range(preds_in_frame):
        if frame["score"][i] >= pcd_detection_threshold:
          x, y, z, dx, dy, dz, heading_angle = frame['boxes_lidar'][i]
          if HALF_EXTENTS:
            dx /= 2
            dy /= 2
          heading_angle = t_u.normalize_angle(heading_angle)
          brake = 0
          speed = 0 
          pred_class_name = frame["name"][i]
          if pred_class_name in ["Car", "Cyclists"]:
            pred_class = 0
          elif pred_class_name == "Pedestrian":
            pred_class = 1
          else:
            # print("Should not get here!!!!!")
            # print(pred_class)
            pred_class = 0
          bbox = np.array([x, y, dx, dy, heading_angle, speed, brake, pred_class])
          pred_bounding_box_second.append(bbox)

      """
      def non_maximum_suppression(self, bounding_boxes, iou_treshhold=0.2):

        corners = [self.get_bb_corner(box) for box in bounding_boxes]
        conf_scores = [box["score"] for box in bounding_boxes]

        filtered_boxes = []
        # bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

        if len(bounding_boxes) == 0:  # If no bounding boxes are detected can't do NMS
            return filtered_boxes

        confidences_indices = np.argsort(conf_scores)
        while (len(confidences_indices) > 0):
            idx = confidences_indices[-1]
            current_bb_corner = corners[idx]
            current_bb_dict = bounding_boxes[idx]
            filtered_boxes.append(current_bb_dict)
            confidences_indices = confidences_indices[:-1]  # Remove last element from the list

            if (len(confidences_indices) == 0):
                break

            for idx2 in deepcopy(confidences_indices):
                if (self.iou_bbs(current_bb_corner, corners[idx2]) > iou_treshhold):  # Remove BB from list
                    confidences_indices = confidences_indices[confidences_indices != idx2]

        return filtered_boxes
      """
      # TRACKING + MATCHING for speed prediction
      if TRACKING:
        boxes_corner_rep = [get_bb_corner(box) for box in pred_bounding_box_second]
        # x,y, extent_x, extent_y, yaw, speed, brake, class
        """
        if DEBUG_SAVE_BB and self.step < 100:
          save_dir = "saves/data/DEBUG"
          np.save(f'{save_dir}/{self.step}_SECOND.npy', np.array(pred_bounding_box_second, dtype=object), allow_pickle=True)
          np.save(f'{save_dir}/{self.step}_TF.npy', np.array(pred_bounding_box.cpu(), dtype=object), allow_pickle=True)
          # np.save(f'{save_dir}/{self.step}_LIDAR_1.npy', np.array(input_data['lidar'][1], dtype=object), allow_pickle=True)
          np.save(f'{save_dir}/{self.step}_LIDAR.npy', np.array(self.lidar_buffer_with_intensity[-1], dtype=object), allow_pickle=True)
          # np.save(f'{save_dir}/{self.step}_LIDAR_3.npy', np.array(lidar_bev[-1].cpu(), dtype=object), allow_pickle=True)
        """     
        self.bb_buffer_tracking.append(boxes_corner_rep)
        self.update_bb_buffer_tracking()
        self.instances = self.match_bb(self.bb_buffer_tracking)  # Associate bounding boxes to instances
        self.list_of_unique_instances = [l[0] for l in self.instances]      
        speed, unnormalized_speed = self.get_speed()
        print(f"Speed predictions: {unnormalized_speed}")
        if speed:
          speed = speed[::-1]
          speed_iter = 0
          for ix, box in enumerate(pred_bounding_box_second):
            if ix not in self.list_of_unique_instances:
              continue
            # box = np.array([x, y, dx, dy, heading_angle, speed, brake, pred_class])   
            box[5] = speed[speed_iter]
            speed_iter += 1

      # Padding
      pred_bounding_box_padded_second = torch.zeros((self.planning_config.max_num_bbs, 8), dtype=torch.float32).to(self.device)

      if len(pred_bounding_box_second ) > 0:
        # Pad bounding boxes to a fixed number
        pred_bounding_box_second = np.stack(pred_bounding_box_second)
        pred_bounding_box_second = torch.tensor(pred_bounding_box_second, dtype=torch.float32).to(self.device)

        if pred_bounding_box_second.shape[0] <= self.planning_config.max_num_bbs:
          pred_bounding_box_padded_second[:pred_bounding_box_second.shape[0], :] = pred_bounding_box_second
        else:
          pred_bounding_box_padded_second[:self.planning_config.max_num_bbs, :] = pred_bounding_box_second[:self.planning_config.max_num_bbs]

      pred_bounding_box_padded_second = pred_bounding_box_padded_second.unsqueeze(0)

    if USE_PERC_PLANT or OPENPCDET:
      # TODO: dont use torch.tensor on a tensor, rather use sourceTensor.clone().detach()
      speed = torch.tensor(gt_velocity, dtype=torch.float32).to(self.device).unsqueeze(0)
      target_point = torch.tensor(tick_data['target_point'], dtype=torch.float32).to(self.device)
      route = tick_data['route']
      junction = torch.tensor(self.is_junction, dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)

      # TODO: Use stop_for_stop_sign as flag instead for the stop sign input
      pred_classes = [i[7].item() for i in normal_pred_bounding_box]
      pred_light = [j == 2 for j in pred_classes]
      pred_stop_sign = [j == 3 for j in pred_classes]
      pred_light_hazard = any(pred_light)
      # if pred_light_hazard:
      #   print(pred_light_hazard)
      pred_stop_sign_hazard = any(pred_stop_sign)
      pred_light_hazard = torch.IntTensor([[pred_light_hazard]]).to("cuda:0")
      pred_stop_sign_hazard = torch.IntTensor([[pred_stop_sign_hazard]]).to("cuda:0")

      pred_wps = []
      pred_target_speeds = []
      pred_checkpoints = [] 
      pred_bbs = []

      # TODO: Remove Debug prints when finishing the code
      # print("\n\n\n")
      # print(f"Step: {self.step}")
      # print(f"target_point: {target_point[0]}")
      # print(f"pred_light_hazard: {pred_light_hazard.item()}")
      # print(f"pred_stop_sign_hazard: {pred_stop_sign_hazard.item()}")
      # print(f"Junction: {junction.item()}")
      # print(f"velocity: {velocity.item()}")
      
      for i in range(self.planning_model_count):
        pred_wp, pred_target_speed, pred_checkpoint, pred_bb = self.planning_nets[i].forward(bounding_boxes=pred_bounding_box_padded_second,# pred_bounding_box_padded, # bounding_boxes_padded,
                                                                                  route=route, # pred_checkpoint,
                                                                                  target_point=target_point,
                                                                                  light_hazard=pred_light_hazard, # light_hazard,
                                                                                  stop_hazard=pred_stop_sign_hazard, # stop_sign_hazard,
                                                                                  junction=junction,
                                                                                  velocity=speed)

      
      # print(f"Pred WP: {pred_wp}") # torch.Size([1, 40, 2])
      pred_wps.append(pred_wp)
      pred_bbs.append(t_u.plant_quant_to_box(self.planning_config, pred_bb))
      if self.planning_config.use_controller_input_prediction:
        # True
        pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
        pred_checkpoints.append(pred_checkpoint[0][1])

      if self.planning_config.use_wp_gru:
        # True
        self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

      pred_bbs = torch.stack(pred_bbs, dim=0).mean(dim=0)
      if self.planning_config.use_controller_input_prediction:
        # True
        pred_target_speed = torch.stack(pred_target_speeds, dim=0).mean(dim=0)
        pred_aim_wp = torch.stack(pred_checkpoints, dim=0).mean(dim=0)
        pred_aim_wp = pred_aim_wp.detach().cpu().numpy()
        pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0
        
        if self.uncertainty_weight:
          # True
          uncertainty = pred_target_speed.detach().cpu().numpy()
          if uncertainty[0] > self.planning_config.brake_uncertainty_threshold:
            pred_target_speed = self.planning_config.target_speeds[0]
          else:
            pred_target_speed = sum(uncertainty * self.planning_config.target_speeds)
        else:
          pred_target_speed_index = torch.argmax(pred_target_speed)
          pred_target_speed = self.planning_config.target_speeds[pred_target_speed_index]

      if self.planning_config.inference_direct_controller and \
          self.planning_config.use_controller_input_prediction:
        steer, throttle, brake = self.planning_nets[0].control_pid_direct(pred_target_speed, pred_angle, speed, False)
      else:
        # True
        steer, throttle, brake = self.planning_nets[0].control_pid(self.pred_wp, speed, False)
      # print(f"Control: Steer - {steer}, Throttle - {throttle}, Brake - {brake}")
    else:
      if self.config.use_wp_gru:
        self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

      if self.config.use_controller_input_prediction:
        pred_target_speed = torch.stack(pred_target_speeds, dim=0).mean(dim=0)

        if self.uncertainty_weight:
          uncertainty = pred_target_speed.detach().cpu().numpy()
          if uncertainty[0] > self.config.brake_uncertainty_threshold:
            pred_target_speed = self.config.target_speeds[0]
          else:
            pred_target_speed = sum(uncertainty * self.config.target_speeds)
        else:
          pred_target_speed_index = torch.argmax(pred_target_speed)
          pred_target_speed = self.config.target_speeds[pred_target_speed_index]

      if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
        steer, throttle, brake = self.nets[0].control_pid_direct(pred_target_speed, tick_data['angle'], gt_velocity)
      elif self.config.use_wp_gru and not self.config.inference_direct_controller:
        steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, gt_velocity)
      else:
        raise ValueError('An output representation was chosen that was not trained.')

    # 0.1 is just an arbitrary low number to threshold when the car is stopped
    if gt_velocity < 0.1:
      self.stuck_detector += 1
    else:
      self.stuck_detector = 0

    # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
    if self.stuck_detector > self.config.stuck_threshold:
      self.force_move = self.config.creep_duration

    if self.force_move > 0:
      emergency_stop = False
      if self.config.backbone not in ('aim'):
        # safety check
        safety_box = deepcopy(self.lidar_buffer[-1])

        # z-axis
        safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

        # y-axis
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

        # x-axis
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        emergency_stop = (len(safety_box) > 0)  # Checks if the List is empty

      if not emergency_stop:
        print('Detected agent being stuck. Step: ', self.step)
        throttle = max(self.config.creep_throttle, throttle)
        brake = False
        self.force_move -= 1
      else:
        print('Creeping stopped by safety box. Step: ', self.step)
        throttle = 0.0
        brake = True
        self.force_move = self.config.creep_duration

    if self.stop_sign_controller:
      if stop_for_stop_sign:
        throttle = 0.0
        brake = True

    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

    # CARLA will not let the car drive in the initial frames.
    # We set the action to brake so that the filter does not get confused.
    if self.step < self.config.inital_frames_delay:
      self.control = carla.VehicleControl(0.0, 0.0, 1.0)
    else:
      self.control = control

    return control

  def stop_sign_controller_step(self, ego_speed):
    """Checks whether the car is intersecting with one of the detected stop signs"""
    if self.clear_stop_sign > 0:
      self.clear_stop_sign -= 1

    if len(self.bb_buffer) < 1:
      return False
    stop_sign_stop_predicted = False
    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)
    origin = carla.Location(x=0.0, y=0.0, z=0.0)

    car_box = carla.BoundingBox(origin, extent)

    for bb in self.bb_buffer[-1]:
      if bb[7] == 3:  # Stop sign detected
        self.stop_sign_buffer.append(bb)

    if len(self.stop_sign_buffer) > 0:
      # Check if we need to stop
      stop_box = self.stop_sign_buffer[0]
      stop_origin = carla.Location(x=stop_box[0], y=stop_box[1], z=0.0)
      stop_extent = carla.Vector3D(stop_box[2], stop_box[3], 1.0)
      stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
      stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box[4]), 0.0)

      if t_u.check_obb_intersection(stop_carla_box, car_box) and self.clear_stop_sign <= 0:
        if ego_speed > 0.01:
          stop_sign_stop_predicted = True
        else:
          # We have cleared the stop sign
          stop_sign_stop_predicted = False
          self.stop_sign_buffer.pop()
          # Stop signs don't come in herds, so we know we don't need to clear one for a while.
          self.clear_stop_sign = 100

    if len(self.stop_sign_buffer) > 0:
      # Remove boxes that are too far away
      if np.linalg.norm(self.stop_sign_buffer[0][:2]) > abs(self.config.max_x):
        self.stop_sign_buffer.pop()

    return stop_sign_stop_predicted

  def bb_detected_in_front_of_vehicle(self, ego_speed):
    if len(self.bb_buffer) < 1:  # We only start after we have 4 time steps.
      return False

    collision_predicted = False

    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)

    # Safety box
    bremsweg = ((ego_speed.cpu().numpy().item() * 3.6) / 10.0)**2 / 2.0  # Bremsweg formula for emergency break
    safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # plus one meter is the car.

    center_safety_box = carla.Location(x=safety_x, y=0.0, z=1.0)

    safety_bounding_box = carla.BoundingBox(center_safety_box, extent)
    safety_bounding_box.rotation = carla.Rotation(0.0, 0.0, 0.0)

    for bb in self.bb_buffer[-1]:
      # We just give them some arbitrary height. Does not matter
      bb_extent_z = 1.0
      loc_local = carla.Location(bb[0], bb[1], 0.0)
      extent_det = carla.Vector3D(bb[2], bb[3], bb_extent_z)
      bb_local = carla.BoundingBox(loc_local, extent_det)
      bb_local.rotation = carla.Rotation(0.0, np.rad2deg(bb[4]).item(), 0.0)

      if t_u.check_obb_intersection(safety_bounding_box, bb_local):
        collision_predicted = True

    return collision_predicted

  def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                [np.sin(orientation_target),
                                 np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    return t_u.algin_lidar(lidar, pos_diff, rot_diff)
  
  def align_lidar_with_intensity(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target, 0.0, 0.0]) - np.array([x, y, 0.0, 0.0])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0, 0.0],
                                [np.sin(orientation_target),
                                 np.cos(orientation_target), 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    return t_u.algin_lidar_with_intensity(lidar, pos_diff, rot_diff)

  def update_stop_box(self, boxes, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target]) - np.array([x, y])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target)],
                                [np.sin(orientation_target), np.cos(orientation_target)]])
    pos_diff = rotation_matrix.T @ pos_diff

    # Rotation matrix in local coordinate system
    local_rot_matrix = np.array([[np.cos(rot_diff), -np.sin(rot_diff)], [np.sin(rot_diff), np.cos(rot_diff)]])

    for _, box_pred in enumerate(boxes):
      box_pred[:2] = (local_rot_matrix.T @ (box_pred[:2] - pos_diff).T).T
      box_pred[4] = t_u.normalize_angle(box_pred[4] - rot_diff)

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
    Also writes logging files to disk.
    """
    if self.save_path is not None and self.nets:
      self.lon_logger.dump_to_json()
      if len(self.nets[0].speed_histogram) > 0:
        with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
          ujson.dump(self.nets[0].speed_histogram, f, indent=4)

      if self.config.tp_attention:
        if len(self.tp_attention_buffer) > 0:
          print('Average TP attention: ', sum(self.tp_attention_buffer) / len(self.tp_attention_buffer))
          with gzip.open(self.save_path / 'tp_attention.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(self.tp_attention_buffer, f, indent=4)

        del self.tp_attention_buffer

    del self.nets
    del self.config

    if self.planning_nets:
      del self.planning_nets
      del self.planning_config
    if OPENPCDET:
      del self.pcd_model
      del self.pcd_test_loader 

  def update_bb_buffer_tracking(self):
    if (len(self.state_log) < 2):  # Start after we have the second measurement
        return

    current_state = np.array(self.state_log[-1])
    R_curr = torch.tensor([[np.cos(current_state[2]), -np.sin(current_state[2])],
                          [np.sin(current_state[2]), np.cos(current_state[2])]])

    for j in range(len(self.bb_buffer_tracking[-1])):  # over detections in last buffer timestep (new detections)
        for k in range(self.bb_buffer_tracking[-1][j].shape[0]):  # Loop over points of the box
            self.bb_buffer_tracking[-1][j][k, 1] = -1 * self.bb_buffer_tracking[-1][j][k, 1]  # make y coordinate to the right
            self.bb_buffer_tracking[-1][j][k, :2] = torch.tensor(current_state[0:2].copy()) + (
                              R_curr @ self.bb_buffer_tracking[-1][j][k, :2])


  def match_bb(self, buffer_bb):
    instances = []
    # We only start after we have 4 time steps.
    if (len(buffer_bb) < self.max_num_bb_forecast):
        return instances

    all_indices = []
    for i in range(len(buffer_bb) - 1):
        if (len(buffer_bb[i]) == 0 or len(buffer_bb[i + 1]) == 0):
            # Timestep without bounding boxes so there is no match
            all_indices.append([])
            continue

        matrix_size = max(len(buffer_bb[i]), len(buffer_bb[i + 1]))

        # Initialize with a large value so that bb that don't exist get matched last.
        ious = np.ones((matrix_size, matrix_size)) * 10.0
        for j in range(len(buffer_bb[i])):
            for k in range(len(buffer_bb[i + 1])):
                # Invert IOU here to convert value to costs
                ious[j, k] = 1.0 - iou_bbs(buffer_bb[i][j], buffer_bb[i + 1][k])

        m = Munkres()
        indexes = m.compute(ious)
        all_indices.append(indexes)
    
    inv_instances = []
    # Create instance for every newest bb.
    for i in range(len(buffer_bb[-1]) - 1, -1, -1):
        instance = [i]
        write = True
        continue_adding_bbs = True
        last_timestep_index = i
        # Loops over available timesteps starting with the latest
        for j in range(len(buffer_bb) - 1, 0, -1):
            if (continue_adding_bbs == False):
                break

            # There was a timestep with no matches / no bbs.
            if (len(all_indices[j - 1]) == 0):
                # If we have enough bb write the instance, else delete it.
                if (len(instance) < self.min_num_bb_forecast):
                    write = False
                break
            # Loops over pairs for each timestep
            for k in range(len(all_indices[j - 1])):
                # Find the match for the current bb
                if (all_indices[j - 1][k][1] == last_timestep_index):
                    # Check if the matched bb actually exists
                    if (all_indices[j - 1][k][0] >= len(buffer_bb[j - 1])):
                        # This instance has a timestep without a bb
                        if (len(instance) >= self.min_num_bb_forecast):
                            # Stop instance here and write it
                            continue_adding_bbs = False
                            break
                        else:
                            # There are less total bb than needed. Delete instance!
                            write = False
                    else:
                        instance.append(all_indices[j - 1][k][0])
                        last_timestep_index = all_indices[j - 1][k][0]
                        break

        if (write == True):
            inv_instances.append(instance)
    return inv_instances

  def get_speed(self):
    # We only start after we have 4 time steps.
    if (len(self.bb_buffer_tracking) < self.max_num_bb_forecast):
        return False, False

    speed = []
    unnormalized_speed = []

    self.instance_future_bb = []
    for i in range(len(self.instances)):
      
        bb_array = self.get_bb_of_instance(i)  # Format of BB: [x,y, orientation, speed, extent_x, extent_y]

        # 0 index is the oldest timestep
        # Ids are from old -> new
        box_m1 = bb_array[-1]  # Most recent bounding box
        box_m2 = bb_array[-2]

        distance_vector_m2 = box_m1[0:2] - box_m2[0:2]
        # Our predictions happen at 100ms intervals. So we need to multiply by 10 to get m/s scale.
        velocity_m2 = np.linalg.norm(distance_vector_m2) / (
                    0.5 * self.lidar_freq)  # TODO Tim changed the freq ad hoc to half -> Does this make sense here???
        
        unnormalized_speed.append(velocity_m2)
        
        if velocity_m2 < 0.01: velocity_m2 = 0.0
        if velocity_m2 > 8 : velocity_m2 = 8.0
        speed.append(velocity_m2)

    return speed, unnormalized_speed
    
  def get_bb_of_instance(self, instance_id):
      '''
      Args:
          instance_id: The instance if of the bounding box in the self.instances array
      Returns:
          List of bounding boxes belonging to that instance. The first item is the oldest bb, the last one is the most recent one.
          An instance can have a varying number of past bounding boxes, so accessing the array from back to front is advised.
          Format of BB: [x,y, orientation, speed, extent_x, extent_y]
      '''
      if (len(self.bb_buffer_tracking) < self.max_num_bb_forecast):  # We only start after we have 4 time steps.
          return []
      instance_bbs = []

      for j in range(self.max_num_bb_forecast):  # From oldest to newest BB
          inv_timestep = (self.max_num_bb_forecast - 1) - j
          if (len(self.instances[instance_id]) <= inv_timestep):
              continue  # This instance does not have a bb at this timestep
          bb = self.bb_buffer_tracking[j][self.instances[instance_id][inv_timestep]]

          instance_bbs.append(np.array([bb[4, 0], bb[4, 1]]))

      return instance_bbs
"""   def get_detections(self, input_data):
    
    lidar_pc = PointCloud(input_data['lidar'][1])
    lidar_unified = transform_pc_to_unified(lidar_pc)
    pc = lidar_unified.pointcloud
    if self.cfg.EXCLUDE_LIDAR_BACK: # remove all with negative x coordinates (since coordinates in uniformed coordinates)
        pc = pc[pc[:,0] > 0] 
    # create batch
    input_dict = {'frame_id': 000000, 'points': pc}
    test_set = self.pcd_test_loader.dataset
    batch_dict = [test_set.prepare_data(input_dict)]
    batch_dict = test_set.collate_batch(batch_dict)
    load_data_to_gpu(batch_dict)
    # predict
    with torch.no_grad():
        pred_dicts, ret_dict = self.pcd_model(batch_dict)

    annos = test_set.generate_prediction_dicts(
        batch_dict, pred_dicts, test_set.class_names,
        output_path=None
    )
    threshold = self.pcd_detection_threshold

    det_labels_car = read_bbs_from_results_frame(annos[0], threshold)
    for det in det_labels_car:
        det["yaw"] = normalize_angle(det["yaw"])
    return det_labels_car

  def non_maximum_suppression(self, bounding_boxes, iou_treshhold=0.2):

    corners = [get_bb_corner(box) for box in bounding_boxes]
    conf_scores = [box["score"] for box in bounding_boxes]

    filtered_boxes = []
    # bounding_boxes = np.array(list(itertools.chain.from_iterable(bounding_boxes)), dtype=np.object)

    if len(bounding_boxes) == 0:  # If no bounding boxes are detected can't do NMS
        return filtered_boxes

    confidences_indices = np.argsort(conf_scores)
    while (len(confidences_indices) > 0):
        idx = confidences_indices[-1]
        current_bb_corner = corners[idx]
        current_bb_dict = bounding_boxes[idx]
        filtered_boxes.append(current_bb_dict)
        confidences_indices = confidences_indices[:-1]  # Remove last element from the list

        if (len(confidences_indices) == 0):
            break

        for idx2 in deepcopy(confidences_indices):
            if (self.iou_bbs(current_bb_corner, corners[idx2]) > iou_treshhold):  # Remove BB from list
                confidences_indices = confidences_indices[confidences_indices != idx2]

    return filtered_boxes

  def finalize_instances(self, speed, bbs, bbs_corner):
    label_final = []
    label_final.append({'class': 'Car', 'extent': [1.5107464790344238, 4.901683330535889, 2.128324270248413],
                        'position': [-1.3, 0.0, -2.5], 'yaw': 0, 'num_points': -1, 'distance': -1, 'speed': 0.0,
                        'brake': 0.0, 'id': 99999})

    if speed == False:
        # speed = [0.0]*len(bbs)
        return label_final

    speed = speed[::-1]
    speed_iter = 0
    for ix, box in enumerate(bbs):
        if ix not in self.list_of_unique_instances:
            continue

        label_final.append({})
        label_final[-1]['class'] = 'Car'
        label_final[-1]['extent'] = box["extent"]
        label_final[-1]['position'] = box["position"]
        # vehicles are predicted in vehicle coordinate system but we need it in lidar coordinate system
        label_final[-1]['yaw'] = box["yaw"]
        label_final[-1]['speed'] = speed[speed_iter]
        label_final[-1]['id'] = ix
        speed_iter += 1
        label_final[-1]["score"] = box["score"] 
        label_final[-1]["distance"] = box["distance"]

    return label_final """


# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
  # Kinematic bicycle model.
  # Numbers are the tuned parameters from World on Rails
  front_wb = -0.090769015
  rear_wb = 1.4178275

  steer_gain = 0.36848336
  brake_accel = -4.952399
  throt_accel = 0.5633837

  locs_0 = x[0]
  locs_1 = x[1]
  yaw = x[2]
  speed = x[3]

  if brake:
    accel = brake_accel
  else:
    accel = throt_accel * throttle

  wheel = steer_gain * steer

  beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
  next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
  next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
  next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
  next_speed = speed + accel * dt
  next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

  next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

  return next_state_x


def measurement_function_hx(vehicle_state):
  '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
  return vehicle_state


def state_mean(state, wm):
  '''
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def measurement_mean(state, wm):
  '''
  We use the arctan of the average of sin and cos of the angle to
  calculate the average of orientations.
  :param state: array of states to be averaged. First index is the
  timestep.
  '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def residual_state_x(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


def residual_measurement_h(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y

def normalize_angle(x):
  x = x % (2 * np.pi)    # force in range [0, 2 pi)
  if x > np.pi:          # move to [-pi, pi)
      x -= 2 * np.pi
  return x

def get_bb_corner(box):
  ext = box[2:4]
  pos = box[0:2]
  yaw = box[4]
  r = np.array(([np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]))

  p1 = pos + r @ ([-0.5, -0.5] * ext)
  p2 = pos + r @ ([0.5, -0.5] * ext)
  p3 = pos + r @ ([0.5, 0.5] * ext)
  p4 = pos + r @ ([-0.5, 0.5] * ext)
  p5 = pos
  p6 = pos + r @ (np.array([0, ext[1] * 1 * 0.5]))

  two_d_bb = torch.from_numpy(np.array([p1, p2, p3, p4, p5, p6]))

  # add global pos - nope
  # two_d_bb = two_d_bb + self.state_log[-1][:2]

  three_d_bb = np.c_[two_d_bb, np.ones(6)]
  return torch.tensor(three_d_bb).squeeze()


def iou_bbs(bb1, bb2):
    a = Polygon([(bb1[0, 0], bb1[0, 1]), (bb1[1, 0], bb1[1, 1]), (bb1[2, 0], bb1[2, 1]), (bb1[3, 0], bb1[3, 1])])
    b = Polygon([(bb2[0, 0], bb2[0, 1]), (bb2[1, 0], bb2[1, 1]), (bb2[2, 0], bb2[2, 1]), (bb2[3, 0], bb2[3, 1])])
    intersection_area = a.intersection(b).area
    union_area = a.union(b).area
    iou = intersection_area / union_area
    return iou