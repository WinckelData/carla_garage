"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.
"""

import os
import torch
import torch.nn.functional as F
import pickle
from plant import PlanT
from data_agent import DataAgent
from data import CARLA_Data
import math
import cv2
import numpy as np

import carla
from config import GlobalConfig
import transfuser_utils as t_u

SAVE_PATH = os.environ.get('SAVE_PATH', None)
PERC_DEBUG = True

from nav_planner import extrapolate_waypoint_route
from collections import deque

def get_entry_point():
  return 'PerceptionPlanTAgent'


class PerceptionPlanTAgent(DataAgent):
  """
    Privileged driving agent used for data collection.
    Drives by accessing the simulator directly.
    """

  def setup(self, path_to_conf_file, route_index=None):
    super().setup(path_to_conf_file, route_index)

    torch.cuda.empty_cache()

    with open(os.path.join(path_to_conf_file, 'config.pickle'), 'rb') as args_file:
      loaded_config = pickle.load(args_file)

    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the save config.
    self.config.__dict__.update(loaded_config.__dict__)

    self.config.debug = int(os.environ.get('VISU_PLANT', 0)) == 1
    self.device = torch.device('cuda:0')

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    self.config.inference_direct_controller = int(os.environ.get('DIRECT', 0))
    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
    if self.uncertainty_weight:
      print('Uncertainty threshold: ', self.config.brake_uncertainty_threshold)

    # Load model files
    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(path_to_conf_file):
      if file.endswith('.pth'):
        self.model_count += 1
        print("\n Loading Main Agent:")
        print(os.path.join(path_to_conf_file, file))
        net = PlanT(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(path_to_conf_file, file), map_location=self.device)

        net.load_state_dict(state_dict, strict=False)
        net.cuda()
        net.eval()
        self.nets.append(net)

    if PERC_DEBUG: 
      from model import LidarCenterNet
      self.initialized = False
      self.det_th = 0.4
      perc_path_to_conf_file = '/home/luis/Desktop/HIWI/carla_garage/pretrained_models/longest6/tfpp_all_0'

      # Load the config saved during training
      with open(os.path.join(perc_path_to_conf_file, 'config.pickle'), 'rb') as args_file:
        loaded_config = pickle.load(args_file)

      # Generate new config for the case that it has new variables.
      self.perc_config = GlobalConfig()
      # Overwrite all properties that were set in the saved config.
      self.perc_config.__dict__.update(loaded_config.__dict__)

      self.perc_nets = []
      self.perc_model_counts = 0
      for file in os.listdir(perc_path_to_conf_file):
        if file.endswith('.pth'):
          self.perc_model_counts += 1
          print("\n Loading Perception Agent:")
          print(os.path.join(perc_path_to_conf_file, file))
          net = LidarCenterNet(self.perc_config)
          if self.perc_config.sync_batch_norm:
            # Model was trained with Sync. Batch Norm.
            # Need to convert it otherwise parameters will load wrong.
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
          state_dict = torch.load(os.path.join(perc_path_to_conf_file, file), map_location=self.device)

          net.load_state_dict(state_dict, strict=False)
          net.cuda()
          net.eval()
          self.perc_nets.append(net)
      assert self.config.lidar_seq_len == self.perc_config.lidar_seq_len
      assert self.config.data_save_freq == self.perc_config.data_save_freq
    if self.config.debug:
      self.init_map = False

  def sensors(self):
    result = super().sensors()
    if self.config.debug:
      result += [{
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
          'id': 'rgb_debug'
      }]
    return result

  def perc_plant_tick(self, input_data):
    """"""
    rgb = []
    for camera_pos in ['']:
      rgb_cam = 'rgb' + camera_pos
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

    # gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1][:2])
    speed = input_data['speed'][1]['speed']
    compass = t_u.preprocess_compass(input_data['imu'][1][-1])

    result = {
        'rgb': rgb,
        'compass': compass,
    }

    result['lidar'] = input_data['lidar']

    # if not self.filter_initialized:
    #   self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
    #   self.filter_initialized = True

    # self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
    # self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
    # filtered_state = self.ukf.x
    # self.state_log.append(filtered_state)

    # result['gps'] = filtered_state[0:2]

    # waypoint_route = self._route_planner.run_step(filtered_state[0:2])

    # if len(waypoint_route) > 2:
    #   target_point, far_command = waypoint_route[1]
    # elif len(waypoint_route) > 1:
    #   target_point, far_command = waypoint_route[1]
    # else:
    #   target_point, far_command = waypoint_route[0]

    # if (target_point != self.target_point_prev).all():
    #   self.target_point_prev = target_point
    #   self.commands.append(far_command.value)

    # one_hot_command = t_u.command_to_one_hot(self.commands[-2])
    # result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

    # ego_target_point = t_u.inverse_conversion_2d(target_point, result['gps'], result['compass'])
    # ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

    # result['target_point'] = ego_target_point

    result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

    one_hot_command = t_u.command_to_one_hot(self.commands[-2])
    result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)


    if self.save_path is not None:
      waypoint_route = self._waypoint_planner.run_step(result['gps'])
      waypoint_route = extrapolate_waypoint_route(waypoint_route, self.perc_config.route_points)
      route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])[:self.perc_config.route_points]
      self.lon_logger.log_step(route)

    return result


  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
     
    if not ('hd_map' in input_data.keys()) and not self.initialized:
      control = carla.VehicleControl()
      control.steer = 0.0
      control.throttle = 0.0
      control.brake = 1.0
      self.initialized = True
      return control

    if self.config.debug and not self.init_map:
      self.nets[0].init_visualization()
      self.init_map = True

    tick_data = super().run_step(input_data, timestamp, plant=True)
    # ['lidar', 'rgb', 'rgb_augmented', 'semantics', 'semantics_augmented', 'depth', 'depth_augmented', 'bev_semantics', 'bev_semantics_augmented', 'bounding_boxes', 'pos_global', 'theta', 'speed', 'target_speed', 'target_point', 'target_point_next', 'command', 'next_command', 'aim_wp', 'route', 'steer', 'throttle', 'brake', 'control_brake', 'junction', 'vehicle_hazard', 'light_hazard', 'walker_hazard', 'stop_sign_hazard', 'stop_sign_close', 'walker_close', 'angle', 'augmentation_translation', 'augmentation_rotation', 'ego_matrix']
    
    tick_data_two = self.perc_plant_tick(input_data)
    # ['rgb', 'compass', 'lidar', 'speed']

    if self.config.debug:
      camera = input_data['rgb_debug'][1][:, :, :3]
      rgb_debug = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
      rgb_debug = np.transpose(rgb_debug, (2, 0, 1))

    target_point = torch.tensor(tick_data['target_point'], dtype=torch.float32).to(self.device).unsqueeze(0)

    # Preprocess route the same way we did during training
    route = tick_data['route']
    if len(route) < self.config.num_route_points:
      num_missing = self.config.num_route_points - len(route)
      route = np.array(route)
      # Fill the empty spots by repeating the last point.
      route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
    else:
      route = np.array(route[:self.config.num_route_points])

    if self.config.smooth_route:
      route = self.data.smooth_path(route)
    route = torch.tensor(route, dtype=torch.float32)[:self.config.num_route_points].to(self.device).unsqueeze(0)

    light_hazard = torch.tensor(tick_data['light_hazard'], dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)
    stop_sign_hazard = torch.tensor(tick_data['stop_sign_hazard'],
                                    dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)
    junction = torch.tensor(tick_data['junction'], dtype=torch.int32).to(self.device).unsqueeze(0).unsqueeze(0)

    if not PERC_DEBUG:
      bounding_boxes, _ = self.data.parse_bounding_boxes(tick_data['bounding_boxes'])
      bounding_boxes_padded = torch.zeros((self.config.max_num_bbs, 8), dtype=torch.float32).to(self.device)

      if len(bounding_boxes) > 0:
        # Pad bounding boxes to a fixed number
        bounding_boxes = np.stack(bounding_boxes)
        bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32).to(self.device)

        if bounding_boxes.shape[0] <= self.config.max_num_bbs:
          bounding_boxes_padded[:bounding_boxes.shape[0], :] = bounding_boxes
        else:
          bounding_boxes_padded[:self.config.max_num_bbs, :] = bounding_boxes[:self.config.max_num_bbs]

      bounding_boxes_padded = bounding_boxes_padded.unsqueeze(0)

    speed = torch.tensor(tick_data['speed'], dtype=torch.float32).to(self.device).unsqueeze(0)
    
    if PERC_DEBUG:
      from copy import deepcopy
      compute_debug_output = self.perc_config.debug and (not self.save_path is None)
      
      gt_velocity = tick_data['speed']
      velocity = gt_velocity.reshape(1, 1)  # used by transfuser


      lidar_indices = []
      for i in range(self.perc_config.lidar_seq_len):
        lidar_indices.append(i * self.perc_config.data_save_freq)


      if self.perc_config.backbone in ('aim'):  # Image only method
        raise ValueError('Not implemented')
      else:
        # Voxelize LiDAR and stack temporal frames
        lidar_bev = []
        # prepare LiDAR input
      
        for i in lidar_indices:
          # TODO: self.lidar_buffer, self.state_log, self.align_lidar
          lidar_point_cloud = deepcopy(self.lidar_buffer[i])

          # For single frame there is no point in realignment. The state_log index will also differ.
          if self.perc_config.realign_lidar and self.perc_config.lidar_seq_len > 1:
            raise ValueError('Not implemented')

          lidar_histogram = torch.from_numpy(
              self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                    use_ground_plane=self.perc_config.use_ground_plane)).unsqueeze(0)

          lidar_histogram = lidar_histogram.to(self.device, dtype=torch.float32)
          lidar_bev.append(lidar_histogram)

          lidar_bev = torch.cat(lidar_bev, dim=1)
      
      for i in range(self.perc_model_counts):
        if self.perc_config.backbone in ('transFuser', 'aim', 'bev_encoder'):
          pred_wp, \
          pred_target_speed, \
          pred_checkpoint, \
          pred_semantic, \
          pred_bev_semantic, \
          pred_depth, \
          pred_bb_features,\
          attention_weights,\
          pred_wp_1,\
          selected_path = self.perc_nets[i].forward(
            rgb=tick_data_two['rgb'],
            lidar_bev=lidar_bev,
            target_point=torch.FloatTensor(tick_data['target_point']).unsqueeze(0).to("cuda:0"), # tick_data['target_point'],
            ego_vel=torch.FloatTensor(velocity).to("cuda:0"), # tick_data['target_point'],# velocity,
            command=tick_data_two['command'])
          # Only convert bounding boxes when they are used.
          #if self.perc_config.detect_boxes and (compute_debug_output or self.perc_config.backbone in ('aim') or
          #                                self.stop_sign_controller):
          
          if self.step % 25 == 0:
            pass
            # print(f"Step: {self.step} \t {pred_wp}")
          # TODO: Might need NMS here
          pred_bounding_box = self.perc_nets[i].convert_features_to_bb_metric(pred_bb_features)
      
          # Filter bounding boxes
          pred_bounding_box = [box[:-1] for box in pred_bounding_box if box[-1] >= self.det_th ]
          pred_bounding_box_padded = torch.zeros((self.config.max_num_bbs, 8), dtype=torch.float32).to(self.device)

          if len(pred_bounding_box) > 0:
            # Pad bounding boxes to a fixed number
            pred_bounding_box = np.stack(pred_bounding_box)
            pred_bounding_box = torch.tensor(pred_bounding_box, dtype=torch.float32).to(self.device)

            if pred_bounding_box.shape[0] <= self.config.max_num_bbs:
              pred_bounding_box_padded[:pred_bounding_box.shape[0], :] = pred_bounding_box
            else:
              pred_bounding_box_padded[:self.config.max_num_bbs, :] = pred_bounding_box[:self.config.max_num_bbs]

          pred_bounding_box_padded = pred_bounding_box_padded.unsqueeze(0)
          
        else:
          raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

      # TODO: Ensure we extract the correct values even if the light gets excluded due to max_num_bbs
      pred_classes = [i[7].item() for i in pred_bounding_box_padded[0]]
      pred_light = [j == 2 for j in pred_classes]
      pred_stop_sign = [j == 3 for j in pred_classes]
      pred_light_hazard = any(pred_light)
      pred_stop_sign_hazard = any(pred_stop_sign)
      

      # Debugging Hazard Predictions

      if stop_sign_hazard or pred_stop_sign_hazard:
        if torch.any(stop_sign_hazard) == pred_stop_sign_hazard:
          # if self.step % 15 == 0:
          print(f"Step:{self.step} \t Hazard Stop: \t Correctly Predicted")                      
        else:
          print(f"ERROR: \t Step:{self.step} \t Hazard Stop: \t GT: {torch.any(stop_sign_hazard)}, \t PRED:{pred_stop_sign_hazard}")
          # print(pred_classes)
      if light_hazard or pred_light_hazard:
        if torch.any(light_hazard) == pred_light_hazard:
          # if self.step % 15 == 0:
          # print(f"Step:{self.step} \t Hazard Light: \t Correctly Predicted")
        else:
          print(f"ERROR: \t Step:{self.step} \t Hazard Light: \t GT: {torch.any(light_hazard)}, \t PRED:{pred_light_hazard}")
          # print(pred_classes)

    # Correct the Format:
    pred_light_hazard = torch.IntTensor([[pred_light_hazard]]).to("cuda:0")
    pred_stop_sign_hazard = torch.IntTensor([[pred_stop_sign_hazard]]).to("cuda:0")


    pred_wps = []
    pred_target_speeds = []
    pred_checkpoints = []
    pred_bbs = []
    for i in range(self.model_count):
      pred_wp, pred_target_speed, pred_checkpoint, pred_bb = self.nets[i].forward(bounding_boxes=pred_bounding_box_padded, # bounding_boxes_padded,
                                                                                  route=route, # pred_checkpoint,
                                                                                  target_point=target_point,
                                                                                  light_hazard=pred_light_hazard, # light_hazard,
                                                                                  stop_hazard=pred_stop_sign_hazard, # stop_sign_hazard,
                                                                                  junction=junction,
                                                                                  velocity=speed.unsqueeze(1))

      pred_wps.append(pred_wp)
      pred_bbs.append(t_u.plant_quant_to_box(self.config, pred_bb))
      if self.config.use_controller_input_prediction:
        pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
        pred_checkpoints.append(pred_checkpoint[0][1])

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    pred_bbs = torch.stack(pred_bbs, dim=0).mean(dim=0)

    if self.config.use_controller_input_prediction:
      pred_target_speed = torch.stack(pred_target_speeds, dim=0).mean(dim=0)
      pred_aim_wp = torch.stack(pred_checkpoints, dim=0).mean(dim=0)
      pred_aim_wp = pred_aim_wp.detach().cpu().numpy()
      pred_angle = -math.degrees(math.atan2(-pred_aim_wp[1], pred_aim_wp[0])) / 90.0

      if self.uncertainty_weight:
        uncertainty = pred_target_speed.detach().cpu().numpy()
        if uncertainty[0] > self.config.brake_uncertainty_threshold:
          pred_target_speed = self.config.target_speeds[0]
        else:
          pred_target_speed = sum(uncertainty * self.config.target_speeds)
      else:
        pred_target_speed_index = torch.argmax(pred_target_speed)
        pred_target_speed = self.config.target_speeds[pred_target_speed_index]

    if self.config.inference_direct_controller and \
        self.config.use_controller_input_prediction:
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_target_speed, pred_angle, speed, False)
    else:
      steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, speed, False)

    control = carla.VehicleControl()
    control.steer = float(steer)
    control.throttle = float(throttle)
    control.brake = float(brake)

    # Visualize the output of the last model
    if self.config.debug and (not self.save_path is None):
      self.nets[i].visualize_model(save_path=self.save_path,
                                   step=self.step,
                                   rgb=torch.tensor(rgb_debug),
                                   target_point=tick_data['target_point'],
                                   pred_wp=pred_wp,
                                   gt_wp=route,
                                   gt_bbs=bounding_boxes_padded,
                                   pred_speed=uncertainty,
                                   gt_speed=speed,
                                   junction=junction,
                                   light_hazard=light_hazard,
                                   stop_sign_hazard=stop_sign_hazard,
                                   pred_bb=pred_bbs)

    return control

  def destroy(self, results=None):
    del self.nets
    if PERC_DEBUG:
      del self.perc_nets
    super().destroy(results)