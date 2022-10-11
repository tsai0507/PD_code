import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
#test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
test_scene = "apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height":1.5 ,  # Height of sensors in meters, relative to the agent1.5 /1:27
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)(- is down,+is up)
}


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8) #0~255
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    # a BEV_RGB visual sensor, to the agent
    bev_rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    bev_rgb_sensor_spec.uuid = "bev_color_sensor"
    bev_rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    bev_rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    bev_rgb_sensor_spec.position = [0.0, settings["sensor_height"], -1.5]
    bev_rgb_sensor_spec.orientation = [
        -np.pi/2,
        0.0,
        0.0,
    ]
    bev_rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec,bev_rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([0.0, 0.0, 0.0])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
SAVE_BEV_IMG="z"
FINISH="f"
print("#############################")
print("use keyboard to control the agent")
print(" w for go forward  ")
print(" a for turn left  ")
print(" d for trun right  ")
print(" f for finish and quit the program")
print("#############################")


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)
        cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("BEV_RGB", transform_rgb_bgr(observations["bev_color_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        camera_locate=[sensor_state.position[0],sensor_state.position[1],sensor_state.position[2]]

        #回傳front_rgb,bev_rgb,front_depth,座標
        return transform_rgb_bgr(observations["color_sensor"]) ,transform_rgb_bgr(observations["bev_color_sensor"]) ,transform_depth(observations["depth_sensor"]),camera_locate

#用來單次儲存BEV_project所需的兩張照片(front_rgb,bev_rgb)
def bev_save_image(save_img):
    cv2.imwrite('front_view_path.png',save_img[0])
    cv2.imwrite('top_view_path.png',save_img[1])

#相機座標每移動一次就將存下front_rgb,depth_rgb,讓reconstruct來重建模型
def save_reconstruct(save_img,count):
    #用count來命名照片,才能有多張照片資料
    cv2.imwrite('./reconstuct_data/'+'rgb_'+str(count)+'.png',save_img[0])
    cv2.imwrite('./reconstuct_data/'+'img1_depth'+str(count)+'.png',save_img[2])

#因為在reconstruct需要有ground truth路線,我們把每次相機座標存到camera_path.txt
def save_camera_locate(locat):
    f.write(str(locat[0])+'\n')
    f.write(str(locat[1])+'\n')
    f.write(str(locat[2])+'\n')
### 前置作業 ###
path = './reconstuct_data/camera_path.txt'      #開啟txt檔路徑
f = open(path, 'w')                          
count=1                                         #幫照片編號

#進while之前先走一步
#每走一部都需要：
action = "move_forward"
save_img=navigateAndSee(action)                 #回傳照片資料與座標
save_reconstruct(save_img,count)                #存下reconstruct照片
count=count+1                                   #更新count
save_camera_locate(save_img[3])                 #存下座標

while True:
    keystroke = cv2.waitKey(0)                  #等待按鍵事件
    if keystroke == ord(FORWARD_KEY):           #ord()取得char得ASCII
        action = "move_forward"
        save_img=navigateAndSee(action)         #回傳照片資料與座標
        save_reconstruct(save_img,count)        #存下reconstruct照片
        count=count+1                           #更新count
        save_camera_locate(save_img[3])         #存下座標
        print("action: FORWARD")
    elif keystroke == ord(LEFT_KEY):
        action = "turn_left"
        save_img=navigateAndSee(action)
        save_reconstruct(save_img,count) 
        count=count+1
        save_camera_locate(save_img[3])     
        print("action: LEFT")     
    elif keystroke == ord(RIGHT_KEY):
        action = "turn_right"
        save_img=navigateAndSee(action)
        save_reconstruct(save_img,count) 
        count=count+1  
        save_camera_locate(save_img[3])     
        print("action: RIGHT")  
    elif keystroke == ord(SAVE_BEV_IMG): 
        bev_save_image(save_img)                #存下BEV_projection所需照片
        print("action: SAVE_BEV_IMG")
    elif keystroke == ord(FINISH):
        count=count-1
        print("action: FINISH ")
        print("All image number: ",count)
        f.close()                               
        break
    else:
        print("INVALID KEY")
        continue
