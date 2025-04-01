#!/usr/bin/env python3
"""
This script generates 10 versions of your hand-tracking code with different initial delays,
and saves them as:
    _traj6_100ms.py, _traj6_200ms.py, ..., _traj6_1000ms.py

It then zips all the files into generated_files.zip for easy download.
"""

import os
import zipfile

# The base template for your code.
# In the template, the placeholder {delay} will be replaced with the actual delay value,
# and the placeholder <DELAY_MS> will be replaced with a human‐readable delay in ms.
base_code = r'''#!/usr/bin/env python3
import csv
import time
import json
import roslibpy
import threading
from queue import Queue, Empty  # thread-safe queue

import pandas as pd

from HandTrackerRenderer import HandTrackerRenderer
import argparse

# --- Argument Parsing (unchanged) ---
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                    help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a: dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerEdge import HandTracker
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTracker import HandTracker

tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        use_handedness_average=not args.use_last_handedness,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

# --- End of Argument Parsing ---

# For temp buffer of hand pose when no hand
prev_data = ""

def telexr_post_hand_pose(hands):
    global prev_data
    if len(hands) > 0:
        # Extract x, y, z values in meters with six decimal places
        x = hands[0].xyz[0] / 1000
        y = hands[0].xyz[1] / 1000
        z = hands[0].xyz[2] / 1000

        # three zeros: if no movement detected, use the previous data
        if x == 0.0 and y == 0.0 and z == 0.0:
            return prev_data if prev_data != "" else None
        
        # irrational pose: if values are outside expected ranges, use previous data
        if abs(x) > 1.0 or abs(y) >= 1.0 or abs(z) >= 1.0:
            return prev_data if prev_data != "" else None

        formatted_data = f"[{x:.6f}, {y:.6f}, {z:.6f}, 0.0, 0.0, 0.0, 0.0]"
        prev_data = formatted_data
        print(f"X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f}")
        return formatted_data
    else:
        return prev_data if prev_data != "" else None

def telexr_read_and_post_hand_post(df, message_id):
    usr_pose = df['usr_pose'][message_id]
    return usr_pose

def init_sent():
    count = 0 
    # Wait until at least one hand is detected or a maximum number of frames is reached.
    while True:
        frame, hands, bag = tracker.next_frame()
        if len(hands) > 0 or count >= 1000: # 500 is 5s (default), 1000 is 10s for autorun
            break
        else:
            if frame is None:
                break
            frame = renderer.draw(frame, hands, bag)
            # Use the first entry from the CSV during init.
            data_string = df['usr_pose'][0]
            time1_string = df['Timestamp'][0]
            message_data = {
                'message_id': 0,
                'fused_pose': data_string,
                'time1': time1_string
            }
            with open("hand_data", "w") as file:
                file.write(data_string + "\n")
            
            # Publish the initial hand data
            if message_id % 1 == 0:
                talker.publish(roslibpy.Message({'data': json.dumps(message_data)}))
                print(f"Init Sent: ID {message_id}, data: {data_string}")
            count += 1
            time.sleep(0.01)

# --- Setup ROS Communication ---
with open("xr_ip", "r") as file:
    xr_ip = file.read().strip()
port = 9090
message_id = 0

client = roslibpy.Ros(host=xr_ip, port=port)
client.run()
talker = roslibpy.Topic(client, '/chatter', 'std_msgs/String')

# Read trajectory CSV (adjust the path as needed)
df = pd.read_csv('../../data/traj_12_6/traj6.csv')  # change here for different traj

# --- Create a thread-safe Queue for poses ---
from queue import Queue, Empty
pose_queue = Queue()

# Flag to indicate when to stop the transmission thread
exit_flag = False

def transmission_thread():
    # Initial delay (<DELAY_MS>)
    time.sleep({delay})
    while not exit_flag or not pose_queue.empty():
        try:
            # Block for a short time waiting for an item
            message_data = pose_queue.get(timeout=0.1)
            talker.publish(roslibpy.Message({'data': json.dumps(message_data)}))
            print(f"Sent from transmission thread: ID {message_data['message_id']}, data: {message_data['fused_pose']}")
            pose_queue.task_done()
        except Empty:
            continue

# Start the transmission thread
import threading
tx_thread = threading.Thread(target=transmission_thread, daemon=True)

output_file = 'data.csv'
data_log = []
lst_exe_time = []

try:
    init_sent()
    tx_thread.start()
    # Main loop: hand tracking & buffering poses
    while True:
        time1 = time.time()
        frame, hands, bag = tracker.next_frame()
        if frame is None:
            break
        frame = renderer.draw(frame, hands, bag)
        
        if message_id < len(df) - 1:
            # Use the CSV data or use telexr_post_hand_pose(hands) if you want to use the live hand data
            data_string = telexr_read_and_post_hand_post(df, message_id)
        else:
            break

        if data_string is not None:
            message_id += 1
            message_data = {
                'message_id': message_id,
                'fused_pose': data_string,
                'time1': time1
            }
            
            # Save locally to file
            with open("hand_data", "w") as file:
                file.write(data_string + "\n")
            
            # Instead of sending immediately, push the message into the queue
            pose_queue.put(message_data)
            print(f"Buffered: ID {message_id}, data: {data_string}")
            
            exe_time = time.time() - time1
            lst_exe_time.append(exe_time)
            data_log.append([time1, message_id, data_string])
        
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    # Signal the transmission thread to exit
    exit_flag = True
    # Wait for all buffered items to be sent
    pose_queue.join()
    print("\nSaving data to CSV...")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "msg_id", "usr_pose"])
        writer.writerows(data_log)
    print(f"Data saved to {output_file}. Exiting...")
    if lst_exe_time:
        print(f"Average execution time: {sum(lst_exe_time)/len(lst_exe_time)}")
    renderer.exit()
    tracker.exit()
'''

# Define the delays for the 10 files (in seconds). Here we generate files for 0.1 to 1.0 seconds.
delays = [0.1 * i for i in range(0, 11)]  # [0.1, 0.2, ..., 1.0]

# Create an output directory for the generated files
output_dir = "traj6"
os.makedirs(output_dir, exist_ok=True)

filenames = []

for delay in delays:
    delay_ms = int(delay * 1000)  # convert to milliseconds
    filename = f"_traj6_{delay_ms}ms.py"
    filenames.append(filename)
    file_content = base_code.replace("{delay}", str(delay)).replace("<DELAY_MS>", f"{delay_ms}ms")
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(file_content)

# # Now, create a ZIP archive containing all the generated files.
# zip_filename = "generated_files.zip"
# with zipfile.ZipFile(zip_filename, "w") as zipf:
#     for file in filenames:
#         zipf.write(os.path.join(output_dir, file), arcname=file)

# print(f"Zip file '{zip_filename}' created with the following files:")
# for file in filenames:
#     print("  -", file)
