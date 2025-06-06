#!/usr/bin/env python3
import csv
import time
import json
import roslibpy

from HandTrackerRenderer import HandTrackerRenderer
import argparse

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
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

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

# for temp buffer of hand pose when no hand
prev_data = ""

def telexr_post_hand_pose(hands):
    global prev_data
    if len(hands) > 0:
        # Extract x, y, z values in meters with six decimal places
        x = hands[0].xyz[0] / 1000
        y = hands[0].xyz[1] / 1000
        z = hands[0].xyz[2] / 1000

        # three zeros
        if x == 0.0 and y == 0 and z == 0:
            if prev_data!="":
                return prev_data
            else:
                return None
        
        # irrational pose
        if abs(x) > 1.0 or abs(y) >= 1.0 or abs(z) >= 1.0:
            if prev_data!="":
                return prev_data
            else:
                return None

        # Format the data - rotation neglect
        formatted_data = f"[{x:.6f}, {y:.6f}, {z:.6f}, 0.0, 0.0, 0.0, 0.0]"
        # formatted_data = f"[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]" # <RTEN> test
        prev_data = formatted_data

        # Optionally, print for verification
        print(f"X: {x:.6f}, Y: {y:.6f}, Z: {z:.6f}")

        return formatted_data
    else:
        if prev_data!="":
            return prev_data
        else:
            return None

# setup the client talker
xr = '10.13.145.127'
# xr = 'localhost'
port = 9090
message_id = 0

output_file = 'data.csv'
data_log = []

client = roslibpy.Ros(host=xr, port=port)
client.run()

talker = roslibpy.Topic(client, '/chatter', 'std_msgs/String')

try:
    ##=======================If need frame output=======================
    while True:
        # start timer: time1
        time1 = time.time()

        # Run hand tracker on next frame
        # 'bag' contains some information related to the frame 
        # and not related to a particular hand like body keypoints in Body Pre Focusing mode
        # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
        frame, hands, bag = tracker.next_frame()
        if frame is None: break
        # Draw hands
        frame = renderer.draw(frame, hands, bag)
        # <RTEN> telexr hand pose added ---------------------
        data_string = telexr_post_hand_pose(hands)

        if data_string != None:
            message_id += 1
            message_data = {
                'message_id': message_id,
                'fused_pose': data_string,
                'time1': time1
            }

            # Write to the file "hand_data"
            with open("hand_data", "w") as file:
                file.write(data_string + "\n")
            
            # publish the hand data for robot to subscribe
            # sent the hand_data
            if message_id % 1 == 0: # <RTEN> control packet lost 25% with %4 != 0
                talker.publish(roslibpy.Message({'data': json.dumps(message_data)}))
                print(f"Sent: ID {message_id}, data: {data_string}")
            
            # log data
            print("exe time:", time.time() - time1)
            data_log.append([time1, message_id, data_string])
        # ---------------------------------------------------
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break
    renderer.exit()
    tracker.exit()
    ##=======================If Headless mode=======================
    # while True:
    #     # start timer: time1
    #     time1 = time.time()

    #     # Run hand tracker on next frame
    #     # 'bag' contains some information related to the frame 
    #     # and not related to a particular hand like body keypoints in Body Pre Focusing mode
    #     # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
    #     frame, hands, bag = tracker.next_frame()

    #     # <RTEN> telexr hand pose added ---------------------
    #     data_string = telexr_post_hand_pose(hands)
    #     if data_string != None:
    #         message_id += 1
    #         message_data = {
    #             'message_id': message_id,
    #             'fused_pose': data_string,
    #             'time1': time1
    #         }

    #         # Write to the file "hand_data"
    #         with open("hand_data", "w") as file:
    #             file.write(data_string + "\n")

    #         # publish the hand data for robot to subscribe
    #         # sent the hand_data
    #         if message_id % 1 != 0: # <RTEN> control packet lost 25% with %4 != 0
    #             talker.publish(roslibpy.Message({'data': json.dumps(message_data)}))
    #             print(f"Sent: ID {message_id}, data: {data_string}")

    #         # print("exe time:", time.time() - time1)
    #         # log data
    #         data_log.append([time1, message_id, data_string])
        # ---------------------------------------------------
except Exception as e:
    print(f"Error: {e}")
finally:
    print("\nSaving data to CSV...")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "msg_id", "usr_pose"])
        writer.writerows(data_log)
    print(f"Data saved to {output_file}. Exiting...")
    renderer.exit()
    tracker.exit()