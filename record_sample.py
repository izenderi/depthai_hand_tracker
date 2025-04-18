from depthai_sdk import OakCamera, RecordType
import depthai as dai

FPS = 30
with OakCamera() as oak:
    color = oak.create_camera('CAM_A', resolution='1080p', encode='mjpeg', fps=FPS)
    color.config_color_camera(isp_scale=(2,3)) # 720P color frames, mjpeg

    stereo = oak.create_stereo(resolution='720p', fps=FPS)
    stereo.config_stereo(align=color, subpixel=True, lr_check=True)
    stereo.node.setOutputSize(640, 360) # 720p, downscaled to 640x360 (decimation filter, median filtering)
    # On-device post processing for stereo depth
    config = stereo.node.initialConfig.get()
    stereo.node.setPostProcessingHardwareResources(3, 3)
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = 10_000 # 10m
    config.postProcessing.decimationFilter.decimationFactor = 2
    config.postProcessing.decimationFilter.decimationMode = dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
    stereo.node.initialConfig.set(config)
    """
    Depth will get encoded on-host with FFV1 codec (it supports 16bit grey), and color will be
    encoded on-device with MJPEG codec. FFV1 works well with .avi container, so depth video will be
    saved as .avi, and color video will be saved as .mp4.
    """
    record_components = [stereo.out.depth, color.out.encoded]
    oak.record(record_components, './', record_type=RecordType.VIDEO).configure_syncing(True, threshold_ms=500/30)
    oak.start(blocking=True)

# ====================================================================================================================

# from depthai_sdk import OakCamera, RecordType
# from depthai_sdk.args_parser import ArgsParser
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--recordStreams', action='store_true', help="Record frames to file")
# parser.add_argument('--saveStreamsTo', type=str, help="Save frames to directory", default="./record")
# args= ArgsParser.parseArgs(parser=parser)

# with OakCamera(args=args) as oak:
#     cams = oak.create_all_cameras()
#     left = oak.camera('left')
#     right = oak.camera('right')
#     if left is not None and right is not None:
#         stereo = oak.create_stereo(left=left, right=right)
#         oak.visualize(stereo)
#     # Sync & save all streams
#     if args["recordStreams"]:
#         oak.record(cams, args["saveStreamsTo"], RecordType.VIDEO_LOSSLESS).configure_syncing(True, 50)
#     oak.visualize(cams, fps=True)

#     oak.start(blocking=True)

# ====================================================================================================================
# from depthai_sdk import OakCamera, RecordType

# with OakCamera() as oak:
#     color = oak.create_camera('color', resolution='1080P', fps=10, encode='H265')
#     left = oak.create_camera('left', resolution='800p', fps=10, encode='H265')
#     right = oak.create_camera('right', resolution='800p', fps=10, encode='H265')

#     stereo = oak.create_stereo(left=left, right=right)
#     nn = oak.create_nn('mobilenet-ssd', color, spatial=stereo)

#     # Sync & save all (encoded) streams
#     oak.record([color.out.encoded, left.out.encoded, right.out.encoded], './record', RecordType.VIDEO) \
#         .configure_syncing(enable_sync=True, threshold_ms=50)

#     oak.visualize([color.out.encoded], fps=True)

#     oak.start(blocking=True)