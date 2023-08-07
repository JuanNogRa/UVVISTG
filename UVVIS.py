# Import the required module for text 
# to speech conversion
from gtts import gTTS
from io import BytesIO
# import the opencv library
import cv2
import numpy as np
import math
import os
import configparser
import sys
import wget
import argparse
import pygame


def on_mouse_display_depth_value(event, x, y, flags, params):
    # when the left mouse button has been clicked
    if (event == cv2.EVENT_LBUTTONDOWN):

    # unpack the set of parameters
        f, B, disparity_scaled = params
        
        # safely calculate the depth and display it in the terminal
        if (disparity_scaled[y,x] > 0):
            depth = f * (B / disparity_scaled[y,x])
            changeInX = 626.35 - x
            changeInY = 352.393 - y
            theta_angle= np.degrees(math.atan2(changeInY,changeInX))
        else:
            depth = 0
            theta_angle=0
            phi_angle=0
        # as the calibration for the ZED camera is in millimetres, divide
        # by 1000 to get it in metres
        print("Profundidad @ (" + str(x) + "," + str(y) + "): " + '{0:.2f}'.format(depth / 1000) + "m"+" Angulo delta: "+'{0:1d}'.format(int(theta_angle)))
        tts = gTTS(text = "Profundidad " + '{0:.2f}'.format(depth / 1000) + "m"+" Angulo delta: "+'{0:1d}'.format(int(theta_angle)), lang='es', slow=False)
        textTovoice(tts)
def textTovoice(tts) :
    # convert to file-like object
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            #--- play it ---
            pygame.init()
            pygame.mixer.init()
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
def download_calibration_file(serial_number) :
    if os.name == 'nt' :
        hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
    else :
        hidden_path = '/usr/local/zed/settings/'
    calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'

    if os.path.isfile(calibration_file) == False:
        url = 'http://calib.stereolabs.com/?SN='
        filename = wget.download(url=url+str(serial_number), out=calibration_file)

        if os.path.isfile(calibration_file) == False:
            print('Invalid Calibration File')
            return ""

    return calibration_file
  
def init_calibration(calibration_file, image_size) :

    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208 :
        resolution_str = '2K'
    elif image_size.width == 1920 :
        resolution_str = 'FHD'
    elif image_size.width == 1280 :
        resolution_str = 'HD'
    elif image_size.width == 672 :
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                   float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])


    left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)


    right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)

    R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                         [0, left_cam_fy, left_cam_cy],
                         [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                          [0, right_cam_fy, right_cam_cy],
                          [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Resolution :
    width = 1280
    height = 720
def main() :

    if len(sys.argv) == 1 :
        print('Please provide ZED serial number')
        exit(1)

    # Open the ZED camera
    cap = cv2.VideoCapture(1,cv2.CAP_MSMF)
    if cap.isOpened() == 0:
        exit(-1)

    image_size = Resolution()
    image_size.width = 1280
    image_size.height = 720

    # Set the video resolution to HD720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)

    serial_number = int(sys.argv[1])
    calibration_file = download_calibration_file(serial_number)
    if calibration_file  == "":
        exit(1)
    print("Calibration file found. Loading...")

    camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)
#     max_disparity = 128
#     window_size = 21
#     wls_lmbda = 800
#     wls_sigma = 1.2
    max_disparity = 128
    window_size = 21
    wls_lmbda = 800
    wls_sigma = 1.2
    windowNameD = "right RECT" # window name
    
    while True :
        # Get a new frame from camera
        retval, frame = cap.read()
        # Extract left and right images from side-by-side
        left_right_image = np.split(frame, 2, axis=1)
        # Display images
        #cv2.imshow("left RAW", left_right_image[0])

        left_rect = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(windowNameD, right_rect)
        
        if cv2.waitKey(3) >= 0 :
            print('disparity map calculing...',)
            # Passing the text and language to the engine, 
            # here we have marked slow=False. Which tells 
            # the module that the converted audio should 
            # have a high speed
            tts = gTTS(text = 'Calculando Mapa de Disparidad', lang='es', slow=False)
            textTovoice(tts)
            stereoSGBM = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities = max_disparity, # max_disp has to be dividable by 16 f. E. HH 192, 256
#             blockSize=window_size,
#             P1=8 * window_size ** 2,       # 8*number_of_image_channels*SADWindowSize*SADWindowSize
#             P2=32 * window_size ** 2,      # 32*number_of_image_channels*SADWindowSize*SADWindowSize
#             disp12MaxDiff=1,
#             uniquenessRatio=15,
#             speckleWindowSize=0,
#             speckleRange=2,
#             preFilterCap=63,
#             mode=cv2.STEREO_SGBM_MODE_HH
            )
        
            disparity_SGBM = stereoSGBM.compute(left_rect, right_rect)
        
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereoSGBM)
            wls_filter.setLambda(wls_lmbda)
            wls_filter.setSigmaColor(wls_sigma)

            # remember to convert to grayscale (as the disparity matching works on grayscale)

            grayL = cv2.cvtColor(left_rect,cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right_rect,cv2.COLOR_BGR2GRAY)
        
            # perform preprocessing - raise to the power, as this subjectively appears
            # to improve subsequent disparity calculation

            grayL = np.power(grayL, 0.75).astype('uint8')
            grayR = np.power(grayR, 0.75).astype('uint8')
        
            left_matcher=stereoSGBM
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        
            displ = left_matcher.compute(cv2.UMat(grayL),cv2.UMat(grayR))  # .astype(np.float32)/16
            dispr = right_matcher.compute(cv2.UMat(grayR),cv2.UMat(grayL))  # .astype(np.float32)/16
            displ = np.int16(cv2.UMat.get(displ))
            dispr = np.int16(cv2.UMat.get(dispr))
            disparity = wls_filter.filter(displ, grayL, None, dispr)
            
            # scale the disparity to 8-bit for viewing
            # divide by 16 and convert to 8-bit image (then range of values should
            # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
            # so we fix this also using a initial threshold between 0 and max_disparity
            # as disparity=-1 means no disparity available
            _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
            disparity_scaled = (disparity / 16.).astype(np.uint8)
            cv2.setMouseCallback(windowNameD,on_mouse_display_depth_value, (698.83,120.004,disparity_scaled))
            cv2.imshow("Depth Map",disparity_scaled)
    exit(0)

if __name__ == "__main__":
    main()
