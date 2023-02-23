import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from skimage.feature import match_template, peak_local_max
from skimage.transform import AffineTransform

from models import UnetRoad
from line_intersection import line_intersection_extraction


def rotate_image_by_yaw(img, yaw):
    matrix = cv2.getRotationMatrix2D(scale=1, angle=yaw, center=(0, 0))
    rotation = AffineTransform(matrix=np.vstack([matrix, np.array([[0, 0, 1]])]))

    rotated_points = np.array([
        rotation(point)[0]
        for point in [[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]]
    ])
    new_height = int(rotated_points[:, 1].max() - rotated_points[:, 1].min())
    new_width = int(rotated_points[:, 0].max() - rotated_points[:, 0].min())

    matrix[1, 2] = - rotated_points[:, 1].min()
    matrix[0, 2] = - rotated_points[:, 0].min()

    return cv2.warpAffine(img, matrix, (new_width, new_height))


if __name__ == "__main__":

    # MAP
    #img_map = cv2.cvtColor(cv2.imread('map_image_s5.jpg'), cv2.COLOR_BGR2RGB)
    mask_map = cv2.imread('map_feat_road_s5.jpg', 0)

    # get crosses on map roads
    #_, inter_points_map = line_intersection_extraction(mask_map)
    #kp_map = [cv2.KeyPoint(float(i), float(j), 1) for i, j in inter_points_map]

    # map coordinates
    pm0 = (35.0875854492188, 48.5965922514567)
    pm1 = (35.1617431640625, 48.5965922514567)
    pm2 = (35.1617431640625, 48.5384317740504)
    pm3 = (35.0875854492188, 48.5384317740504)

    step_x = (pm1[0]-pm0[0])/mask_map.shape[1]
    step_y = (pm1[1]-pm2[1])/mask_map.shape[0]
    print('Map degrees in px:', step_x, step_y)

    # init road segmentation model
    model_cls = UnetRoad('weights/uav_road_segm_efn0.pt')

    # load video and logs
    vid = cv2.VideoCapture('../sample5/full_video.mp4')

    log_df = pd.read_csv('tlogs_sample5e.csv')
    log_df_gps = log_df[log_df.source.str.contains('GPS_RAW')]
    log_df_att = log_df[log_df.source == 'ATTITUDE']
    log_df_mnt = log_df.fillna(method='ffill')[log_df.source == 'MOUNT_STATUS']

    # time sync constant between logs and video
    tsync = 759.411

    # time range for processing
    start_time = 9*60
    end_time = 11*60+30

    fps = vid.get(cv2.CAP_PROP_FPS)
    width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('frames per second =', fps, 'shape:', width, height)

    step_sec = 5
    step = fps*step_sec
    current_time = start_time
    # setup start frame
    frame_id = int(fps*start_time)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    print('Start process video...')
    while vid.isOpened():
        ret, frame = vid.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, ((width//32)*32, (height//32)*32))
        # infer camera frame
        mask = model_cls.infer_single_image(frame)
        # get crosses on camera roads
        lines_cam, inter_points_cam = line_intersection_extraction(mask)

        information_row = log_df_gps.iloc[
            np.argmin(
                np.abs(
                    log_df_gps['time'].to_numpy() - (current_time+tsync)
                )
            )
        ].values
        yaw = log_df_att.iloc[np.argmin(np.abs(log_df_att['time'].to_numpy() - (current_time+tsync)))].yaw

        # apply yaw to map
        mask_map_rotated = rotate_image_by_yaw(mask_map, yaw/np.pi*180)
        #img_map_rotated = rotate_image_by_yaw(img_map, yaw/np.pi*180)

        # template matching
        tm_res = match_template(mask_map_rotated, mask, pad_input=True, constant_values=0)
        peaks = peak_local_max(tm_res,
                               min_distance=50,
                               threshold_abs=0.1,
                               threshold_rel=0.5,
                               num_peaks=5)

        kp_cam = [cv2.KeyPoint(float(i), float(j), 1) for i, j in inter_points_cam]
        print('Vis')
        plt.figure()
        plt.subplot(2,3,1)
        plt.imshow(frame)
        plt.subplot(2,3,2)
        plt.imshow(mask)

        tmp_img = cv2.drawKeypoints(frame, kp_cam, None, color=(255,0,0))
        if lines_cam is not None:
            for i in range(0, len(lines_cam)):
                l = lines_cam[i][0]
                cv2.line(tmp_img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv2.LINE_AA)

        plt.subplot(2,3,3)
        plt.imshow(tmp_img)

        #plt.subplot(2,3,4)
        #plt.imshow(img_map_rotated[4000:6800, 3000:6000])
        plt.subplot(2,3,5)
        #mask_map_rotated = cv2.drawKeypoints(mask_map_rotated, kp_map, None, color=(255,0,0))
        plt.imshow(mask_map_rotated)
        plt.scatter(peaks[:, 0], peaks[:, 1], marker='.')

        plt.subplot(2,3,6)
        plt.imshow(tm_res[4000:6800, 3000:6000])


        # move to further processing step
        frame_id += step
        current_time += step_sec
        if frame_id > end_time*fps:
            break
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        plt.show()
