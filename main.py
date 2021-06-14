import cv2
import data_process
import os

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

root_dir = './trainval/'
save_root = os.getcwd() + '/iteration1/'
result_file = os.getcwd() + '/iteration1/val_result/'
if __name__ == '__main__':
    names, root_dirs, sizes, locators = data_process.pre_process()

    # 设置追踪器
    # 除了MIL外还可以使用下面类型
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']
    tracker_type = tracker_types[1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

    for i in range(len(names)):
        dir = root_dirs[i]
        video_size = sizes[i]
        locator = locators[i]
        # frame_path是每一帧视频图片的列表
        frame_path = [dir + '0' * (8 - len(str(index + 1))) + str(index + 1) + '.jpg' for index in range(video_size)]
        x = [locator[index] for index in range(0, len(locator), 2)]
        y = [locator[index] for index in range(1, len(locator), 2)]
        # 定义初始边界框
        # 起点坐标 + 大小
        bbox = (min(x), min(y), max(x) - min(x), max(y) - min(y))
        first_frame = cv2.imread(frame_path[0])
        # 标记bbox
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(first_frame, p1, p2, (255, 0, 0), 2, 1)
        save_path = save_root + frame_path[0][2:]
        print(save_path[:-12])
        if not os.path.exists(path=save_path[:-12]):
            os.makedirs(save_path[:-12])
        cv2.imwrite(save_path, first_frame)
        tracker.init(first_frame, bbox)
        # 记录修正的数据的历史
        fix_log_x = []
        fix_log_y = []
        bbox_log = [bbox]
        frame_log = [first_frame]
        lost_log = [False]
        # 保存结果
        if not os.path.exists(path=result_file):
            os.makedirs(result_file)
        result_path = result_file + names[i] + ".txt"
        with open(result_path, mode="w") as f:
            f.write(str(bbox[0]) + "," + str(bbox[1]) + ",")
            f.write(str(bbox[0] + bbox[2]) + "," + str(bbox[1]) + ",")
            f.write(str(bbox[0]) + "," + str(bbox[1] + bbox[3]) + ",")
            f.write(str(bbox[0] + bbox[2]) + "," + str(bbox[1] + bbox[3]) + "\n")
        # 循环读帧
        for index in range(len(frame_path)):
            if index == 0:
                continue
            frame = cv2.imread(frame_path[index])
            ok, (tmp_bbox) = tracker.update(frame)
            tmp_bbox = list(tmp_bbox)
            bbox = list(bbox)
            # 用以前的记录对其进行修正
            fix_log_x.append(tmp_bbox[0] - bbox[0])
            fix_log_y.append(tmp_bbox[1] - bbox[1])
            bbox[0] += tmp_bbox[0] - bbox[0]
            bbox[1] += tmp_bbox[1] - bbox[1]
            # # 迭代三每一帧都重新初始化追踪器并且采用历史查找
            # tracker.init(frame, bbox)
            # bbox_log.append(bbox)
            # frame_log.append(frame)
            # # 检查修正历史，如果长时间(2次)无任何修正，说明框体已经走远，重新追踪
            # print(fix_log_x[len(fix_log_x) - 2:len(fix_log_x)], fix_log_y[len(fix_log_y) - 2:len(fix_log_y)])
            # if abs(fix_log_x[len(fix_log_x) - 1]) <= 1 and abs(fix_log_y[len(fix_log_y) - 1]) <= 1:
            #     print("目标丢失")
            #     lost_log.append(True)
            #     # 如果连续两次丢失目标，则重新捕获（从前往后）
            #     if lost_log[-1] and lost_log[-2]:
            #         for j in range(0, len(bbox_log)):
            #             tracker.init(frame_log[j], bbox_log[j])
            #             ok, try_bbox = tracker.update(frame)
            #             if ok:
            #                 print("目标重新捕获(正序)")
            #                 bbox = try_bbox
            #                 break
            #     # 丢失目标（非连续两次）从后往前查找历史
            #     else:
            #         for j in range(len(bbox_log) - 1, -1, -1):
            #             tracker.init(frame_log[j], bbox_log[j])
            #             ok, try_bbox = tracker.update(frame)
            #             if ok:
            #                 print("目标重新捕获（倒序）")
            #                 bbox = try_bbox
            #                 break
            #     # 拿出历史追踪器寻找
            # else:
            #     lost_log.append(False)
            # 标记bbox
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            save_path = save_root + frame_path[index][2:]
            if not os.path.exists(path=save_path[:-12]):
                os.makedirs(save_path[:-12])
            cv2.imwrite(save_path, frame)
            # 保存结果
            result_path = result_file + names[i] + ".txt"
            with open(result_path, mode="w") as f:
                f.write(str(bbox[0]) + "," + str(bbox[1]) + ",")
                f.write(str(bbox[0] + bbox[2]) + "," + str(bbox[1]) + ",")
                f.write(str(bbox[0]) + "," + str(bbox[1] + bbox[3]) + ",")
                f.write(str(bbox[0] + bbox[2]) + "," + str(bbox[1] + bbox[3]) + "\n")
