# coding: utf-8

from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2
import time
import multiprocessing as mp
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize, letterbox_resize1
from args import *
import os
global gol_savepath
global gol_savestate

# mp.set_start_method(method='spawn')
def cam_queue_img_put(q_put, q_get, cap_num,gol_savestate,gol_savepath,vedioname):
    # cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    cap = cv2.VideoCapture(vedioname)
    video_fps = int(cap.get(5))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    # frame_time = 1 / video_fps
    index = 0
    savename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'-cam-%d'% cap_num
    if gol_savestate:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        savepath = gol_savepath + '/' + savename +'.mp4'
        videoWriter = cv2.VideoWriter(savepath, fourcc, video_fps, (video_width, video_height))

#warningpart
    codepath = os.getcwd()
    waringpath = codepath + '/' + 'waring'
    datepath = waringpath + '/' + time.strftime("%Y-%m-%d", time.localtime())
    cap_path = datepath + '/' + 'cam-%d' % cap_num
    pic_cnt = 0
    if not os.path.exists(waringpath) and os.path.exists(datepath):
        os.makedirs(waringpath)
        os.makedirs(datepath)
    if not os.path.exists(cap_path):
        os.makedirs(cap_path)

    while True:
        index += 1
        is_opened, frame = cap.read()
        frame,_,_,_ = letterbox_resize1(frame,480,360)
        q_put.put(frame) if is_opened else None
        q_put.get() if q_put.qsize() > 1 else None  # 保证队列里只有一张图片
        while q_get.empty() and index == 1:
            time.sleep(0.01)
        time.sleep(0.05)
        # time.sleep(frame_time-0.00)
        if not q_get.empty():
            (label_id, score, boxes, fps) = q_get.get()
            # img = np.copy(origin_img)
            no_helmet = np.sum(label_id == 1)
            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                plot_one_box(frame, [x0, y0, x1, y1], label=args.classes[label_id[i]] + ', {:.2f}'.format(score[i]),
                             color=color_table[label_id[i]])
            cv2.putText(frame, '{:.2f}fps'.format(fps), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            if no_helmet > 0:
                cv2.putText(frame, 'no_helmet:%d' % no_helmet, (0, 320), 0,
                            fontScale=1, color=(0, 0, 255), thickness=2)
                pic_cnt += 1
                if pic_cnt>200:
                    pic_num = time.strftime("%H-%M-%S",time.localtime())+ '-nohelmet_num-%d'% cap_num
                    path = cap_path + '/' + pic_num +'.jpg'
                    #cv2.imwrite("C:/Pycharm/Projects/ScreenShot/Fault/{}.jpg".pic_num, frame)
                    cv2.imwrite(path, frame)
                    pic_cnt=0
            # cv2.imshow('NO.%d cam' % cap_num, frame)
        else:
            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                plot_one_box(frame, [x0, y0, x1, y1],
                             label=args.classes[label_id[i]] + ', {:.2f}'.format(score[i]),
                             color=color_table[label_id[i]])
            cv2.putText(frame, '{:.2f}fps'.format(fps), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            if no_helmet > 0:
                cv2.putText(frame, 'no_helmet:%d' % no_helmet, (0, 320), 0,
                            fontScale=1, color=(0, 0, 255), thickness=2)
                pic_cnt+=1
                if pic_cnt>200:
                    pic_num = time.strftime("%H-%M-%S",time.localtime())+ '-nohelmet_num-%d'% cap_num
                    path = cap_path + '/' + pic_num +'.jpg'
                    #cv2.imwrite("C:/Pycharm/Projects/ScreenShot/Fault/{}.jpg".pic_num, frame)
                    cv2.imwrite(path, frame)
                    pic_cnt=0
        cv2.imshow('NO.%d cam' % cap_num, frame)
        cap_Window(cap_num)
        if gol_savestate:
            videoWriter.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('NO.%d cam' % cap_num, cv2.WND_PROP_AUTOSIZE) < 1:
            break
    cap.release()
    if gol_savestate:
        videoWriter.release()
    cv2.destroyAllWindows()


def ved_queue_img_put(q_put, q_get, img_path,gol_savestate,gol_savepath):
    vid = cv2.VideoCapture(img_path)
    video_frame_cnt = int(vid.get(7))
    video_fps = int(vid.get(5))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    vidname = img_path.split('/')[-1]
    delay = 1 / video_fps
    index = 0
    delays = []
    ave = 0
    if gol_savestate:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        savepath = gol_savepath+'/'+vidname.split('.')[0]+'_result.mp4'
        videoWriter = cv2.VideoWriter(savepath, fourcc, video_fps, (video_width, video_height))
    for i in range(video_frame_cnt):
        begin = time.time()
        index += 1
        is_opened, frame = vid.read()
        q_put.put(frame) if is_opened else None
        q_put.get() if q_put.qsize() > 1 else None  # 保证队列里只有一张图片
        while q_get.empty() and index == 1:
            time.sleep(0.01)
        if index >= 10:
            time.sleep(delay - ave)
        if not q_get.empty():
            (label_id, score, boxes, fps) = q_get.get()
            # img = np.copy(origin_img)
            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                plot_one_box(frame, [x0, y0, x1, y1], label=args.classes[label_id[i]] + ', {:.2f}'.format(score[i]),
                             color=color_table[label_id[i]])
            cv2.putText(frame, '{:.2f}fps'.format(fps), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imshow(vidname, frame)
        else:
            for i in range(len(boxes)):
                x0, y0, x1, y1 = boxes[i]
                plot_one_box(frame, [x0, y0, x1, y1],
                             label=args.classes[label_id[i]] + ', {:.2f}'.format(score[i]),
                             color=color_table[label_id[i]])
            cv2.putText(frame, '{:.2f}fps'.format(fps), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imshow(vidname, frame)
        if gol_savestate:
            videoWriter.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(vidname, cv2.WND_PROP_AUTOSIZE) < 1:
            break
        end = time.time()
        if index < 10 and index >= 2:
            delays.append(end - begin)
            if index == 9:
                ave = np.mean(delays)
            # print(delay,ave,delays)
    vid.release()
    if gol_savestate:
        videoWriter.release()
    cv2.destroyAllWindows()


def picture_op(picture_path, args, sess, boxes, scores, labels, input_data):
    if picture_path == '':
        return
    img_name = picture_path.split('/')[-1]
    color_table = get_color_table(args.num_class)
    img_ori = cv2.imread(picture_path)
    height_ori, width_ori = img_ori.shape[:2]
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
    boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ' {:.2f}'.format(scores_[i]),
                     color=color_table[labels_[i]])
    cv2.imshow(img_name, img_ori)
    global gol_savestate
    global gol_savepath
    if  gol_savestate:
        savepath = gol_savepath+'/'+img_name.split('.')[0]+'_result.'+img_name.split('.')[1]
        cv2.imwrite(savepath, img_ori)
    cv2.waitKey(0)


def batch_picture_op(filename, file_path, output_path, args, sess, boxes, scores, labels, input_data):
    if file_path == ' ' or output_path == '':
        return
    img_name = filename.split('.')[0] + '_result' + '.' + filename.split('.')[1]
    print(img_name)
    color_table = get_color_table(args.num_class)

    img_ori = cv2.imread(file_path + '/' + filename)
    height_ori, width_ori = img_ori.shape[:2]
    img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
    boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ' {:.2f}'.format(scores_[i]),
                     color=color_table[labels_[i]])
    # cv2.imshow('Detection result', img_ori)
    cv2.imwrite(output_path + '/' + img_name, img_ori)
    # cv2.waitKey(0)


def vedio_op(vedio_path, args, sess, boxes, scores, labels, input_data):
    origin_img_q = mp.Queue(maxsize=2)
    result_img_q = mp.Queue(maxsize=4)
    process = mp.Process(target=ved_queue_img_put, args=(origin_img_q, result_img_q, vedio_path,gol_savestate,gol_savepath))
    setattr(process, "daemon", True)
    process.start()
    while process.is_alive():
        while origin_img_q.qsize() == 0:
            if not process.is_alive():
                # print("子进程结束")
                origin_img_q.close()
                result_img_q.close()
                return
            # time.sleep(0.01)
        origin_img = origin_img_q.get()
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(origin_img, args.new_size[0], args.new_size[1])
        else:
            img = cv2.resize(origin_img, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        # time.sleep(0.5)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        result_img_q.get() if result_img_q.qsize() > 1 else None
        result_img_q.put((labels_, scores_, boxes_, fps))


def cam_op(args, sess, boxes, scores, labels, input_data, cam_num=0):
    # 创建进程
    origin_img_q = mp.Queue(maxsize=2)
    result_img_q = mp.Queue(maxsize=2)
    process = mp.Process(target=cam_queue_img_put, args=(origin_img_q, result_img_q, cam_num))
    setattr(process, "daemon", True)
    process.start()
    while process.is_alive():
        while origin_img_q.qsize() == 0:
            if not process.is_alive():
                # print("子进程结束")
                return
            # time.sleep(0.01)
        origin_img = origin_img_q.get()
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize1(origin_img, args.new_size[0], args.new_size[1])
        else:
            img = cv2.resize(origin_img, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        # time.sleep(0.5)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        result_img_q.get() if result_img_q.qsize() > 1 else None
        result_img_q.put((labels_, scores_, boxes_, fps))
    origin_img_q.close()
    result_img_q.close()


def cam_op_new(args, sess, boxes, scores, labels, input_data, cam_num=[0]):
    # 创建进程
    num = len(cam_num)
    vedio_list = ['./demo/in/vedio/vedio3.mp4','./demo/in/vedio/vedio2.mp4','./demo/in/vedio/vedio4.mp4','./demo/in/vedio/vedio5.mp4']
    origin_img_q_list = [mp.Queue(maxsize=2) for i in range(num)]
    result_img_q_list = [mp.Queue(maxsize=2) for i in range(num)]
    processes = [mp.Process(target=cam_queue_img_put, args=(origin_img_q_list[i], result_img_q_list[i], cam_num[i],gol_savestate,gol_savepath,vedio_list[i])) for
                 i in range(num)]
    [setattr(process, "daemon", True) for process in processes]
    # process = mp.Process(target=cam_queue_img_put, args=(origin_img_q, result_img_q,cam_num))
    # setattr(process, "daemon", True)
    [process.start() for process in processes]
    i = 1
    while num > 0:
        index = i % num
        while not processes[index].is_alive():
            del processes[index]
            origin_img_q_list.pop(index).close()
            result_img_q_list.pop(index).close()
            num -= 1
            i = 0
            index = 0
            if num == 0:
                return
        while origin_img_q_list[index].qsize() == 0:
            if processes[index].is_alive():
                i += 1
                index = i % num
            else:
                del processes[index]
                origin_img_q_list.pop(index).close()
                result_img_q_list.pop(index).close()
                num -= 1
                i = 0
                index = 0
                if num == 0:
                    return
        origin_img = origin_img_q_list[index].get()
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(origin_img, args.new_size[0], args.new_size[1])
        else:
            img = cv2.resize(origin_img, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        # time.sleep(0.5)
        end_time = time.time()
        fps = 1 / (end_time - start_time)/num
        result_img_q_list[index].get() if result_img_q_list[index].qsize() > 1 else None
        result_img_q_list[index].put((labels_, scores_, boxes_, fps))
        i += 1


def cap_Window(cap_num):
    num = len(str(cap_num))
    if num < 5:
        cv2.resizeWindow("NO.%d cam" % (cap_num), 480, 360)
        cv2.moveWindow('NO.0 cam', 210, 0)
        cv2.moveWindow('NO.3 cam', 690, 0)
        cv2.moveWindow('NO.1 cam', 210, 360)
        cv2.moveWindow('NO.2 cam', 690, 360)
    else:
        width = 270
        height = 360
        pic = 0
        cv2.resizeWindow("NO.%d cam" % (cap_num), width, height)
        for j in range(2):
            for i in range(5):
                cv2.moveWindow('NO.%d cam' % pic, 270 * i, 360 * j)
                pic += 1