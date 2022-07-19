import json
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as  plt
root = "E:/data/V100/dataset/VIL100" # 这是数据的根地址
json_train = "../cache/V100_train.json"
dbfile = os.path.join(root,'data','db_info.yaml')
imgdir = os.path.join(root,'JPEGImages')
jsondir = os.path.join(root,'json')
## 我们需要思考一下测试的时候
def generate_json(names,json_dir,save_file):
    json_context =  {}
    max_points = 0
    max_lanes = 0
    lower = 100
    for index,video_name in enumerate(names):
        path = os.path.join(json_dir,video_name)
        frames = []
        video_context = {}
        size = [0,0]
        print(video_name,os.listdir(path))
        ## json文件中的大小存在错误，所以需要自己进行获取
        image = cv2.imread(
            os.path.join("E:\data\V100\dataset\VIL100\JPEGImages", video_name + "/" + os.listdir(path)[0][:-5]))
        size[0],size[1] = image.shape[:2]
        for frame_name in os.listdir(path):
            frame_context = {}
            with open(os.path.join(path,frame_name)) as f:
                jsonString = json.load(f)
                lanes = jsonString['annotations']['lane']
                ids = []
                lanes_x = []
                lanes_y = []
                if len(lanes)==0:
                    print(video_name,frame_name,"no lanes here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                for i in range(len(lanes)):
                    points = lanes[i]['points']
                    if len(points)==0:
                        print("this is no points")
                        continue
                    ids.append(lanes[i]['lane_id'])
                    x, y = list(zip(*points))
                    lanes_x.append(x[::-1])
                    lanes_y.append(y[::-1])
                    lower = min(lower,min(np.array(y)/size[0]))

                    max_points = max(max_points,len(x))
            frame_context['lanes_id'] = ids
            frame_context['lanes_x'] = lanes_x
            frame_context['lanes_y'] = lanes_y
            frame_context['size'] = tuple(size)
            max_lanes = max(max_lanes,len(lanes_x))
            video_context[frame_name.split(".")[0]] = frame_context
            frames.append(frame_context)
        print(size)
        if size[0]>size[1]:
            print("notice !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        json_context[video_name] = video_context
        print(f"has handles {index} videos")
    # json_data = json.dumps(json_context,indent=4)
    # with open(save_file,"w") as f:
    #     f.write(json_data)
    # f.close()
    return max_points,max_lanes,lower

with open(dbfile,'r') as f:
    db = yaml.load(f,Loader = yaml.Loader)['sequences']
    info = db
    max_points,max_lanes = 0,0
    for split in ['train','test']:
        videos_name = [info['name'] for info in db if info['set'] == split]
        mp,mlane,lower = generate_json(videos_name,jsondir,f"../cache/V100_{split}.json")
        max_points = max(mp,max_points)
        max_lanes = max(mlane,max_lanes)
        print("split",mp,mlane,lower)
    print(max_points,"the final max points is this")
    print(max_lanes)

# # def visual():
# #     ## 说明这个数据集的顺序没有任何的问题
# #     img = os.path.join(root,"JPEGImages","0_Road001_Trim003_frames","00000.jpg")
# #     print(img)
# #     img = cv2.imread(img)
# #     print(img.shape)
# #     plt.figure()
# #     with open("../cache/V100_train.json") as f:
# #         json_data = json.load(f)["0_Road001_Trim003_frames"]['00000']
# #         lanes_x,lanes_y = json_data['lanes_x'],json_data['lanes_y']
# #         for index,(lane_x,lane_y) in enumerate(zip(lanes_x,lanes_y)):
# #             lane = np.stack([lane_x,lane_y]).transpose()
# #             lane = np.array([lane], np.int64)
# #             print(lane[0])
# #             cv2.polylines(img, lane, isClosed=False, color=[255,255,0], thickness=5)
# #             plt.imshow(img)
# #             plt.show()
# #             plt.pause(0)
# # visual()
# =======
# # def generate_json(names,json_dir,save_file):
# #     json_context =  {}
# #     max_points = 0
# #     max_lanes = 0
# #     for index,video_name in enumerate(names):
# #         path = os.path.join(json_dir,video_name)
# #         frames = []
# #         video_context = {}
# #         size = [0,0]
# #         for frame_name in os.listdir(path):
# #             frame_context = {}
# #             with open(os.path.join(path,frame_name)) as f:
# #                 jsonString = json.load(f)
# #                 lanes = jsonString['annotations']['lane']
# #                 size[0],size[1] = jsonString['info']['height'],jsonString['info']['width']
# #                 ids = []
# #                 lanes_x = []
# #                 lanes_y = []
# #                 for i in range(len(lanes)):
# #                     points = lanes[i]['points']
# #                     if len(points)==0:
# #                         print("this is no points")
# #                         continue
# #                     ids.append(lanes[i]['lane_id'])
# #                     x, y = list(zip(*points))
# #                     lanes_x.append(x)
# #                     lanes_y.append(y)
# #                     max_points = max(max_points,len(x))
# #             frame_context['lanes_id'] = ids
# #             frame_context['lanes_x'] = lanes_x
# #             frame_context['lanes_y'] = lanes_y
# #             frame_context['size'] = tuple(size)
# #             max_lanes = max(max_lanes,len(lanes_x))
# #             video_context[frame_name.split(".")[0]] = frame_context
# #             frames.append(frame_context)
# #         video_context['frames'] = frames
# #         json_context[video_name] = video_context
# #         print(f"has handles {index} videos")
# #     json_data = json.dumps(json_context)
# #     with open(save_file,"w") as f:
# #         f.write(json_data)
# #     f.close()
# #     return max_points,max_lanes
# #
# # with open(dbfile,'r') as f:
# #     db = yaml.load(f,Loader = yaml.Loader)['sequences']
# #     info = db
# #     max_points,max_lanes = 0,0
# #     for split in ['train','test']:
# #         videos_name = [info['name'] for info in db if info['set'] == split]
# #         mp,mlane = generate_json(videos_name,jsondir,f"../cache/V100_{split}.json")
# #         max_points = max(mp,max_points)
# #         max_lanes = max(mlane,max_lanes)
# #         print("split",mp,mlane)
# #     print(max_points,"the final max points is this")
# #     print(max_lanes)
#
# def visual():
#     ## 说明这个数据集的顺序没有任何的问题
#     img = os.path.join(root,"JPEGImages","0_Road001_Trim003_frames","00000.jpg")
#     print(img)
#     img = cv2.imread(img)
#     print(img.shape)
#     plt.figure()
#     with open("../cache/V100_train.json") as f:
#         json_data = json.load(f)["0_Road001_Trim003_frames"]['00000']
#         lanes_x,lanes_y = json_data['lanes_x'],json_data['lanes_y']
#         for index,(lane_x,lane_y) in enumerate(zip(lanes_x,lanes_y)):
#             lane = np.stack([lane_x,lane_y]).transpose()
#             lane = np.array([lane], np.int64)
#             print(lane[0])
#             cv2.polylines(img, lane, isClosed=False, color=[255,255,0], thickness=5)
#             plt.imshow(img)
#             plt.show()
#             plt.pause(0)
# visual()
# >>>>>>> 535daacee8ccdd264f48494ccbae9b8796d2f216
