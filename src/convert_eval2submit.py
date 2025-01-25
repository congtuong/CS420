import json 
from tqdm import tqdm

input = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/runs/inference/all_data_5class/exp/predictions.json"
output = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/predictions_yolov6_all_data_5class.json"

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId

def main():
    # map_dict = {
    #     0:3,
    #     1:1,
    #     3:1,
    #     2:2,
    #     5:0,
    #     7:4
    # }   
    map_dict = {
        0:3,
        1:1,
        3:1,
        2:2,
        5:0,
        7:4
    }
    with open(input) as f:
        data = json.load(f)
    for i in tqdm(range(len(data))):
        data[i]['image_id'] = get_image_Id(data[i]['image_id'])
        # data[i]['category_id'] = map_dict[data[i]['category_id']]
        data[i]['category_id'] = int(data[i]['category_id'])
    with open(output, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
