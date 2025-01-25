import json 
from tqdm import tqdm

input = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/yolor/runs/test/yolor-val18/last_predictions.json"
output = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/predictions.json"

# names:
#   0: Bus
#   1: Bike
#   2: Car
#   3: Pedestrian
#   4: Truck



map_dict = {
    0:3,
    1:1,
    3:1,
    2:2,
    5:0,
    7:4
}

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId

def main():
    with open(input) as f:
        data = json.load(f)
    
    predictions = []
    print(type(data))
    for i in tqdm(range(len(data))):
        if data[i]['category_id'] in list(map_dict.keys()):
            data[i]['image_id'] = get_image_Id(data[i]['image_id'])
            data[i]['category_id'] = map_dict[data[i]['category_id']]
            predictions.append(data[i])
    with open(output, 'w') as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
