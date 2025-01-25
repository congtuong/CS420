
import json 
from tqdm import tqdm

input_A = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/fisheye_a/exp/predictions.json"
input_N = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/fisheye_n/exp/predictions.json"

output = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/predictions.json"

def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId

def main():
    with open(input_A) as f:
        data_A = json.load(f)
    with open(input_N) as f:
        data_N = json.load(f)
    for i in tqdm(range(len(data_A))):
        data_A[i]['image_id'] = get_image_Id(data_A[i]['image_id'])
    for i in tqdm(range(len(data_N))):
        data_N[i]['image_id'] = get_image_Id(data_N[i]['image_id'])
    data_A.extend(data_N)
    print(len(data_A))
    with open(output, 'w') as f:
        json.dump(data_A, f)

if __name__ == "__main__":
    main()
