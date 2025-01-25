from pathlib import Path
import random
import shutil
from tqdm import tqdm

def load_data():
    root_dir = Path("/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024")
    images = list((root_dir / "images").glob("*.png"))

    return images

images_test = load_data()
              
set_id = set([
    "_".join(t.stem.split("_")[:1])
    for t in images_test
])

sorted(set_id)

print(len(set_id))

random.seed(42)
output_dir = Path("./test")
for _time in tqdm("MAEN"):
    test_name = []
    for _id in set_id:
        for t in images_test:
            if _id == t.stem.split("_")[0] and _time == t.stem.split("_")[1]:
                test_name.append(t)
    # print(test_name[:5])
    (output_dir / _time / "test" / "images").mkdir(parents=True, exist_ok=True)

    print("Done making dir")
    print("Start copying to ", output_dir / _time / "test" / "images")
    print(_time)
    for img in tqdm(test_name):
        shutil.copy(img, output_dir / _time / "test" / "images")


