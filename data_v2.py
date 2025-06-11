import os, numpy as np, math, random, matplotlib.pyplot as plt, pickle, cv2
from PIL import Image
from skimage.io import imshow

def get_linemod_path(model_class):
    return os.path.join(os.getcwd(), 'LINEMOD', model_class)

def master_list_gen(base_path):
    imageList = os.listdir(os.path.join(base_path, 'JPEGImages'))
    maskList = os.listdir(os.path.join(base_path, 'mask'))
    labelList = os.listdir(os.path.join(base_path, 'labels'))
    
    if len(imageList) != len(maskList) or len(imageList) != len(labelList):
        raise Exception("image, mask, and label list lengths do not match.")
    
    return [[a, b, c] for a, b, c in zip(imageList, maskList, labelList)]

def get_data_split(gen_new=True, split=.8, model_class='cat'):
    if gen_new:  
        basePath = get_linemod_path(model_class)
        masterList = master_list_gen(basePath)
        random.shuffle(masterList)
        splitPoint = round(len(masterList) * split)
        
        splitDict = {
            "trainData": masterList[:splitPoint],
            "validData": masterList[splitPoint:]
        }
        with open(f"{model_class}_trainSplit", 'wb') as f:
            pickle.dump(splitDict, f)
    else: 
        with open(f"{model_class}_trainSplit", 'rb') as f:
            splitDict = pickle.load(f)
    return (splitDict["trainData"], splitDict["validData"])

def read_image(filePath, height=480, width=640):
    image = Image.open(filePath)
    image = image.resize((width, height))
    return np.array(image)

def get_data_split_image(getValid, modelClass='cat'):
    trainData, validData = get_data_split(modelClass=modelClass)
    basePath = get_linemod_path(modelClass)

    if getValid:
        choice = random.choice(validData)
    else:
        choice = random.choice(trainData)

    imagePath = os.path.join(basePath, 'JPEGImages', choice[0])
    labelPath = os.path.join(basePath, 'labels', choice[2])

    with open(labelPath) as f:
        labels = f.readline().split(' ')[1:19]
    image = read_image(imagePath)
    return image, labels

def set_training_pixel(out_img, y, x, labels, height, width):
    """Assign unit vectors pointing from coordinate to keypoint in image"""
    for i in range(9):
        y_diff = height * float(labels[i * 2 + 1]) - y  
        x_diff = width * float(labels[i * 2]) - x  
        mag = math.sqrt(y_diff ** 2 + x_diff ** 2)
        
        out_img[y][x][i * 2 + 1] = y_diff / mag 
        out_img[y][x][i * 2] = x_diff / mag



def coordsTrainingGenerator(modelClass, batchSize, masterList=None, height=480, width=640, augmentation=True, altLabels=True):
    basePath = get_linemod_path(modelClass)
    if masterList is None:
        masterList = master_list_gen(basePath)
        random.shuffle(masterList)

    i = 0
    while True:
        xBatch = []
        yCoordBatch = []
        for b in range(batchSize):
            if i == len(masterList):
                i = 0
                random.shuffle(masterList)
            x = read_image(os.path.join(basePath, 'JPEGImages', masterList[i][0]), height, width)

            labels_dir = 'altLabels' if altLabels else 'labels'
            with open(os.path.join(basePath, labels_dir, masterList[i][2])) as f:
                labels = f.readline().split(' ')[1:19] # we dump the class label

            yCoordsLabels = np.zeros((height, width, 18))  # 9 coordinates
            modelMask = read_image(os.path.join(basePath, 'mask', masterList[i][1]), height, width)

            if augmentation:
                if random.choice([True, False]):  # vertical flip
                    x = np.flipud(x)
                    modelMask = np.flipud(modelMask)
                    for j in range(len(labels) // 2):
                        labels[j * 2 + 1] = str(round(1 - float(labels[j * 2 + 1]), 6))
                if random.choice([True, False]):  # horizontal flip
                    x = np.fliplr(x)
                    modelMask = np.fliplr(modelMask)
                    for j in range(len(labels) // 2):
                        labels[j * 2] = str(round(1 - float(labels[j * 2]), 6))

            modelCoords = np.where(modelMask == 255)[:2]
            for modelCoord in zip(modelCoords[0][::3], modelCoords[1][::3]):
                set_training_pixel(yCoordsLabels, modelCoord[0], modelCoord[1], labels, height, width)

            xBatch.append(x)
            yCoordBatch.append(yCoordsLabels)
            i += 1
        yield (np.array(xBatch), np.array(yCoordBatch))