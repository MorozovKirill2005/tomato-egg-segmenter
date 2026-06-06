import torch
from ben2 import AutoModel
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Dict
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained("PramaLLC/BEN2").to(device)
model.eval()


def first_preprocessing(image: np.ndarray) -> np.ndarray:
    '''
    функция предварительной обработки изображения, а именно, применения медианного и Гауссова блюра
    '''
    img = image.copy()
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    return img


def get_union_objects_mask(image: np.ndarray) -> np.ndarray:
    '''
    Функция, принимающая изображение и возвращающая маску в которой чёрными пикселями обозначен фон, а белыми - объекты
    '''
    binary_mask = np.array(model.inference(Image.fromarray(image)))[:, :, 3]
    binary_mask = (binary_mask > 200).astype(np.uint8) * 255
    binary_mask = cv2.medianBlur(binary_mask, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary_mask


def get_object_masks(binary_mask: np.ndarray) -> np.ndarray:
    '''
    Функция, принимающая маску объектов и возвращающая маску, где каждому пикселю соответствует число - номер объекта
    '''
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    coordinates = peak_local_max(dist, min_distance=23, labels=binary_mask)
    markers = np.zeros(binary_mask.shape, dtype=np.int32)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    labels = watershed(-dist, markers, mask=binary_mask)
    return labels


def tomatoes_classification(features: np.ndarray) -> np.ndarray:
    '''
    Функция для определения цвета помидоров, использующая алгоритм класстеризации K-means
    '''
    mean_red = np.array([129, 200, 99])
    mean_yellow = np.array([12, 210, 172])
    red_dist = ((features - mean_red) ** 2).sum(axis=1)
    yellow_dist = ((features - mean_yellow) ** 2).sum(axis=1)
    dists = np.stack([yellow_dist, red_dist])
    classification = dists.argmin(axis=0)
    if np.unique(classification).size == 1:
        return classification
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans_model.fit_predict(X)
    if clusters[np.argmax(features[:, 0])] == 0:
        clusters = (1 - clusters)
    return clusters


def object_visualization(image: np.ndarray, labels: np.ndarray) -> Tuple[Image.Image, Dict[str, int]]:
    '''
    Функция, которая определяет класс каждого объекта из labels, ведёт подсчёт объектов и визуализирует на картинке их маски
    '''
    results = {
        'red': 0,
        'yellow': 0,
        'egg': 0,
        'total': 0
    }
    result_image = image.copy()
    tomatoes = []
    tomatoes_masks = []
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for label in range(1, labels.max() + 1):
        object_mask = (labels == label).astype(np.uint8)
        area = object_mask.sum()
        if area < 100:
            continue
        kernel = np.ones((5, 5), np.uint8)
        mask_for_statistic = cv2.erode(object_mask, kernel, iterations=1)
        mean_h, mean_s, mean_v = cv2.mean(hsv, mask=mask_for_statistic)[:3]
        if mean_s > 150:
            tomatoes.append([mean_h, mean_s, mean_v])
            tomatoes_masks.append(object_mask)
        else:
            results['egg'] += 1
            result_image[object_mask == 1] = [255, 255, 255]
        results['total'] += 1
    if len(tomatoes):
        tomatoes = np.array(tomatoes)
        classification = tomatoes_classification(tomatoes)
        for color, object_mask in zip(classification, tomatoes_masks):
            if color:
                result_image[object_mask == 1] = [255, 0, 0]
                results['red'] += 1
            else:
                result_image[object_mask == 1] = [255, 255, 0]
                results['yellow'] += 1
    return Image.fromarray(result_image), results


def processing(image: Image.Image) -> Tuple[Image.Image, Dict[str, int]]:
    image = np.array(image.convert("RGB"))
    image_after_preprocessing = first_preprocessing(image)
    binary_mask = get_union_objects_mask(image_after_preprocessing)
    labels = get_object_masks(binary_mask)
    result_image, results = object_visualization(image, labels)
    return result_image, results
