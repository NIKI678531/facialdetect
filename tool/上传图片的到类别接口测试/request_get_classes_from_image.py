import requests

with open('img_1172713_0.jpeg', 'rb') as image_file:
    response = requests.post('http://0.0.0.0:21000/get_classes_from_image', files={'image': image_file})
    # response = requests.post('http://0.0.0.0:5006/get_classes_from_image', files={'image': image_file})
    print(response.json())