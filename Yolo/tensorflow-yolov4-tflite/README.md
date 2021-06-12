original source : https://github.com/hunglc007/tensorflow-yolov4-tflite   (hunglc007)

I needed the coordinates of the detected object.
So I added some code to core/utils.py.

The result is as shown in the image below.

![coordinate](https://user-images.githubusercontent.com/66017052/121765705-d5a02d80-cb87-11eb-91ee-524968f9d9a1.png)

https://user-images.githubusercontent.com/66017052/121765720-ecdf1b00-cb87-11eb-8a3a-d3fded58e345.mp4

It also prints the number of detected objects of each class.


YOLOv4 tensorflow implementation들 중 tflite를 통해 프로젝트를 진행하고 있었는데, 박스에 좌표값이 필요했습니다.
때문에 core/utils.py에 일부 코드를 추가해 매 프레임 마다 박스의 중앙 좌표와 클래스별 감지된 객체 수를 출력하게끔 했습니다.
