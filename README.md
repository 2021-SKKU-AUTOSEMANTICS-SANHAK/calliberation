# calliberation
python calliberate.py로 실행시킬 수 있습니다.
구조도 위의 4개의 점 (직사각형 형태)을 선택한 후, 동영상의 이미지에서 구조도 위에 찍은 점과 순서, 위치를 맞춰서 선택합니다.
한번 실행할 때 마다 1개의 Matrix가 만들어집니다.
만들어 진 후 test를 위하여 4개의 점을 이미지 위에 선택하면 해당 점이 구조도 위에 표시됩니다.

## 옵션
--camera
0일 경우 cam1 (가게측), 1일 경우 cam2 (창고측)이 선택됩니다.

--resolution
해상도 : 기본은 640입니다.
메인 시스템에서 640 * 480 형태로 Tracking이 진행되기 때문에 해당 사이즈로 진행이 되게 됩니다. 만약 Tracking을 진행할 때 해상도를 다르게 할 생각이라면, 해당 해상도에 맞는 다른 Matrix를 제작해야 합니다.
(코드 내에서 이미지 크기를 바꿔주는 기능은 없습니다. 원하는 사이즈에 맞는 이미지를 다시 만들면 됩니다)

## 저장형태
coor_(camera number)_daiso_(resolution).npy, txt가 생성됩니다.
