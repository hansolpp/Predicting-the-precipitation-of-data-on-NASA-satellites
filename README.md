# **Predicting the precipitation of data on NASA satellites**

  

![image-20200513130152223](https://user-images.githubusercontent.com/57663398/81794699-20471480-9546-11ea-8734-8c38de3c89cb.png)    

​    

# 1. 연구의 필요성

  산업이 팽창하고 복잡해지면서 날씨와 기후의 영향이 모든 산업으로 확산되고 크기가 증가되는 추세입니다. 특히 수력발전, 태양광 발전 등의 에너지 산업은 날씨와 기후에 연관이 큰 분야입니다.  또한 강수량은 수재해 예측과 악천후 관측, 재난 예방 기술에 필요한 요소입니다. 2011년 발생한 태국 대홍수는 동남아 지역 자동차 산업 기업에 3조 이상의 매출피해를 유발하여 산업부문의 기후변화 대응의 시급성을 시사하기도 하였습니다.



  기존의 기상 레이더는 넓은 영역의 강우 분포 정보를 제공할 수 있지만, 레이더로 부터 송신된 전자파 빔으로 측정한 신호는 수km 상공에 강수입자에 대한 신호이기 때문에 실제 저층 강우와는 오차가 있습니다. 또한 우량계는 강우량을 직접 측정할 수 있지만, 강한 바람이나 우량계 입구 막힘, 보정 오차 등에 의해 관측오차가 발생할수 있으며 높은 공간 해상도의 강우량을 제공하지 못합니다. 따라서 관측 사각지역에서 강우량을 측정하는 데 마이크로파 이미지를 통해 강우량을 계산해 강우량 정보를 제공 할 수 있습니다.

​      

​     

# 2. 연구 목적

  현재 기상분야에서는 수치 모델 또는 물리적 특성을 기반으로 대기현상을 분석하고 예측하며, 물리현상을 설명할 수 없는 머신러닝은 활용하지 않고 있습니다. 하지만 머신러닝의 영역을 확장시키면 기상분야에서도 활용 하여 대기 현상을 더 정교하게 이해할 수 있을 것으로 예상됩니다.

  본 연구에서는 GPM위성의 GMI, DPR센서로 기록된 마이크로웨이브파를 이용한 온도측정 이미지와 강수량 데이터를 머신러닝의 학습데이터로 활용하여, 위성 마이크로파 이미지를 이용해 강수량 이미지를 산출합니다. 마이크로파는 대기 중의 수상체와 상호작용하여 마이크로파 이미져 관측의 방출, 산란 신호를 이용하면 강수 시스템의 미세물리적 특징을 분석할 수 있습니다.

  연구에서 사용하는 데이터는 NASA에서 제공한 GPM 위성 데이터로, NASA의 데이터 정책에 따라 공개되고 있습니다. 마이크로파 이미지는 기존의 기상레이더와는 달리 구름의 내부 정보를 얻을 수 있으며, 우량계보다 넓은 범위의 관측 정보를 얻을 수 있습니다. 이번 연구가 머신러닝이 강우량 추정에 효과적임이 입증된다면, 기존의 강우량 관측에 더 많은 정보를 제공할수 있다고 판단됩니다.

 

​      

# 3. 연구

### 3.1. 데이터 구성 요소

![image](https://user-images.githubusercontent.com/57663398/82110696-49041f80-977b-11ea-81a3-9fc2f311b989.png)



  현재 주어진 데이터는 나사에서 제공된 데이터를 가공한 것으로 GPM(Global Precipitation Measurement) Core 위성의 GMI/DPR 센서에서 북서태평양영역 (육지와 바다를 모두 포함) 에서 관측된 자료입니다.

  각각의 데이터는 마이크로파 이미지를 포함한 데이터이고, (40, 40, 15)의 npy 파일입니다.  40 * 40픽셀의 이미지의 각 픽셀은 15개의 컬럼의 데이터를 가지고 있습니다.(일반 이미지에서는 RGB채널 3개이지만, GMI센서 GPM마이크로파 이미져에는 9개의 채널을 이용한 관측값을 사용합니다). 각 픽셀의 넓이는 GMI센서에 비해 DPR센서의 관측범위가 좁아서, 강우량이 측정되는 DPR센서의 범위의 이미지만을 학습에 사용합니다. 따라서 120 km를 40으로 나눈 3km가 각 픽셀의 한 변 길이 입니다. 

​    

  0~8번 column까지는 GMI센서로 측정한 각 주파수의 밝기 온도 값을 나타냅니다. 그리고 9번 column은 지표 유형을 나타내고 있습니다. 그리고 10,11번은 GMI센서의 위도와 경도를 나타내고 12,13번은 DPR센서의 위도와 경도를 나타냅니다. 마지막으로 14번컬럼은 DPR센서를 이용해서 측정한 강수량을 나타내고 있습니다. 

   

  Train data의 경우 2016 ~ 2018년간의 데이터로 76,345개의 데이터가 있고, 각 데이터의 0 ~ 14번 컬럼까지 모든 값이 주어져 있습니다.  test data는 2019년의 측정 데이터이고 총 2,416개의 데이터를 가지고 있습니다. 각 데이터의 14번 컬럼, 즉 DPR센서를 통해 측정한 강수량값이 주어지지 않습니다.

​    

### 3.2. 데이터 시각화

##### 3.2.1. 각 주파수의 밝기 온도 (1 ~ 9 column) 

  (단위: K, 10.65 GHz~89.0 GHz)

<img src="https://user-images.githubusercontent.com/57663398/81795047-9d728980-9546-11ea-917c-03d3689fa02b.png" align="left" width="250" height="250"> <img src="https://user-images.githubusercontent.com/57663398/81796888-01964d00-9549-11ea-93da-496d402e2002.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795270-e591ac00-9546-11ea-88b6-6adbd1e90fb3.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795457-1a9dfe80-9547-11ea-8f8e-a70db69af821.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795593-3f927180-9547-11ea-896f-6618d684be18.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795671-546f0500-9547-11ea-910f-bc8022f6e713.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795737-68b30200-9547-11ea-8e9e-f517b16d01a0.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795804-7cf6ff00-9547-11ea-8468-ba901e8a5ed5.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795847-90a26580-9547-11ea-9c58-59c62333cf39.png" width="250" height="250">   



  

​      

##### 3.2.2. 지표면 유형(10 column)

  (지표 타입 (앞자리 0: Ocean, 앞자리 1: Land, 앞자리 2: Coastal, 앞자리 3: Inland))

![image-20200513141024335](https://user-images.githubusercontent.com/57663398/81795886-a1eb7200-9547-11ea-8032-0beb43698872.png)

  

  

  

##### 3.2.3. GMI, DPR 위도/경도(11 ~ 14 column) 

![image-20200513141351632](https://user-images.githubusercontent.com/57663398/81795923-b16abb00-9547-11ea-9350-51ad4ae2ad2e.png) ![image-20200513141407568](https://user-images.githubusercontent.com/57663398/81795931-b3347e80-9547-11ea-9480-01a51160b38f.png)

![image-20200513141441563](https://user-images.githubusercontent.com/57663398/81795934-b465ab80-9547-11ea-9b68-f7e07fec4da9.png) ![image-20200513141501604](https://user-images.githubusercontent.com/57663398/81795940-b6c80580-9547-11ea-88d1-fe3c1b8e6c9a.png)

  

  

##### 3.2.4. 강수량, mm/h (15 column) 

![image-20200513141700320](https://user-images.githubusercontent.com/57663398/81796045-da8b4b80-9547-11ea-826d-29f17376e876.png)

​    

​    

### 3.3. 데이터 구성 및 처리

##### 3.3.1. 누락데이터 제거 

train data에서 강수량 데이터를 측정하지 못한경우 값이 -9999로 표현되어 있어 해당 데이터를 제거합니다.



##### 3.3.2. 데이터 불균형 제거

강수량 column에 해당하는 데이터의 합의 분포는 다음과 같습니다. 50 mm 이하의 강수량을 가진 데이터가 50%에 달하며, 데이터가 편향되어 있는 불균형 현상을 보여주고 있습니다.

![image](https://user-images.githubusercontent.com/57663398/82136774-f9d9ef80-984b-11ea-9618-6e7377dbf15c.png)

  

  데이터 불균형 현상을 해결하기 위해서 먼저 데이터를 구간별로 나누고자 하였습니다. 강수량 5 mm 단위로 구간을 정한후 데이터를 구간별 수집하였습니다. 하지만 위 데이터 분포에서 알 수 있듯이 강수량이 많은 데이터가 상대적으로 부족하기에, 단위를 10 mm, 50 mm, 100 mm, 400 mm, 500 mm로 구간별로 나누어 구간별 데이터를 적정량 확보하였습니다. 다음은 데이터 구간의 분포를 나타낸 것입니다. 총 165개의 구간이 있습니다.

 (0, (0, 5], (5, 10], (10, 15], (15, 20], ... ,(455, 465], (465, 475], ... , (2005, 2105], (2105, 2205], (2205, 2305], ... , (7005, ∞])



![image-20200521010744039](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200521010744039.png)

(사진 수정 예정)



위의 그림은 구간으로 나누어서 수집한 데이터의 분포입니다. 앞서 언급한바와 같이 불균형을 시각적으로 확인할 수 있습니다. 



다음으로, 데이터 불균형을 맞추기 위해 up sampling과 down sampling을 진행합니다. 정해진 threshold보다 많은 데이터를 가진 범위는 down sampling을 실시하며 threshold보다 적은 데이터를 가진 범위는 up sampling을 실시합니다.



![image-20200521012545355](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200521012545355.png)

(사진 수정 예정)





먼저 down sampling을 진행합니다. threshold를 400으로 정하고 나머지 데이터는 사용하지 않습니다.

![image-20200521012545355](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200521012545355.png)

(사진 수정 예정)



up sampling을 진행합니다. 정해진 threshold 보다 데이터가 부족한 구간에서 다음과 같이 추가시킵니다. 이미지를 90, 180, 270도 회전 및 대칭 회전하여 데이터를 부풀립니다.



(데이터 회전한 사진도 넣으면 좋음)

(데이터 불균형을 해소하지 않고 학습한 경우를 비교하여 쓸 것)

​	

  

### **3.4. 분석 방법**

#### **3.4.1. U-net**

![image-20200526231843253](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200526231843253.png)

(이미지 출처: https://www.researchgate.net/figure/Convolutional-neural-network-CNN-architecture-based-on-UNET-Ronneberger-et-al_fig2_323597886)

  U-Net은 Biomedical 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델입니다. U-Net은 FCNs보다 확장된 개념의 Up-sampling과 Skip Architecture를 적용한 모델을 제안하고 있습니다. 결과적으로 U-Net의 구조는 아주 적은 양의 학습 데이터만으로 Data Augmentation을 활용하여 여러 Biomedical Image Segmentation 문제에서 우수한 성능을 보여주어 현재 연구에 적용하려 합니다.

  

#### **3.4.2. 평가지표**

![image-20200513174740897](https://user-images.githubusercontent.com/57663398/81796079-e70fa400-9547-11ea-9d17-73d825e98f19.png)

  

  밸런스 한 데이터 일 때는 정확도를 사용 하는 것이 간단하면서 효과적입니다. 하지만 데이터 세트가 밸런스 하지 않은 데이터는 F1 스코어 를 사용하는 것이 좋은 모델을 고르는데 도움이 됩니다. 

![image-20200526232113918](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200526232113918.png)

  그래서 지도학습방법의 결과를 측정하기 위해 MAE/F1 score 를 이용합니다. 먼저 2*2 매트릭스인 오차행렬 (Confusion Matrix)을 이용하여 올바른 것과 올바르지 않은것을 구분합니다.  그후 두 경우를 참과 거짓으로 표현하며 TP(True Positive), TN(True Negative), FP(False Positive), FN(False Negative)으로 구분합니다.





### 3.4. 교차검증

성능측정과 오버피팅을 방지하기 위해 데이터를 나누고 여러 모델을 학습합니다.

![image-20200526235354851](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200526235354851.png)

(이미지 출처: https://m.blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=221599517834&proxyReferer=https:%2F%2Fwww.google.com%2F)



  현재 실시되어지고 있는 모델은 k-fold cross validation입니다. k에는 전체 데이터 셋을 몇 등분할지 결정할 수있습니다.  k = 5로 설정하여 진행하고 있습니다.





![image-20200526235435749](C:\Users\hansol\AppData\Roaming\Typora\typora-user-images\image-20200526235435749.png)

(이미지 출처: https://m.blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=221599517834&proxyReferer=https:%2F%2Fwww.google.com%2F)



  강수량을 label로 하여 구분되는 데이터 셋을 균등하게 설정합니다. (아직 미적용)

####   

# 4. 연구 결과

​    

   

​    

​     

​     

​    

# x. 참고논문

관련논문 : (홍성택, 박병돈, 김종립, 정회경. (2018). 강수량계 종류별 성능시험 및 불확도 분석. 한국정보통신학회논문지, 22(7), 935-942)

관련논문 : (산업부문 기후변화 취약성 평가지표 및 가중치 연구)

(관련 논문 : U-Net: Convolutional Networks for Biomedical Image Segmentation Olaf Ronneberger, Philipp Fischer, Thomas Brox Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015)

(관련논문 : F1 스코어를 이용한 한국어 감정지수 연구)







