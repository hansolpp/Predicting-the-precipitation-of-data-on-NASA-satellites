# **Predicting the precipitation of data on NASA satellites**



![image-20200513130152223](https://user-images.githubusercontent.com/57663398/81794699-20471480-9546-11ea-8734-8c38de3c89cb.png)





### **1. 연구 배경**

  위성은 지구의 70%를 차지하고 있는 해양은 기상 정보는 필수입니다, 강수량 산출은 전지구의 물과 에너지 순환 이해에 중요합니다. 미항공우주국 NASA는 1970년대부터 위성을 이용 강수 산출 연구를 수행해 왔으며, 2014년부터 GPM (Global Precipitation Measurement) 미션을 통해 전 지구 강수량을 30분단위로 제공합니다. NASA에서 사용하고 있는 강수 산출 알고리즘은 물리기반의 베이지안 통계를 활용하고 있어, 전 영역에 대해 고품질 데이터베이스를 만드는 어려움이 있습니다. U-net을 활용 관측값 자체에서 어떠한 경험적 튜닝 없이 강수 산출을 시도하여 NASA보다 정확한 강수산출을 시도합니다.





### **2. 연구의 필요성**

  산업이 팽창하고 복잡해지면서 날씨와 기후의 영향이 모든 산업으로 확산되고 크기가 증가되는 추세입니다. 특히 에너지 산업은 날씨와 기후 관련 위험과 연관이 큰 분야 가운데 하나이기도 합니다. 또한 강수량은 수재해 예측과 악천후 관측, 재난 예방 기술에 필요한 요소입니다.

  2011년 발생한 태국 대홍수는 동남아 지역 자동차 산업 수출/입 거점으로 삼아온 일부 기업에 3조 이상의 매출 피해를 유발하여 산업 부문의 기후변화 대응의 시급성을 시사하였습니다.





### **3. 연구목적**

  현재 기상분야에서는 수치 모델 또는 물리적 특성을 기반으로 대기현상을 분석하고 예측합니다. 물리현상을 설명할 수 없는 머신러닝은 활용하지 않고 있지만, 기상분야에서도 머신러닝의 이미지 분석 적용 영역을 확장시키면 대기 현상을 더 정교하게 이해할 수 있을 것으로 기대할 수 있습니다. 





### **4.연구**

#### 4.1. 데이터 구성 요소

  현재 주어진 데이터는 나사에서 제공된 데이터를 가공한 것으로 GPM(Global Precipitation Measurement) Core 위성의 GMI/DPR 센서에서 북서태평양영역 (육지와 바다를 모두 포함) 에서 관측된 자료입니다.

  각각의 데이터는 마이크로파 이미지를 포함한 데이터이고, (40, 40, 15)의 npy 파일입니다.  40 * 40픽셀의 이미지의 각 픽셀은 15개의 컬럼의 데이터를 가지고 있습니다.(일반 이미지에서는 RGB채널 3개이지만, GMI센서 GPM마이크로파 이미져에는 9개의 채널을 이용한 관측값을 사용합니다). 각 픽셀의 넓이는 GMI센서에 비해 DPR센서의 관측범위가 좁아서, 강우량이 측정되는 DPR센서의 범위의 이미지만을 학습에 사용합니다. 따라서 120 km를 40으로 나눈 3km가 각 픽셀의 한 변 길이 입니다. 



  0~8번 column까지는 GMI센서로 측정한 각 주파수의 밝기 온도 값을 나타냅니다. 그리고 9번 column은 지표 유형을 나타내고 있습니다. 그리고 10,11번은 GMI센서의 위도와 경도를 나타내고 12,13번은 DPR센서의 위도와 경도를 나타냅니다. 마지막으로 14번컬럼은 DPR센서를 이용해서 측정한 강수량을 나타내고 있습니다. 



  Train data의 경우 2016~2018년간의 데이터로 76,345개의 데이터가 있고, 각 데이터의 0~14번 컬럼까지 모든 값이 주어져 있습니다.  test data는 2019년의 측정 데이터이고 총 2,416개의 데이터를 가지고 있습니다. 각 데이터의 14번 컬럼, 즉 DPR센서를 통해 측정한 강수량값이 주어지지 않습니다.



#### **4.2. 데이터 시각화**

##### **4.2.1. 각 주파수의 밝기 온도 (1 ~ 9 column) **

  (단위: K, 10.65GHz~89.0GHz)

<img src="https://user-images.githubusercontent.com/57663398/81795047-9d728980-9546-11ea-917c-03d3689fa02b.png" align="left" width="250" height="250"> <img src="https://user-images.githubusercontent.com/57663398/81796888-01964d00-9549-11ea-93da-496d402e2002.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795270-e591ac00-9546-11ea-88b6-6adbd1e90fb3.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795457-1a9dfe80-9547-11ea-8f8e-a70db69af821.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795593-3f927180-9547-11ea-896f-6618d684be18.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795671-546f0500-9547-11ea-910f-bc8022f6e713.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795737-68b30200-9547-11ea-8e9e-f517b16d01a0.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795804-7cf6ff00-9547-11ea-8468-ba901e8a5ed5.png" align="left" width="250" height="250">   <img src="https://user-images.githubusercontent.com/57663398/81795847-90a26580-9547-11ea-9c58-59c62333cf39.png" align="left" width="250" height="250">   









































##### **4.2.2. 지표면 유형(10 column) **

  (지표 타입 (앞자리 0: Ocean, 앞자리 1: Land, 앞자리 2: Coastal, 앞자리 3: Inland))

![image-20200513141024335](https://user-images.githubusercontent.com/57663398/81795886-a1eb7200-9547-11ea-8032-0beb43698872.png)





##### **4.2.3. GMI, DPR 위도/경도(11 ~ 14 column) **

![image-20200513141351632](https://user-images.githubusercontent.com/57663398/81795923-b16abb00-9547-11ea-9350-51ad4ae2ad2e.png) ![image-20200513141407568](https://user-images.githubusercontent.com/57663398/81795931-b3347e80-9547-11ea-9480-01a51160b38f.png)

![image-20200513141441563](https://user-images.githubusercontent.com/57663398/81795934-b465ab80-9547-11ea-9b68-f7e07fec4da9.png) ![image-20200513141501604](https://user-images.githubusercontent.com/57663398/81795940-b6c80580-9547-11ea-88d1-fe3c1b8e6c9a.png)



##### **4.2.4. 강수량, mm/h (15 column) **

![image-20200513141700320](https://user-images.githubusercontent.com/57663398/81796045-da8b4b80-9547-11ea-826d-29f17376e876.png)





#### 4.3. 데이터 전처리

​	누락데이터제거, 이상치 제거, 특정 데이터에 편중된 모델을 방지하기 위해

​	(현재 진행 중)



### **4.4. 분석 방법**

#### **4.4.1. U-net**

  U-Net은 Biomedical 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델입니다. U-Net은 FCNs보다 확장된 개념의 Up-sampling과 Skip Architecture를 적용한 모델을 제안하고 있습니다. 결과적으로 U-Net의 구조는 아주 적은 양의 학습 데이터만으로 Data Augmentation을 활용하여 여러 Biomedical Image Segmentation 문제에서 우수한 성능을 보여주어 현재 연구에 적용하려 합니다.



  본 연구에서는 딥러닝 기반 이미지 분류기법에 많이 활용되는 U-net 알고리즘 구조를 이용하여 GPM 위성의 GMI 센서의 밝기온도 관측으로부터 강수량을 얻으려 합니다. 데이터는 2016~2018년 3년 동안의 관측 자료를 활용하여 강수량을 출력 값으로 하는 지도학습방법을 이용합니다.



#### **4.4.1. 평가지표**

![image-20200513174740897](https://user-images.githubusercontent.com/57663398/81796079-e70fa400-9547-11ea-9d17-73d825e98f19.png)



  밸런스 한 데이터 일 때는 정확도를 사용 하는 것이 간단하면서 효과적입니다. 하지만 데이터 세트가 밸런스 하지 않은 데이터는 F1 스코어 를 사용하는 것이 좋은 모델을 고르는데 도움이 됩니다. 

![image-20200513174918894](https://user-images.githubusercontent.com/57663398/81796099-eecf4880-9547-11ea-9db6-6443b10fa24b.png)

  그래서 지도학습방법의 결과를 측정하기 위해 MAE/F1 score 를 이용합니다. 의사결정을 위한 모델의 성능을 측정하기 위한 방법으로 정확도(Accuracy), 정밀도 (Precision), 재현율(Recall)을 많이 사용합니다. 먼저 2*2 매트릭스인 오차행렬 (Confusion Matrix)을 이용하여 올바른 것과 올바르지 않은것을 구분합니다.  그후 두 경우를 참과 거짓으로 표현하며 TP(True Positive), TN(True Negative), FP(False Positive), FN(False Negative)으로 구분합니다.



(적중율에 관해서도 추가 예정)



### **5. 연구 결과**

### **6. 고찰**









### **x. 참고논문**

관련논문 : (홍성택, 박병돈, 김종립, 정회경. (2018). 강수량계 종류별 성능시험 및 불확도 분석. 한국정보통신학회논문지, 22(7), 935-942)

관련논문 : (산업부문 기후변화 취약성 평가지표 및 가중치 연구)

(관련 논문 : U-Net: Convolutional Networks for Biomedical Image Segmentation Olaf Ronneberger, Philipp Fischer, Thomas Brox Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015)

(관련논문 : F1 스코어를 이용한 한국어 감정지수 연구)







