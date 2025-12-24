# 人工智慧期末報告
# 使用 Google Colab 和機器學習實現人臉識別
# 11225033 吳育緯 11224142 陳政佑
在人工智慧領域，人臉識別技術扮演著至關重要的角色。無論是智慧型手機的安全解鎖、社交媒體的照片標記，還是更複雜的安全監控系統，人臉識別技術都無處不在。本專案將探討如何利用Google Colab雲端平台，結合機器學習技術，打造一個高效且準確的人臉識別系統。我們將採用人工神經網路（Artificial Neural Network, ANN）作為核心模型，並詳細介紹資料準備、模型訓練、結果分析等關鍵步驟，進而能夠全面了解並實踐人臉識別技術。
# 學習目標
1.瞭解深度學習(Deep Learning)和神經網路(Neural Networks)的基本概念。

2.掌握使用 TensorFlow 進行進行數據處理、模型建構、訓練和評估的技巧。

3.學會進行數據正規化和模型參數的調整。

4.實作建構、訓練和評估深度學習模型。

5.提升問題解決和模型優化的能力。
# 步驟說明
1.人臉檢測 (Face Detection)：在圖像或影片中自動尋找人臉的位置。

2.特徵提取 (Feature Extraction)：從檢測到的人臉中提取獨特的特徵，例如眼睛、鼻子和嘴巴之間的距離和比例。

3.人臉比對 (Face Matching)：將提取的特徵與資料庫中已知人臉的特徵進行比對。

4.身份驗證 (Identity Verification)：基於比對結果，確認或拒絕個體的身份。
# 實作
1.資料準備

在進行人臉識別之前，需要準備包含人臉圖像的資料集。常用的資料集包括 Yale Face Database、Labeled Faces in the Wild (LFW) 和 CelebA。對於初學者，Yale Face Database 是一個不錯的選擇，因为它資料量適中，易於處理。
這邊將用Yale Face Database進行示範

將Yale Face Database上傳Google Drive後掛載至Google Colab

<img width="342" height="477" alt="1" src="https://github.com/user-attachments/assets/ed13f6c4-1ec0-45e6-8321-f675626343f9" />

由於檔名後綴缺失，我們用os模組統一修改檔名為.jpg。

<img width="1918" height="652" alt="2" src="https://github.com/user-attachments/assets/ec8e24a4-b614-47f8-8f7d-8184e705e343" />

接下來再預處理這些圖片變成我們想要的格式，包括讀取圖像、轉換為灰階圖像、調整大小、預處理圖像、提取標籤 等

<img width="1920" height="667" alt="3" src="https://github.com/user-attachments/assets/5bcec93e-02c9-4281-83a0-d309ad1b7b82" />

2.模型建立與訓練

使用人工神經網路 (ANN) 進行人臉識別涉及以下步驟：

1.導入必要的函式庫：

<img width="842" height="251" alt="4" src="https://github.com/user-attachments/assets/0f82dc33-af4f-419d-88da-df7ee5016e5d" />

2.載入資料集：

<img width="698" height="161" alt="5" src="https://github.com/user-attachments/assets/757ff676-a93c-4877-8171-3cda733f5e58" />

3.分割訓練集和測試集：

(Tips: 20%資料作為測試集，80%資料作為訓練集)

<img width="1121" height="76" alt="6" src="https://github.com/user-attachments/assets/9c31166f-61b1-43ac-83db-b352bb28dc34" />

4.開始訓練

建立並訓練 MLP 模型 (Build and Train MLP Model):

先設定隱藏層的神經元數量為 200。

初始化一個 MLPClassifier 模型，並設定以下參數：
hidden_layer_sizes=(n_neurons,)：定義了神經網路的結構，這裡表示有一個隱藏層，包含 200 個神經元。

指定優化算法為 Adam，這是一種常用且高效的梯度下降優化器。
設定激活函數為 ReLU (Rectified Linear Unit)，有助於解決梯度消失問題並加速訓練。

batch_size設定每次訓練迭代時使用的樣本數量為32，並在訓練過程中輸出進度訊息，啟用提前停止機制，如果驗證分數在一段時間內沒有改善，訓練會自動停止，以防止過擬合。
max_iter設定最大迭代次數為 500。

model.fit(X_train.reshape(X_train.shape[0], -1), y_train) -->使用訓練資料來訓練模型。X_train.reshape(X_train.shape[0], -1) 將圖像資料從多維數組（例如 (樣本數, 高度, 寬度)）展平為二維數組（樣本數, 像素總數)），這是 MLPClassifier 所需的輸入格式。

然後進行預測並轉換標籤

使用訓練好的模型對測試集 X_test 進行預測，得到編碼後的預測標籤

將預測的數值標籤轉換回原始的文字類別名稱，將真實的測試集數值標籤也轉換回原始的文字類別名稱，以便與預測結果進行比較。

評估模型性能

印出分類報告，包含每個類別的精確率、召回率、F1 分數和支援樣本數，以及總體的準確度等指標。

印出混淆矩陣，它顯示了模型正確和錯誤分類的詳細情況，可以幫助我們了解模型在哪個類別上表現良好，在哪個類別上容易混淆。

<img width="858" height="313" alt="7" src="https://github.com/user-attachments/assets/61bf293a-dead-4d8e-a62f-565b1d626454" />

# 訓練成果

訓練結果顯示，模型在多層感知器 (MLP) 神經網路的訓練和測試中取得了顯著的進步。經過多次迭代，模型的驗證分數最高達到了 1.00，並在訓練過程中自動停止，以避免過度擬合。最終，模型在測試集上的整體準確度為 88%。

詳細的分類報告顯示，對於大部分的subject，模型都能達到很高的精確率、召回率和 F1 分數。例如，subject01、subject02 等多個類別都達到了 100% 的表現。然而，在某些類別，如 subject03、subject07、subject08、subject09、subject10、subject12、subject15，模型的表現略有波動，例如在 subject03 和 subject10 上有 50% 的召回率。

混淆矩陣進一步說明了模型的預測細節，它顯示了模型在各個類別上的實際分類與預測分類。雖然大部分預測是正確的，但您可以看到少數樣本被錯誤地分類到其他相似的類別中，例如 subject03 有一個樣本被錯分為 subject07，subject07中有一個樣本被錯分為 subject15。

# 總體來說，這個模型表現良好，但對於某些特定類別的識別，未來仍有優化的潛力。

<img width="870" height="530" alt="8" src="https://github.com/user-attachments/assets/88ca269d-9622-4419-a5a1-11b7e979b3c5" />
<img width="517" height="410" alt="9" src="https://github.com/user-attachments/assets/8e304a00-ff67-4bc8-8d95-37980bff4376" />
<img width="302" height="318" alt="10" src="https://github.com/user-attachments/assets/f979c704-a65d-4baf-af0c-583956ecf7ad" />

# 心得與結論

作為初學者，透過這個專案能夠學習深度學習的理論知識，而且還能從中獲得實際操作經驗，對於理解、掌握深度學習概念來說很重要。無論是從數據處理、建構模型架構到模型訓練及評估，都是了解深度學習流程的重要組成部分。此外也可以讓讀者有機會練習調整模型參數（如學習率、層數、神經元數量等），這是優化模型性能的關鍵技能。讓學習者可以透過實驗不同的設定來觀察對模型性能的影響，這樣的實務經驗對於如何成為一名深度學習工程師非常重要，也是一個非常好的入門學習範例，適合我們透過實作及詳細介紹實際理解深度學習的基本原理和方法。





