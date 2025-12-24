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

由於檔名後綴缺失
