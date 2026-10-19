# NotebookLM 文獻研讀與原論文對比分析摘要

本文件提取自紅酒品質多元線性迴歸專案結案報告（[HW2_Report.md](../HW2_Report.md)）第八階段，記錄透過 Google NotebookLM 輔助研讀 Cortez et al. (2009) 原始論文之核心發現與方法學比較。

---

## 一、文獻基本資訊

- **原始論文出處**：Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547-553.
- **資料集來源**：[Kaggle Red Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) (Cortez et al., 2009)
- **輔助研究工具**：Google NotebookLM

---

## 二、NotebookLM 原文核心研究摘要

本研究旨在利用客觀的理化檢測數據預測葡萄牙 Vinho Verde 紅酒的感官品質評分（0–10 分）。研究使用包含 1,599 筆紅酒樣本、11 項化學特徵（如酒精濃度、硫酸鹽、揮發性酸等）的資料集，並比較多元線性迴歸（Multiple Regression, MR）、**類神經網路（Neural Network, NN）與支援向量機（Support Vector Machine, SVM）**三種模型。

實驗顯示，非線性的 SVM 預測表現最佳（紅酒平均絕對偏差 $\text{MAD} = 0.46$），顯著優於 MR（$\text{MAD} = 0.50$）與 NN（$\text{MAD} = 0.51$）；MR 與 NN 在紅酒上的預測準確度十分接近（容忍度 $T=0.5$ 時均為 $59.1\%$）。研究透過敏感度分析（Sensitivity Analysis）結合後向選擇法（Backward Selection）評估特徵重要性，發現硫酸鹽（sulphates）、pH 值、總二氧化硫與酒精濃度對紅酒品質預測最具影響力。

對 HW2 實作的參考價值在於：適當的特徵選擇能有效精簡變數並維持模型效能；而多元線性迴歸雖然具有高度可解釋性，但因受限於線性假設，其預測能力上限不及非線性模型。

---

## 三、原論文與本次 HW2 實作方法之深度比較

1. **工具體系與環境差異**：
   - **Cortez et al. (2009) 原文**：採用 **R 語言** 與 `rminer` 函式庫實現結合敏感度分析的變數篩選與模型訓練（包含 SVM、NN 與 MR）。
   - **本次 HW2 專案**：採用 **Python Scikit-Learn** 生態系，實現嚴格無資料外洩（Leak-free）的機器學習管線（Pipeline），結合 5 折交叉驗證的 `SelectKBest` 進行系統化特徵維度評估。

2. **模型定位與延伸對照**：
   - 兩者皆將多元線性迴歸視為核心的**基礎基準模型（Baseline）**，重視係數對食品理化指標的解釋力。
   - 原文側重探索 SVM 與神經網路等非線性演算法之逼近極限；本次 HW2 則以多元線性迴歸為主模型，輔以 Ridge 脊迴歸與隨機森林（Random Forest）作為對照組，進一步量化了非線性模型的泛化差距（Generalization Gap = 0.3876）。

3. **不確定性量化之創新演進**：
   - 原文主要著重於點預測誤差（MAD 與容忍度分類率）；本次 HW2 則實作了現代統計學前沿的**分割保形預測（Split Conformal Prediction）**，產出具備分佈無假設且有限樣本保證的 95% 預測區間（實證測試覆蓋率達 99.06%），為紅酒批次檢驗決策提供了嚴謹的風險防護界限。
