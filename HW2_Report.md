# 資料探勘作業二（Homework 2）：紅酒品質多元線性迴歸預測與 CRISP-DM 實踐分析報告

---

## 專案基本資訊
- **學生姓名**：[待填寫]
- **學號**：[待填寫，例如 7115056206]
- **課程名稱**：資料探勘 (Data Mining)
- **程式檔案**：[hw2.py](file:///c:/Users/john1/Desktop/物聯網/1015/hw2.py)（繳交前將重新命名為學號，如 `7115056206.py`）
- **資料集來源**：Kaggle Red Wine Quality Dataset (Cortez et al., 2009)
  - Kaggle 原始連結：[https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
  - UCI 原始論文出處：Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547-553.
- **核心分析模型**：**多元線性迴歸 (Multiple Linear Regression)** 為必選核心主模型，並輔以 **Ridge 脊迴歸** 與 **Random Forest Regressor 隨機森林** 作為對照模型。

---

## 摘要 (Executive Summary)

本研究以跨行業資料探勘標準流程（**CRISP-DM**）為指導框架，針對葡萄牙「Vinho Verde」紅酒之 11 項物理化學指標建立紅酒感官評鑑品質（`quality`）之預測模型。本作業全面實現了無資料外洩（Leak-free）的機器學習管線（Pipeline）、5 折交叉驗證特徵選取（SelectKBest）、主模型係數解釋、多模型比較，以及本作業特別要求之**具有統計理論依據的 95% 保形預測區間（Split Conformal Prediction）**。

### 核心成果指標摘要
| 模型名稱 | 訓練集 R² | 測試集 R² | 測試集 MAE | 測試集 RMSE | 5 折 CV 平均 RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **多元線性迴歸 (主模型)** | **0.3480** | **0.4032** | **0.5035** | **0.6245** | **0.6618 ± 0.0136** |
| **Ridge 脊迴歸 (對照組)** | 0.3480 | 0.4032 | 0.5036 | 0.6245 | 0.6617 ± 0.0136 |
| **Random Forest (對照組)** | 0.9266 | 0.5390 | 0.4224 | 0.5489 | 0.5995 ± 0.0263 |

- **特徵選取最佳子集**：8 項關鍵特徵（CV RMSE 達最佳值 `0.6593`）。
- **95% 分割保形預測區間邊界**：$\hat{y} \pm 1.5021$（區間全寬 $3.0042$）。
- **測試集實證覆蓋率 (Empirical Coverage)**：**99.06%**（317/320 筆樣本命中，超越 95% 之理論名義覆蓋率保證）。

---

## 第一階段：商業理解 (Business Understanding)

### 1.1 專案背景與預測目標
紅酒品質的傳統評定嚴重仰賴專業品酒師（Sommeliers）進行感官品評（Sensory Evaluation）。然而，人工品評存在顯著痛點：
1. **主觀偏差與疲勞**：品酒師的味覺與嗅覺容易受到心理狀態、品評順序、時間與生理疲勞影響。
2. **評鑑成本昂貴且無法即時化**：在現代化酒莊大規模發酵與陳釀過程中，無法隨時邀請專家評分。
3. **製程調整延遲**：若能在釀造化驗階段直接透過儀器量測的物理與化學參數（如酸度、酒精濃度、亞硫酸鹽含量等）預測最終成品質感，釀酒師便能動態調控發酵溫度、添加劑劑量與熟成時機，大幅降低報廢損失並提高高端酒款產出率。

因此，本專案的商業目標為：**以 11 項可客觀測量的理化特徵為基礎，構建高可解釋性之多元線性迴歸預測模型，精確量化各化學成分對品質分數之邊際貢獻，並輸出具備統計可靠度保證之品質預測區間。**

### 1.2 輸入特徵與目標變數定義
本資料集共包含 11 個數值型理化輸入特徵與 1 個目標評分變數：

| 變數名稱 | 資料型態 | 單位/尺度 | 物理與化學意義及釀造影響 |
| :--- | :---: | :---: | :--- |
| **fixed acidity** (固定酸) | float64 | $\text{g(酒石酸)/dm}^3$ | 大多為酒石酸（tartaric acid），不易蒸發，維持酒體基礎酸度結構。 |
| **volatile acidity** (揮發性酸) | float64 | $\text{g(醋酸)/dm}^3$ | 主要為醋酸（acetic acid），過量會產生刺鼻醋酸腐味，嚴重破壞口感。 |
| **citric acid** (檸檬酸) | float64 | $\text{g/dm}^3$ | 微量存在，賦予紅酒清新果香感與爽脆風味。 |
| **residual sugar** (殘糖量) | float64 | $\text{g/dm}^3$ | 發酵後殘留糖分，低於 1 g/L 罕見，紅酒通常屬於甘口（Dry），殘糖量適中。 |
| **chlorides** (氯化物/鹽分) | float64 | $\text{g(NaCl)/dm}^3$ | 紅酒中鹽分含量，過多會使口感偏鹹甚至苦澀。 |
| **free sulfur dioxide** (游離二氧化硫) | float64 | $\text{mg/dm}^3$ | 溶解之 $\text{SO}_2$ 氣體，主要扮演抗氧化與抑制雜菌滋生之防腐功能。 |
| **total sulfur dioxide** (總二氧化硫) | float64 | $\text{mg/dm}^3$ | 游離與結合態 $\text{SO}_2$ 總和，低濃度無感，若超過 50 ppm 會產生刺鼻硫磺味。 |
| **density** (密度) | float64 | $\text{g/cm}^3$ | 接近水密度，主要受酒精（低密度）與殘糖（高密度）兩者消長影響。 |
| **pH** (酸鹼值) | float64 | 0 ~ 14 尺度 | 衡量紅酒酸性強弱，多數紅酒介於 3.0 ~ 4.0 之間，影響微生物穩定性與酒體色澤。 |
| **sulphates** (硫酸鹽添加劑) | float64 | $\text{g(K}_2\text{SO}_4)/\text{dm}^3$ | 釀酒添加劑，促進產生防腐之二氧化硫氣體，抑制微生物並抗氧化。 |
| **alcohol** (酒精濃度) | float64 | % vol | 酒精體積百分比，直接影響酒體醇厚度（Body）、溫潤感與風味平衡。 |
| **quality** (品質評分，**Target**) | int64 | 0 ~ 10 整數評分 | 至少三名感官品評專家獨立評分的評級中位數，紅酒資料集範圍為 3 到 8 分。 |

---

## 第二階段：資料理解 (Data Understanding)

### 2.1 資料讀取與分隔符號判定
在載入 `winequality-red.csv` 時，程式使用自動探測邏輯，檢測首行字元：
- 探測結果：原始檔案使用**分號（`;`）**作為分隔符號（`delimiter=';'`）。
- 資料集維度：共 **1,599** 筆觀測樣本、**12** 個欄位（11 個輸入特徵 + 1 個目標變數 `quality`），嚴格符合官方規格。

### 2.2 資料品質與重複值檢驗
- **缺失值（Missing Values）**：全資料集統計缺失值數量為 **0**，無任何空值或需刪除之欄位。
- **重複列（Duplicate Rows）**：檢驗發現存在 **240** 筆特徵完全相同之重複樣本（佔比約 15%）。
  > **資料探勘決策**：在食品理化化驗中，儀器量測精度（如小數點後兩位）有限，不同生產批次或相鄰儲酒桶具有相同理化測量值且由不同專家評鑑，屬於合理的客觀現象。為完整保留 Cortez et al. (2009) 之經典評測分佈，本專案全數保留該 1,599 筆樣本，不人為隨意剔除。

### 2.3 探索性資料分析 (EDA) 與視覺化

#### (1) 特徵相關性熱圖 (Correlation Heatmap)
下圖為 11 項特徵與品質變數間之 Pearson 相關係數矩陣熱圖：

![Correlation Heatmap](correlation_heatmap.png)

**關鍵發現**：
1. **正相關前三名**：
   - `alcohol` 與品質具最強正相關（$r = +0.48$），表明較高酒精濃度通常伴隨更佳的成熟度與評分。
   - `sulphates` 與品質呈正相關（$r = +0.25$），顯示適量的亞硫酸鹽保護酒質不易劣化。
   - `citric acid` 與品質呈正相關（$r = +0.23$），適量酸度帶來清新度。
2. **負相關前兩名**：
   - `volatile acidity` 與品質呈現最強負相關（$r = -0.39$），醋酸過高是劣質酒的主要指標。
   - `total sulfur dioxide` 與品質呈負相關（$r = -0.19$），二氧化硫累積過量影響嗅覺純淨度。
3. **特徵間的多重共線性（Multicollinearity）現象**：
   - `fixed acidity` 與 `citric acid` 呈強正相關（$r = +0.67$）。
   - `fixed acidity` 與 `density` 呈強正相關（$r = +0.67$）。
   - `fixed acidity` 與 `pH` 呈顯著負相關（$r = -0.68$）。
   - `alcohol` 與 `density` 呈顯著負相關（$r = -0.50$）。

#### (2) 探索性特徵散佈圖 (Exploratory Scatter Plots)
針對與品質關聯最顯著的 4 項關鍵特徵（alcohol, volatile acidity, sulphates, citric acid），加入微小抖動（Jitter）以呈現離散評分下的分佈趨勢：

![Exploratory Scatter Plots](exploratory_scatter_plots.png)

由圖中趨勢線（紅線）清晰可見：高分酒（品質 7、8）高度集中在高酒精度、低揮發性酸、較高硫酸鹽之區域。

---

## 第三階段：資料準備 (Data Preparation)

### 3.1 訓練集與測試集劃分 (80/20 Train/Test Split)
為確保後續評估的公正性與學術可復現性：
- 採用 `train_test_split(test_size=0.2, random_state=42)`。
- **訓練集 (Training Set)**：1,279 筆樣本 (80%)。
- **測試集 (Testing Set)**：320 筆樣本 (20%)。

### 3.2 避免資料外洩 (Data Leakage Prevention) 與 Pipeline 設計
在資料採礦實踐中，若將全體資料集一起進行標準化或特徵選取，會使測試集資訊滲透進訓練階段，導致模型效能虛高（Data Leakage）。
- 本專案採用 `sklearn.pipeline.Pipeline` 將特徵處理封裝：
  1. `SimpleImputer(strategy='median')`：提供未來部署時缺失值的安全填充。
  2. `StandardScaler()`：將各特徵縮放至均值為 0、標準差為 1 的標準常態分佈。
  3. `SelectKBest` 或 `LinearRegression`。
- **關鍵原則**：所有 `fit()` 動作僅於訓練集（或交叉驗證的訓練折）執行，測試集僅執行 `transform()` 與 `predict()`。

### 3.3 特徵選取分析 (Feature Selection via 5-Fold Cross Validation)
採用 `SelectKBest(score_func=f_regression)` 在訓練集內部透過 5 折交叉驗證，測試特徵維度 $k=1 \sim 11$ 對模型的影響：

![Feature Selection Metrics](feature_selection_metrics.png)

#### 5 折交叉驗證特徵選取詳細記錄表：
| 特徵數量 (k) | 選入特徵清單 | CV RMSE (越低越好) | CV RMSE 標準差 | CV R² (越高越好) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | alcohol | 0.7120 | ±0.0201 | 0.2162 |
| 2 | volatile acidity; alcohol | 0.6741 | ±0.0102 | 0.2966 |
| 3 | volatile acidity; sulphates; alcohol | 0.6670 | ±0.0128 | 0.3113 |
| 4 | volatile acidity; citric acid; sulphates; alcohol | 0.6667 | ±0.0122 | 0.3120 |
| 5 | volatile acidity; citric acid; total sulfur dioxide; sulphates; alcohol | 0.6634 | ±0.0121 | 0.3192 |
| 6 | volatile acidity; citric acid; total sulfur dioxide; density; sulphates; alcohol | 0.6633 | ±0.0124 | 0.3194 |
| 7 | volatile acidity; citric acid; chlorides; total sulfur dioxide; density; sulphates; alcohol | 0.6613 | ±0.0126 | 0.3237 |
| **8 (最佳)** | **fixed acidity; volatile acidity; citric acid; chlorides; total sulfur dioxide; density; sulphates; alcohol** | **0.6593** | **±0.0128** | **0.3276** |
| 9 | 上述 8 項 + free sulfur dioxide | 0.6594 | ±0.0157 | 0.3274 |
| 10 | 上述 9 項 + pH | 0.6604 | ±0.0151 | 0.3258 |
| 11 (全選) | 全體 11 項特徵 | 0.6618 | ±0.0136 | 0.3226 |

**結論**：在 $k=8$ 時，5 折 CV RMSE 取得全域最低點（`0.6593`），且剔除了具多重共線性且貢獻微弱的 `residual sugar`、`pH` 與 `free sulfur dioxide`。

---

## 第四階段：模型建構 (Modeling)

### 4.1 必選主模型：多元線性迴歸 (Multiple Linear Regression)
作為本作業必選的核心主模型，我們在 1,279 筆訓練資料上擬合標準化多元線性迴歸管線：
$$\hat{y} = \beta_0 + \sum_{j=1}^{11} \beta_j X_j$$

#### 迴歸係數分析表 (Regression Coefficients)
下表列出依「標準化係數絕對值」排序之迴歸權重，能直觀衡量各物理化學變數對紅酒品質的相對重要度：

| 變數 (Feature) | 標準化係數 ($\beta_{\text{std}}$) | 原始尺度係數 ($\beta_{\text{raw}}$) | 對品質之影響方向與物理意涵 |
| :--- | :---: | :---: | :--- |
| **(截距項 Intercept)** | **5.6239** | **14.3551** | 平均基準品質評分 |
| **alcohol** | **+0.2966** | +0.2819 | 強烈正向貢獻（每提高 1% 酒精，評分提升約 0.28 分） |
| **volatile acidity** | **-0.1794** | -1.0013 | 強烈負向衝擊（揮發酸每增加 1 g/L，評分驟降 1 分） |
| **sulphates** | **+0.1468** | +0.8412 | 顯著正向貢獻（保護酒體抗氧化，提升口感潔淨度） |
| **total sulfur dioxide** | **-0.1200** | -0.0036 | 顯著負向抑制（總二氧化硫過高抑制酒香） |
| **chlorides** | **-0.0891** | -1.8065 | 負向影響（鹽度偏高劣化酒體品質） |
| **pH** | **-0.0606** | -0.3937 | 輕微負向影響 |
| **free sulfur dioxide** | **+0.0580** | +0.0056 | 輕微正向防腐作用 |
| **fixed acidity** | **+0.0398** | +0.0231 | 正向支撐酒體酸度骨架 |
| **citric acid** | **-0.0275** | -0.1408 | 輕微調節作用 |
| **density** | **-0.0192** | -10.3516 | 微弱影響 |
| **residual sugar** | **+0.0094** | +0.0066 | 影響極微弱（甘口紅酒殘糖差異對品質影響最小） |

### 4.2 參考對照模型 (Reference Models)
為提供客觀的比較基準，另行訓練兩種參考模型：
1. **Ridge Regression ($\alpha=1.0$)**：施加 $L_2$ 權重衰減，用於抑制多重共線性引起的權重膨脹。
2. **Random Forest Regressor ($n=100$)**：非線性集成樹模型，捕捉特徵間之高階非線性交互作用。

---

## 第五階段：模型評估 (Evaluation)

### 5.1 完整評估指標對比 (Evaluation Metrics)

| 評估指標 | 多元線性迴歸 (主模型) | Ridge 脊迴歸 | Random Forest 隨機森林 |
| :--- | :---: | :---: | :---: |
| **5 折 CV 平均 RMSE** | 0.6618 ± 0.0136 | 0.6617 ± 0.0136 | **0.5995 ± 0.0263** |
| **5 折 CV 平均 R²** | 0.3226 | 0.3227 | **0.4443** |
| **訓練集 R² (Train R²)** | 0.3480 | 0.3480 | **0.9266** |
| **測試集 R² (Test R²)** | 0.4032 | 0.4032 | **0.5390** |
| **測試集 MAE** | 0.5035 | 0.5036 | **0.4224** |
| **測試集 MSE** | 0.3900 | 0.3900 | **0.3012** |
| **測試集 RMSE** | 0.6245 | 0.6245 | **0.5489** |

#### 模型比較視覺化 (Model Comparison)
![Model Comparison](model_comparison.png)

**深度對比解析**：
1. **線性迴歸 vs. Ridge 迴歸**：兩者在各項指標上表現幾乎完全相同（Test RMSE 皆為 `0.6245`，Test R² 皆為 `0.4032`）。這證明了特徵尺度經標準化後，原始特徵的多重共線性並未對預測穩定性造成災難性損害，$L_2$ 正則化帶來的偏差-變異權衡在此資料集上相當平衡。
2. **線性迴歸 vs. 隨機森林（泛化差距分析）**：
   - 隨機森林在非線性擬合上展現優勢，測試集 RMSE 降至 `0.5489`，測試集 $R^2$ 提升至 `0.5390`。
   - 然而，隨機森林在訓練集上 $R^2$ 高達 `0.9266`，測試集驟降至 `0.5390`，呈現出顯著的**過擬合泛化差距（Generalization Gap = 0.3876）**。
   - 相反地，多元線性迴歸的訓練集 $R^2$（0.3480）與測試集 $R^2$（0.4032）相當接近，未發生過擬合，且擁有無法被隨機森林黑盒子取代的**公式解析性與化學係數可解釋度**。

### 5.2 線性迴歸診斷圖表深入剖析

#### (1) 實際值 vs. 預測值散佈圖 (Actual vs. Predicted)
![Actual vs Predicted](actual_vs_predicted.png)
- 紅色虛線為理想預測線（$y = \hat{y}$）。
- 觀察可見，模型預測值主要集中在 5.0 至 6.5 分之間。對於極端低分（品質 3）或極端高分（品質 8），線性迴歸出現向均值回歸（Shrinkage toward the mean）現象，這歸因於資料集中品質 3 與 8 的樣本極為稀少（類別不平衡）。

#### (2) 殘差診斷圖 (Residual Diagnostics)
![Residual Plot](residual_plot.png)
- **左圖（殘差 vs. 擬合值）**：圖中呈現清晰的傾斜平行線狀分佈。這是因為真實品質分數為整數離散標籤（$y \in \{3, 4, 5, 6, 7, 8\}$），因而殘差 $e_i = y_i - \hat{y}_i$ 在固定整數 $y$ 下隨連續預測值 $\hat{y}$ 呈斜率為 -1 之直線排列，屬於離散因變數之典型特徵。
- **右圖（殘差分佈直方圖）**：殘差平均值 $\mu = 0.03$、標準差 $\sigma = 0.62$，整體呈現極為優良的鐘形對稱常態分佈，滿足迴歸分析殘差無偏的基本假定。

#### (3) 學習曲線 (Learning Curve)
![Learning Curve](learning_curve.png)
- 當訓練樣本數由 200 筆增加至 1,000 筆以上時，訓練集 RMSE 與 5 折驗證集 RMSE 穩定收斂至約 `0.65 ~ 0.66`，且信賴區間收窄，證明多元線性迴歸已充分收斂，增加樣本數不會改變線性模型的容量上限。

---

### 5.3 重要新增要求：具有統計理論依據之 95% 預測區間 (Prediction Interval)

在實際釀酒工程中，單一數值的「點預測（Point Prediction）」無法反映未來未見酒款品質的不確定性。為提供具有嚴謹統計理論保證的決策界限，本專案採用現代不確定性量化領域之黃金標準——**分割保形預測（Split Conformal Prediction / Inductive Conformal Prediction）**。

#### (1) 95% 保形預測區間圖 (Prediction Interval Plot)
下圖將 320 筆測試集樣本依預測品質遞增排序，展示 95% 預測區間帶與實際觀測點的分佈：

![Prediction Interval](prediction_interval.png)

#### (2) 數學理論與演算法推導
Split Conformal Prediction 具備**有限樣本無分佈假設（Finite-sample Distribution-free）**之統計保證：
1. **資料分割**：將 1,279 筆訓練集以 75% / 25% 獨立切分為：
   - 適當訓練集（Proper Training Set, $n_{\text{prop}} = 959$ 筆）：用於擬合多元線性迴歸管線 $\hat{f}(X)$。
   - 校準集（Calibration Set, $n_{\text{cal}} = 320$ 筆）：完全不參與參數擬合，專門用於量化模型預測誤差分佈。
2. **計算校準集非符合度分數（Nonconformity Scores）**：
   $$R_i = |y_i - \hat{f}(X_i)|, \quad \forall i \in \text{Calibration Set}$$
3. **計算保形分位數（Conformal Quantile $q$）**：
   給定顯著水準 $\alpha = 0.05$（即名義保證水準 $1 - \alpha = 95\%$），精確取樣分位數為：
   $$p = \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}} = \frac{\lceil 321 \times 0.95 \rceil}{320} = \frac{305}{320} \approx 0.953125$$
   $$q = \text{Quantile}(R, p) = 1.5021$$
4. **測試集之 95% 預測區間構造**：
   對任意新樣本 $X_{\text{test}}$，其預測區間定義為：
   $$\hat{C}(X_{\text{test}}) = \left[ \hat{f}(X_{\text{test}}) - q, \; \hat{f}(X_{\text{test}}) + q \right] = \left[ \hat{y} - 1.5021, \; \hat{y} + 1.5021 \right]$$
   區間全寬度為 $2q = 3.0042$ 分。

#### (3) 實證測試覆蓋率 (Empirical Test Coverage)
在完全未參與模型訓練與校準的 320 筆獨立測試集（Test Set）上進行驗證：
- **覆蓋樣本數**：317 筆落入區間內。
- **未覆蓋樣本數**：僅 3 筆超出預測區間（圖中橘色三角形標記）。
- **實證測試覆蓋率 (Empirical Coverage)**：
  $$\text{Coverage}_{\text{test}} = \frac{317}{320} = \mathbf{99.06\%}$$
實證覆蓋率達 $99.06\%$，完全滿足且高於理論名義的 $95\%$ 保證，證明了保形預測區間的高可靠度。

#### (4) 核心統計概念辨析：預測區間 (Prediction Interval) vs. 信賴區間 (Confidence Interval)
本專案在圖表與分析中嚴格明確區分兩者，避免混淆：
- **信賴區間 (Confidence Interval, CI)**：
  - **估計對象**：母體條件期望值 $E[Y|X]$（即真實迴歸平面/平均反應）。
  - **特性**：僅量化「模型參數估計」的不確定性。隨著樣本數 $N \to \infty$，信賴區間寬度將收斂縮小至 0。
- **預測區間 (Prediction Interval, PI)**：
  - **估計對象**：未來**單一獨立觀測值** $Y_{\text{new}} = E[Y|X_{\text{new}}] + \epsilon$。
  - **特性**：除了包含模型參數不確定性之外，還必須涵蓋資料本身的不可約隨機誤差（Irreducible Error $\epsilon$）。因此，**預測區間寬度永遠大於信賴區間**，即便樣本無窮大，其寬度仍受限於噪聲方差 $\sigma_\epsilon$。在品質決策中，釀酒師面對的是每一瓶獨立生產的酒，因此唯有預測區間具備實務意義。

#### (5) 假設條件與局限性 (Assumptions and Limitations)
- **統計假設 (Assumptions)**：
  - **樣本可交換性 (Exchangeability)**：校準集與測試集樣本滿足獨立同分佈（i.i.d.）即可成立。
  - **無母數特性 (Non-parametric)**：本方法**不需要**假設殘差服從高斯常態分佈，亦不需要誤差具備齊次變異數（Homoscedasticity）。
- **局限性 (Limitations)**：
  - **常數區間寬度**：標準 Split Conformal Prediction 輸出固定寬度（$\pm 1.5021$），未能隨特徵空間中局部稀疏或異質變異區域自動縮放（若需變動寬度，未來可擴充為局部加權保形預測或 Conformalized Quantile Regression）。
  - **樣本利用折損**：需將訓練資料切出校準集，略微犧牲了模型參數訓練的可用數據量。

---

## 第六階段：模型部署 (Deployment)

### 6.1 模型持久化與自動驗證
- **序列化檔案**：經完整訓練之標準化多元線性迴歸 Pipeline 已序列化儲存為 [red_wine_model.pkl.gz](file:///c:/Users/john1/Desktop/物聯網/1015/red_wine_model.pkl.gz)（採用 Gzip level 3 壓縮）。
- **重載驗證機制**：腳本在完成儲存後，立即透過 `joblib.load()` 重新載入，並對測試集樣本進行推論比對。經 `assert np.allclose()` 嚴格比對，反序列化模型之輸出與原始模型達到浮點數完全一致，通過部署驗證。

### 6.2 實務落地與推論應用範例
在酒莊生產線上，可透過如下 Python 腳本將訓練完成之模型封裝為 RESTful API 或即時邊緣計算模組：

```python
import joblib
import pandas as pd

# 載入持久化管線
pipeline = joblib.load("red_wine_model.pkl.gz")
CONFORMAL_MARGIN = 1.5021  # 95% 保形預測邊界

# 模擬新釀造批次之化驗數據
new_batch = pd.DataFrame([{
    "fixed acidity": 7.3,
    "volatile acidity": 0.65,
    "citric acid": 0.00,
    "residual sugar": 1.2,
    "chlorides": 0.065,
    "free sulfur dioxide": 15.0,
    "total sulfur dioxide": 21.0,
    "density": 0.9946,
    "pH": 3.39,
    "sulphates": 0.47,
    "alcohol": 10.0
}])

# 產出預測評分與 95% 保形預測區間
predicted_quality = pipeline.predict(new_batch)[0]
lower_pi = predicted_quality - CONFORMAL_MARGIN
upper_pi = predicted_quality + CONFORMAL_MARGIN

print(f"預測品質評分: {predicted_quality:.2f}")
print(f"95% 預測區間: [{lower_pi:.2f}, {upper_pi:.2f}]")
```

---

## 第七階段：GPT 輔助內容

在本次作業開發與研究過程中，生成式 AI（GPT）提供了以下具體的工程與理論輔助：
1. **CRISP-DM 標準流程架構梳理**：輔助規劃從商業理解、理化特徵定義到模型部署的完整六階段學術架構，確保邏輯完整且可追溯。
2. **無外洩（Leak-Free）管線實作指導**：協助利用 `sklearn.pipeline.Pipeline` 將特徵標準化與 5 折交叉驗證特徵選取（`SelectKBest`）封裝，徹底杜絕測試集特徵外洩。
3. **Split Conformal Prediction 數學分位數推導**：協助校驗有限樣本保形分位數公式 $\lceil (n+1)(1-\alpha) \rceil / n$，並輔助完成預測區間與信賴區間之本質差異分析。
4. **跨平台編碼與視覺化相容性排查**：協助排查 Windows 環境下 PowerShell `cp950` 字元集編碼問題，確保輸出腳本兼具穩定性與高解析度排版品質。

---

## 第八階段：NotebookLM 摘要

> **說明**：依作業規範要求，本章節保留為結構化佔位符號（Placeholder），待使用者將相關紅酒釀造文獻、Cortez et al. (2009) 原文 PDF 匯入 Google NotebookLM 取得摘要後填入。本專案不虛構任何未經驗證之 NotebookLM 發現、論文引文或 Kaggle 排名。

```markdown
[NotebookLM 摘要佔位符號]
- 文獻來源：Cortez et al. (2009) "Modeling wine preferences by data mining from physicochemical properties"
- 核心發現待填寫：
  1. 理化特徵與品酒師感官分數的非線性關聯分析。
  2. 支持向量機 (SVM) 與多層感知機 (MLP) 在原論文中的基準評估表現。
  3. 各化學成分（如 SO2 抑制機制與酒精口感平衡）之食品科學原理解讀。
```

---

## 專案產出檔案清單檢核

| 檔案名稱 | 說明 | 狀態 |
| :--- | :--- | :---: |
| [hw2.py](file:///c:/Users/john1/Desktop/物聯網/1015/hw2.py) | 完整 Python 原始碼（CRISP-DM 全流程） | 已完成且執行成功 |
| [winequality-red.csv](file:///c:/Users/john1/Desktop/物聯網/1015/winequality-red.csv) | Kaggle / UCI 官方真實資料集 (1599 筆) | 已下載校驗 |
| [red_wine_model.pkl.gz](file:///c:/Users/john1/Desktop/物聯網/1015/red_wine_model.pkl.gz) | 序列化儲存之多元線性迴歸 Pipeline | 已儲存並驗證通過 |
| [evaluation_metrics.json](file:///c:/Users/john1/Desktop/物聯網/1015/evaluation_metrics.json) | 完整量化評估指標 JSON 檔 | 已輸出 |
| [feature_selection_results.csv](file:///c:/Users/john1/Desktop/物聯網/1015/feature_selection_results.csv) | 5 折交叉驗證特徵選取詳細記錄表 | 已輸出 |
| [model_comparison_metrics.csv](file:///c:/Users/john1/Desktop/物聯網/1015/model_comparison_metrics.csv) | 多元線性迴歸、Ridge 與隨機森林對比數據 | 已輸出 |
| [regression_coefficients.csv](file:///c:/Users/john1/Desktop/物聯網/1015/regression_coefficients.csv) | 標準化與原始尺度迴歸係數表 | 已輸出 |
| [learning_curve.csv](file:///c:/Users/john1/Desktop/物聯網/1015/learning_curve.csv) | 學習曲線樣本點與誤差數值 | 已輸出 |
| [actual_vs_predicted.png](file:///c:/Users/john1/Desktop/物聯網/1015/actual_vs_predicted.png) | 實際值 vs. 預測值散佈圖 | 已生成 |
| [feature_selection_metrics.png](file:///c:/Users/john1/Desktop/物聯網/1015/feature_selection_metrics.png) | 特徵數量對應 CV RMSE / R² 趨勢圖 | 已生成 |
| [model_comparison.png](file:///c:/Users/john1/Desktop/物聯網/1015/model_comparison.png) | 模型比較長條圖 (RMSE, MAE, R²) | 已生成 |
| [residual_plot.png](file:///c:/Users/john1/Desktop/物聯網/1015/residual_plot.png) | 殘差散佈圖與常態分佈直方圖 | 已生成 |
| [correlation_heatmap.png](file:///c:/Users/john1/Desktop/物聯網/1015/correlation_heatmap.png) | 11 項特徵與品質之相關性熱圖 | 已生成 |
| [learning_curve.png](file:///c:/Users/john1/Desktop/物聯網/1015/learning_curve.png) | 多元線性迴歸學習曲線圖 | 已生成 |
| [prediction_interval.png](file:///c:/Users/john1/Desktop/物聯網/1015/prediction_interval.png) | 95% 保形預測區間圖 (Split Conformal) | 已生成 |
| [exploratory_scatter_plots.png](file:///c:/Users/john1/Desktop/物聯網/1015/exploratory_scatter_plots.png) | 關鍵理化特徵與品質探索散佈圖 | 已生成 |
| [HW2_Report.md](file:///c:/Users/john1/Desktop/物聯網/1015/HW2_Report.md) | 完整 CRISP-DM 結案分析報告 (Markdown) | 本報告 |
