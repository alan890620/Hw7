# ==========================================
# 匯入所需套件 (Import the required Libraries)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    st.title("糖尿病預測網頁應用程式")
    st.markdown("透過機器學習模型判斷是否罹患糖尿病？")
    
    st.sidebar.title("模型選擇面板")
    st.sidebar.markdown("請選擇您的模型與相關超參數")
    
    # 載入資料
    @st.cache_data
    def load_data():
        data = pd.read_csv("diabetes.csv")
        return data
     
    # 資料切割 (支援動態比例調整)
    def split(df, test_size):
        y = df.Outcome
        x = df.drop(columns=['Outcome'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test
     
    # 繪製評估指標圖表
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if '混淆矩陣 (Confusion Matrix)' in metrics_list:
            st.subheader("混淆矩陣 (Confusion Matrix)")
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap="Blues")
            st.pyplot(plt.gcf())
            plt.clf()

        if 'ROC 曲線 (ROC Curve)' in metrics_list:
            st.subheader("ROC 曲線 (ROC Curve)")
            y_prob = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC 曲線 (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--", label="隨機猜測")
            plt.xlabel("偽陽性率 (False Positive Rate)")
            plt.ylabel("真陽性率 (True Positive Rate)")
            plt.title("受試者工作特徵曲線 (ROC)")
            plt.legend(loc="lower right")
            st.pyplot(plt.gcf())
            plt.clf()
 
    df = load_data() 
    class_names = ['正常 (Non-Diabetes)', '糖尿病 (Diabetes)']
    
    # 側邊欄：設定訓練資料比例
    st.sidebar.subheader("資料集切分設定")
    test_size = st.sidebar.slider("測試集比例 (Test Set Ratio)", 0.1, 0.5, 0.3, 0.05)
    
    x_train, x_test, y_train, y_test = split(df, test_size)
 
    st.sidebar.subheader("選擇分類器")
    classifier = st.sidebar.selectbox("分類器模型", ("決策樹 (Decision Tree)", "邏輯斯迴歸 (Logistic Regression)", "隨機森林 (Random Forest)"))

    # 選擇分類器與超參數設定
    if classifier == '決策樹 (Decision Tree)':
        st.sidebar.subheader("模型參數")
        criterion = st.sidebar.radio("切分品質評估標準 (Criterion)", ("gini", "entropy"), key='criterion')
        splitter = st.sidebar.radio("節點切分方式 (Splitter)", ("best", "random"), key='splitter')
        metrics = st.sidebar.multiselect("選擇要顯示的評估指標：", ('混淆矩陣 (Confusion Matrix)', 'ROC 曲線 (ROC Curve)'))
 
        if st.sidebar.button("開始訓練與評估", key='classify'):
            st.subheader("決策樹模型結果")
            model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            model.fit(x_train, y_train)
            
            # 將訓練好的模型存入 session_state，避免重複訓練
            st.session_state['trained_model'] = model
            st.session_state['classifier_name'] = classifier
            
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("準確率 (Accuracy): ", round(accuracy * 100, 2), "%")
            st.write("精確度 (Precision): ", round(precision_score(y_test, y_pred), 2))
            st.write("召回率 (Recall): ", round(recall_score(y_test, y_pred), 2))
            st.write("F1 分數 (F1-Score): ", round(f1_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)
 
    elif classifier == '邏輯斯迴歸 (Logistic Regression)':
        st.sidebar.subheader("模型參數")
        C = st.sidebar.number_input("正規化參數 (C)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("最大迭代次數 (Max Iterations)", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("選擇要顯示的評估指標：", ('混淆矩陣 (Confusion Matrix)', 'ROC 曲線 (ROC Curve)'))

        if st.sidebar.button("開始訓練與評估", key='classify'):
            st.subheader("邏輯斯迴歸模型結果")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            
            st.session_state['trained_model'] = model
            st.session_state['classifier_name'] = classifier
            
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("準確率 (Accuracy): ", round(accuracy * 100, 2), "%")
            st.write("精確度 (Precision): ", round(precision_score(y_test, y_pred), 2))
            st.write("召回率 (Recall): ", round(recall_score(y_test, y_pred), 2))
            st.write("F1 分數 (F1-Score): ", round(f1_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)
 
    elif classifier == '隨機森林 (Random Forest)':
        st.sidebar.subheader("模型超參數")
        n_estimators = st.sidebar.number_input("樹木數量 (N Estimators)", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("樹的最大深度 (Max Depth)", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("是否使用 Bootstrap 取樣", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("選擇要顯示的評估指標：", ('混淆矩陣 (Confusion Matrix)', 'ROC 曲線 (ROC Curve)'))

        if st.sidebar.button("開始訓練與評估", key='classify'):
            st.subheader("隨機森林模型結果")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap=='True', n_jobs=-1)
            model.fit(x_train, y_train)
            
            st.session_state['trained_model'] = model
            st.session_state['classifier_name'] = classifier
            
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("準確率 (Accuracy): ", round(accuracy * 100, 2), "%")
            st.write("精確度 (Precision): ", round(precision_score(y_test, y_pred), 2))
            st.write("Recall (召回率): ", round(recall_score(y_test, y_pred), 2))
            st.write("F1 分數 (F1-Score): ", round(f1_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    # ==========================================
    # 單筆資料預測功能 (Single Prediction Section)
    # ==========================================
    st.markdown("---")
    st.subheader("💡 單筆預測與機率分佈")
    
    with st.form("single_prediction_form"):
        st.write("請輸入受測者的各項生理與檢驗指標數值：")
        input_data = {}
        features = df.drop(columns=['Outcome']).columns
        
        # 使用 2 欄排版輸入元件
        cols = st.columns(2)
        for i, col in enumerate(features):
            col_idx = i % 2
            with cols[col_idx]:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
                
        submit_single = st.form_submit_button("進行單筆預測")
        
        if submit_single:
            if 'trained_model' in st.session_state:
                model = st.session_state['trained_model']
                input_df = pd.DataFrame([input_data])
                
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0] # [prob_non_diabetes, prob_diabetes]
                
                st.success(f"**預測結果：** {class_names[prediction]}")
                
                # 繪製機率分佈圓餅圖
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(probabilities, labels=class_names, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
                ax.axis('equal')
                ax.set_title("預測機率分佈")
                st.pyplot(fig)
                plt.clf()
            else:
                st.warning("⚠️ 請先至側邊欄選擇模型並點擊 **「開始訓練與評估」** 按鈕來訓練模型！")

    # 顯示原始資料
    if st.sidebar.checkbox("顯示原始資料集", False):
        st.subheader("糖尿病原始資料集 (Diabetes Raw Dataset)")
        st.write(df) 
  
if __name__ == '__main__':
    main()