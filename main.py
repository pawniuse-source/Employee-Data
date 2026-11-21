import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

st.title("Employee Data Dashboard")
st.header("Pawni") 
st.write("This application will analyze the Employee Dataset")
st.subheader("Data Features")
st.markdown ("""
- emp_id
- emp_name
- dept_id
- salary
- gender
- is_active
- department
""")
st.success("Application Started Successfully")
df=pd.read_excel("empsmt_cleaned.xlsx", sheet_name="Sheet1")
st.dataframe(df.head())
st.write("** rows and columns**")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.subheader("Dataset Columns")
for i in df.columns:
    st.write(f"- {i}")
st.subheader("Active vs Inactive Employees")
df_active= df[df["is_active"]== 1]
df_inactive= df[df["is_active"] == 0]
st.write(f"Active Employees: {len(df_active)}")
st.write(f"Inactive Employees: {len(df_inactive)}")
status = st.sidebar.radio("aActive Status",["All","Active Only", "Inactive Only"], key="status_radio")
if status == "Active Only":
    filtered_df=df[df["is_active"] == 1]
elif status == "Inactive Only":
    filtered_df = df[df["is_active"] ==0]
else:
    filtered_df= df
st.subheader("Filtered Dataset View")
st.dataframe(filtered_df)


st.subheader("Data Visualization")
fig, ax = plt.subplots(figsize=(8, 4))
avg_salary = filtered_df.groupby("department")["salary"].mean()

ax.bar(avg_salary.index, avg_salary.values, color="teal", edgecolor="black")
ax.set_title("Average Salary by Department")
ax.set_xlabel("Department")
ax.set_ylabel("Salary")
ax.grid(axis="y", linestyle="--", alpha=0.7)

st.pyplot(fig)

st.divider()
st.header("Machine Learning Predictions")
model_choice = st.selectbox(
    "Choose a Model to Run:", 
    ["Decision Tree Classifier", "Random Forest Classifier", "Linear Regression"]
)

if model_choice in ["Decision Tree Classifier", "Random Forest Classifier"]:
    st.subheader(f"{model_choice} (Predicting Active Status)")
    
    X = pd.get_dummies(df[["department", "gender", "salary"]], drop_first=True)
    y = df["is_active"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_choice == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    st.success(f"**Model Accuracy:** {acc * 100:.2f}%")
    
    st.write("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())

elif model_choice == "Linear Regression":
    st.subheader("Linear Regression (Predicting Salary)")
    reg_df=df[["department", "gender", "is_active", "salary"]].dropna()
    
    X = pd.get_dummies(reg_df[["department", "gender", "is_active"]], drop_first=True)
    y = pd.to_numeric(reg_df["salary"], errors='coerce')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    st.info(f"**Mean Absolute Error (MAE):** Off by about ${mae:,.2f} on average.")
    
    st.write("Sample Predictions vs Actual:")
    results_df = pd.DataFrame({
        "Actual Salary": y_test, 
        "Predicted Salary": [round(pred) for pred in y_pred]
    })
    st.dataframe(results_df)
