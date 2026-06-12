#Data cleaning Pipeline 

#import library in python 
import pandas as pd
import os
from pathlib import Path


#loading the different datasets into a pandas dataframe
#df= pd.read_csv('../Data/Raw/ransomware.csv', sep=',')
DATA_FILE = Path(__file__).resolve().parent.parent / "Data" / "Raw" / "ransomware.csv"
df = pd.read_csv(DATA_FILE)

# Finding the sum of missing values in each column
print("Missing values in each column: \n",(df.isnull().sum()))

#Taking care of missing values in the pandas dataframe of df 
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

non_numeric_columns =  df.select_dtypes(include='object').columns
df[non_numeric_columns] = df[non_numeric_columns].fillna("unknown")
#checking it is clean from missing values or not
print("Output of the cleaning process:\n",df.isnull().sum())

#showing the duplicates in the dataframe , when there is no duplicates it will return 0 
print("The number of duplicates in the Dataframe is :", df.duplicated().sum())
# there is no duplicates in the dataframe so no need to remove them
print("The types in the data frame:", df.dtypes)


#Summary statistics of the cleaned data
print("Summary statistics of the cleaned data:\n", df.describe(include='all'))
print("The number of rows in the cleaned data:", len(df))
print ("The number of columns in the cleaned data:", len(df.columns))
print (df.info())

# Label encode binary features
from sklearn.preprocessing import LabelEncoder  as LD
le = LD()
y = le.fit_transform(df["Sample_Type"])
df["Sample_Type"] = y

# Save cleaned dataset
df.to_csv("../Data/Processed/cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")


# IMporting necessary libraries for machine learning and i will use abbevertion of libraries to shorten my code length  
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report as CR
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import roc_auc_score as RAS
from sklearn.metrics import roc_curve as RC
from sklearn.metrics import accuracy_score, classification_report


#i will be dropping irrelevant columns and case_solved so it won't cause data/answer leakage  as it would be our target output 
df.drop(columns=['HKEY_CURRENT_USER\Software\Microsoft\RestartManager','CryptDecrypt','WSASocketW','ioctlsocket','NtSuspendThread','WriteConsoleA','connect','select','HttpSendRequestA','RegEnumValueA','NtQueryMultipleValueKey','shutdown','WSASend','FindWindowExA','EnumServicesStatusW','GetAsyncKeyState','CertControlStore','HttpQueryInfoA','ObtainUserAgentString','GetAddrInfoW','Module32NextW','NtGetContextThread','NtWriteVirtualMemory','HttpSendRequestW','FindWindowExW','NtQueryFullAttributesFile','recv','Thread32Next','InternetOpenA','InternetConnectA','WriteProcessMemory','ControlService','InternetReadFile','DecryptMessage','HKEY_CURRENT_USER\Software\Microsoft\CommandProcessor',
                    'HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Run','HttpOpenRequestA','NtSetContextThread','RegDeleteValueA','InternetCrackUrlA','RegQueryInfoKeyA','InternetCrackUrlW','SetFileInformationByHandle','IWbemServices_ExecQuery','RegDeleteKeyA','CertOpenStore','InternetGetConnectedState','OutputDebugStringA','CreateRemoteThreadEx','CryptDecodeObjectEx','sendto','FindWindowA','send','SetStdHandle','InternetSetOptionA','WSARecv','CryptExportKey','PRF',], inplace=True)
X = df.drop(columns=['GetBestInterfaceEx'])  


# Y would house our targetted output 
y = df['Sample_Type']

# Split the data: 80% for training, 20% for testing for quality output 
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.25, random_state=42, stratify=y)


# Initialize the scaler
scaler = SS()
# Fit on training data, transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Performing the first Model - Logistic regression
#  Initialize logistic regression
log_reg = LR(max_iter=1000, random_state=42)

# Fit the model
log_reg.fit(X_train_scaled, y_train)

# Predict
y_pred_log = log_reg.predict(X_test_scaled)



print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_log))


# Performing the second model - Random forest
# Initialize random forest with basic parameters
rf = RFC(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_train, y_train)  # Random forest does not require scaling

# Predict
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))

# Confusion matrix
import matplotlib.pyplot as pyp
import seaborn as sbs

cm = CM(y_test, y_pred_rf)
sbs.heatmap(cm, annot=True, fmt='d', cmap='Blues')
pyp.title("Confusion Matrix - Random Forest")
pyp.xlabel("Predicted")
pyp.ylabel("Actual")
pyp.show()
os.makedirs("../Doc/figure", exist_ok=True)
pyp.savefig("../Doc/figure/confusion_matrix.png", dpi=300, bbox_inches="tight")

# ROC Curve
y_proba_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = RC(y_test, y_proba_rf)
pyp.plot(fpr, tpr, label=f"Random Forest (AUC = {RAS(y_test, y_proba_rf):.2f})")
pyp.plot([0, 1], [0, 1], linestyle='--')
pyp.title("ROC Curve")
pyp.xlabel("False Positive Rate")
pyp.ylabel("True Positive Rate")
pyp.legend()
pyp.grid()
pyp.show()
pyp.savefig("../Doc/figure/roc_curve.png", dpi=300, bbox_inches="tight")

#Tuning  hyperparameters 
from sklearn.model_selection import GridSearchCV as GSCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Grid search with 3-fold CV
grid_search = GSCV(RFC(random_state=42),
                           param_grid,
                           cv=3,
                           scoring='f1',
                           n_jobs=-1)

# Fit
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
print("Best Random Forest Parameters:", grid_search.best_params_)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score

# =======================
# 1. SCALE THE FEATURES
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_data = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# =======================
# 2. DEFINE MODEL
# =======================
class RansomwareNet(nn.Module):
    def __init__(self, input_dim):
        super(RansomwareNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

model = RansomwareNet(X_train_t.shape[1])

# =======================
# 3. LOSS & OPTIMIZER
# =======================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =======================
# 4. TRAINING LOOP
# =======================
epochs = 30
model.train()

for epoch in range(epochs):
    epoch_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# =======================
# 5. PREDICTION
# =======================
model.eval()
with torch.no_grad():
    y_proba_dl = model(X_test_t).numpy().ravel()
    y_pred_dl = (y_proba_dl >= 0.5).astype(int)

# Evaluation
print("AUC:", roc_auc_score(y_test, y_proba_dl))
print(classification_report(y_test, y_pred_dl))

print("\n=== SUMMARY ===")
print("Total ransomware detected:", sum(y_proba_dl))
print("Total benign detected:", len(y_proba_dl) - sum(y_proba_dl))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# =========================
# 1. SCALE FEATURES
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# =========================
# 2. RESIDUAL BLOCK
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return x + self.fc(x)

# =========================
# 3. RESIDUAL MLP MODEL
# =========================
class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.block1 = ResidualBlock(256)
        self.fc2 = nn.Linear(256, 128)
        self.block2 = ResidualBlock(128)
        self.out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.block1(x)
        x = torch.relu(self.fc2(x))
        x = self.block2(x)
        x = self.sigmoid(self.out(x))
        return x

model = ResidualMLP(X_train_t.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# 4. TRAINING LOOP
# =========================
epochs = 30
model.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# =========================
# 5. EVALUATION
# =========================
model.eval()
with torch.no_grad():
    y_proba_res = model(X_test_t).numpy().ravel()
    y_pred_res = (y_proba_res >= 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, y_proba_res))
print(classification_report(y_test, y_pred_res))




class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)      # reshape to (batch, 1, features)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

model_cnn = CNN1D(X_train_t.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

# TRAIN
epochs = 30
model_cnn.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model_cnn(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# EVALUATE
model_cnn.eval()
with torch.no_grad():
    y_proba_cnn = model_cnn(X_test_t).numpy().ravel()
    y_pred_cnn = (y_proba_cnn >= 0.5).astype(int)

print("AUC:", roc_auc_score(y_test, y_proba_cnn))
print(classification_report(y_test, y_pred_cnn))

