# Biblioteca para interagir com recursos do Kaggle
import kagglehub
# Biblioteca para permitindo manipular caminhos de sistema de arquivos
from pathlib import Path
# Biblioteca para importar arquivos
import pandas as pd
# Biblioteca utilizada para dividir um conjunto de dados em dois subconjuntos distintos: um conjunto de treinamento e um conjunto de teste.
from sklearn.model_selection import train_test_split
# Biblioteca utilizada para padronizar características, removendo a média e escalonando para variância unitária.
from sklearn.preprocessing import StandardScaler
# Biblioteca para criar modelo de Regressão Logística
from sklearn.linear_model import LogisticRegression
# Biblioteca para gerar um relatório detalhado das métricas de desempenho de um modelo de classificação
from sklearn.metrics import classification_report
# Biblioteca para otimizar e acelerar fluxos de trabalho que envolvem tarefas computacionalmente intensivas
import joblib

# Download latest version
path = kagglehub.dataset_download("akshaydattatraykhare/diabetes-dataset")
# Criar caminho do arquivo
file_path = Path(path) / 'diabetes.csv'
# Carregar base de dados
df = pd.read_csv(file_path)
# Codicional para remover conteudo desnecessário
condition = (df['BloodPressure'] > 1) & (df['Insulin'] > 1)
# Filtra base de dados
df_filtered = df[condition]
# Remover coluna resultado
X = df_filtered.drop('Outcome', axis=1)
# Criar variável que contenha os resultados
y = df_filtered['Outcome']
# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar modelo de regressão logística
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Avaliação
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Salvar modelo e scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Modelo e scaler salvos com sucesso!")