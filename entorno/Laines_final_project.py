#Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import sklearn
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    #Cargar data
    dt_credit_card = pd.read_csv("./data/creditcard.csv")

    #Explorar data
    print(dt_credit_card.shape)
    print(dt_credit_card.info())
    print(dt_credit_card.head())
    print(dt_credit_card.describe())
    dt_credit_card['Class'].unique()

    #Manipulación de datos
    dt_credit_card.isnull()
    dt_credit_card.isnull().sum()
    ##dt_credit_card.dropna(inplace=True)

    #Distribución de los tipos de clases
    plt.figure(figsize=(10, 6))
    sns.countplot(data = dt_credit_card, x="Class")
    plt.title("Distribución de los tipos de clases")
    plt.xlabel('Clase')
    plt.ylabel('Total')
    plt.show()

    #División de las clases de datos
    fraud = dt_credit_card[dt_credit_card['Class']==1]
    valid = dt_credit_card[dt_credit_card['Class']==0]

    fraction_fraud = len(fraud)/float(len(valid))
    print(fraction_fraud)

    print("Fraud cases: ", len(fraud.index))
    print("Valid cases: ", len(valid.index))

    #Detalles de los montos de transacciones fraudulentas
    print("Amount details of the fraudulent transaction")
    fraud["Amount"].describe()

    #Detalles de los montos de transacciones válidas
    print("Amount details of the valid transaction")
    valid.Amount.describe()

    #Gráficos de las transacciones fraudulentas y válidas

    transaction_f_v = dt_credit_card.groupby("Class").mean()["Amount"]

    graphBar=plt.bar(transaction_f_v.index, transaction_f_v.values, color='green')
    plt.title("Promedio de monto de transacciones fraudulentas y válidas")
    plt.xlabel('Clase')
    plt.ylabel('Monto promedio')
    plt.xticks([0, 1], ["Valid", "Fraud"])

    for graphBar, amount in zip(graphBar, transaction_f_v.values):
        plt.text(graphBar.get_x() + graphBar.get_width() / 2 - 0.15, amount, f'{amount:.2f}', ha='center', va='bottom', color='black')

    plt.show()

    #Matriz de correlación
    corrmat = dt_credit_card.corr()
    sns.heatmap(corrmat, vmax = .8, square = True)
    plt.show()
    plt.savefig("graficoCor.jpg")

    #Regresión logística	
    dt_prueba = dt_credit_card.drop(['Class'],axis=1)
    dt_entrenamiento = dt_credit_card['Class']
    dt_prueba = StandardScaler().fit_transform(dt_prueba)

    x_train,x_test,y_train,y_test=train_test_split(dt_prueba,dt_entrenamiento,test_size=0.2,random_state=42)

    print(dt_prueba.shape)
    print(dt_entrenamiento.shape)

    #Histograma
    transaction_time = dt_credit_card['Time']

    # Crea un histograma con el tiempo de las transacciones
    plt.hist(transaction_time, bins=50, color='blue', alpha=0.7)

    # Agrega etiquetas y título al gráfico
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Número de Transacciones')
    plt.title('Distribución del Tiempo de las Transacciones')

    # Muestra el gráfico
    plt.show()
    plt.savefig("graficoHIST.jpg")

    # Para regresión logísitica aplicar algoritmos IPCA

    ## batch_size: Numero de datos que se van a tomar para entrenar el modelo de PCA en cada iteracion 
    ipca=IncrementalPCA(n_components=3, batch_size=10) 

    ## Ajustar los datos de entrenamiento al modelo de IPCA para que pueda aprender de ellos
    ipca.fit(x_train)

    regresion_logistic=LogisticRegression(solver='lbfgs')

    dt_train=ipca.transform(x_train)
    dt_test=ipca.transform(x_test)  #Transformar los datos de prueba con el modelo de PCA para que pueda aprender de ellos 
    regresion_logistic.fit(dt_train,y_train)  #Ajustar los datos de entrenamiento al modelo de regresion logistica para que pueda aprender de ellos 
    print("Score IPCA",regresion_logistic.score(dt_test,y_test))

    y_pred = regresion_logistic.predict(dt_test)

    # Calcula la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Etiquetas de las clases (en tu caso, 'Normal' y 'Fraud')
    LABELS = ['Normal', 'Fraud']

    # Crea la matriz de confusión visual
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()
    plt.savefig("graficoConf.jpg")