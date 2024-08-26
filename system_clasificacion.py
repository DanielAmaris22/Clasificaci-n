import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
class Modelo_clasificacion:
    def __init__(self):
        pass
    def load_data(self):
        path = "C:/Users/57321/Downloads/Python/Practica2/"
        dataset = pd.read_csv(path + "iris_dataset.csv",sep = ";",decimal=",")
        prueba = pd.read_csv(path + "iris_prueba.csv",sep = ";",decimal=",")
        covariables = [ x for x in dataset.columns if x not in ["y"] ]
        X = dataset.get(covariables)
        y = dataset["y"]
        X_nuevo = prueba.get(covariables)
        y_nuevo = prueba["y"]
        return X, y, X_nuevo, y_nuevo
    
    def estandarizacion(self, X):
        Z = preprocessing.StandardScaler()
        Z.fit(X) 
        X_Z = Z.transform(X) 
        return Z, X_Z
    
    def trainning_model(self):
        X,y,X_nuevo, y_nuevo = self.load_data()

        X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.5)
        Z_1, X_train_Z = self.estandarizacion(X_train)
        X_test_Z = Z_1.transform(X_test)
        modelo1 = LogisticRegression(random_state = 123)
        parametros = {'C' : np.arange(0.1, 5.1, 0.1)}
        grilla1 = GridSearchCV( estimator = modelo1, param_grid = parametros,
                               scoring = make_scorer(accuracy_score), cv = 5, n_jobs = -1)
        grilla1.fit(X_train_Z, y_train)
        Z_2 = preprocessing.StandardScaler() 
        Z_2.fit(X_test)
        X_test_Z = Z_2.transform(X_test) 
        X_train_Z = Z_2.transform(X_train)
        modelo2 = LogisticRegression(random_state=123)
        grilla2 = GridSearchCV( estimator = modelo2, param_grid = parametros,
          scoring = make_scorer(accuracy_score), cv = 5, n_jobs = -1)
        grilla2.fit(X_test_Z,y_test)
        y_hat_test = grilla1.predict(X_test_Z)
        y_hat_train = grilla2.predict(X_train_Z)
        u1 = accuracy_score(y_test, y_hat_test)
        u2 = accuracy_score(y_train, y_hat_train)
        Z, X_Z = self.estandarizacion(X)
        X_nuevo_Z = Z.transform(X_nuevo)
        if np.abs(u1-u2) < 10:
            modelo_completo = LogisticRegression(random_state = 123)
            grilla_completa = GridSearchCV(estimator = modelo_completo, param_grid = parametros,
                                       scoring = make_scorer(accuracy_score), cv = 5,n_jobs = -1)
            grilla_completa.fit(X_Z, y)
        else:
            grilla_completa = LogisticRegression(random_state = 123)
            grilla_completa.fit(X_Z, y)
        return X_nuevo_Z, grilla_completa
    def prediccion(self, X_nuevo_Z, grilla_completa):
        y_hat_nuevo = grilla_completa.predict(X_nuevo_Z)
        return y_hat_nuevo
    def evaluacion (self, y, y_hat_nuevo):
        return accuracy_score(y, y_hat_nuevo)
    def modeloClasificacion(self):
        try:
            X, y, X_nuevo,y_nuevo = self.load_data()
            X_nuevo_Z, grilla_completa = self.trainning_model()
            y_hat_nuevo = self.prediccion(X_nuevo_Z, grilla_completa)
            metrica = self.evaluacion(y_nuevo, y_hat_nuevo)
            return {"Resultado":True, "Precision":metrica}
        except Exception as e:
            return {"Resultado":False, "Error":str(e)}