# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:40:53 2019

@author: oscar
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import random
import math
import pandas as pd
from io import open
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Tratamiento de la informacion - Proyecto")
        self.master.iconbitmap("icon.ico")
        self.master.geometry("400x250")
        self.master.resizable(False, False)
        
        etiqueta = tk.LabelFrame(self.master, text="Abrir archivo")
        etiqueta.pack(expand="no")
        button = tk.Button(etiqueta, text = "Buscar el archivo",command = self.fileDialog)
        button.pack()
 
    def fileDialog(self):
        filename = filedialog.askopenfilename(initialdir =  "/", title = "Seleccionar archivo", filetype = (("txt files","*.txt"),("csv files","*.csv"),("Todos los archivos","*.*")) )
        label = tk.Label(self.master, text = filename)
        label.pack()
        self.fragmentFile(filename)
    
    def fragmentFile(self, filename):
        data = open(filename, "r")
        fill = data.readlines()
        data.close()
        
        self.nElements = fill[0]
        self.nAttributes = fill[1]
        self.nClasses = fill[2]
        
        attrib=""
        for i in range(1, int(self.nAttributes)+1):
            if(i == int(self.nAttributes)):
                attrib = attrib + str(i) + ",clase"
            else:
                attrib = attrib + str(i) + ","
        
        train = open("train.csv", "w")
        train.write(attrib)
        train.write("\n")
        
        test = open("test.csv", "w")
        test.write(attrib)
        test.write("\n")
        
        self.contTr = 0
        contTe = 0
        for i in range(3, int(self.nElements)+3):
            rand = random.randrange(1, 11)
            if (rand <= 8):
                train.write(fill[i])
                self.contTr += 1
            else:
                test.write(fill[i])
                contTe += 1
        
        train.close()
        test.close()
        
        self.windowSetOriginal(contTe)
        
    def windowSetOriginal(self, contTe):
        lblOriginal = tk.Label(self.master, text = "Elementos del conjunto original: "+self.nElements)
        lblOriginal.pack()
        
        lblTraining = tk.Label(self.master, text = "Conjunto de entrenamiento: "+str(self.contTr)+" elementos")
        lblTraining.pack()
        
        lblTest = tk.Label(self.master, text = "Conjunto de prueba: "+str(contTe)+" elementos")
        lblTest.pack()
        
        labelSpace = tk.Label(self.master, text = "\n\n")
        labelSpace.pack()
        
        buttonOriginalData = tk.Button(self.master, text = "Datos Originales",command = self.originalDataSet)
        buttonOriginalData.place(x=50, y=200)
        
        self.ks = ttk.Entry(self.master, width=7)
        self.ks.insert(0, "Val. K")
        self.ks.place(x=278, y=175)
        buttonSmoothedData = tk.Button(self.master, text = "Datos Suavizados",command = self.smoothedData)
        buttonSmoothedData.place(x=250, y=200)
        
    def originalDataSet(self):
        self.window=tk.Tk()
        self.window.title("Conjunto de datos original")
        self.window.geometry("445x200")
        self.window.resizable(False, False)
        self.window.iconbitmap("icon.ico")
        self.window.deiconify()
        
        data = pd.read_csv("train.csv", header=0)        
        listAttrib = data.columns[0:int(self.nAttributes)].tolist()
        lbl1 = ttk.Label(self.window, text="Selecciona un atributo")
        lbl1.place(x=50, y=30)
        lbl2 = ttk.Label(self.window, text="Selecciona un atributo")
        lbl2.place(x=250, y=30)
        
        self.comboxAttr = []
        boxAtt1 = ttk.Combobox(self.window, value=listAttrib)
        boxAtt1.place(x=50, y=50)
        self.comboxAttr.append(boxAtt1)
        boxAtt2 = ttk.Combobox(self.window, value=listAttrib)
        boxAtt2.place(x=250, y=50)
        self.comboxAttr.append(boxAtt2)
        
        btnNaive = ttk.Button(self.window, text="Naive Bayes", command=self.naiveBayes)
        btnNaive.place(x=50, y=90)
        self.k = ttk.Entry(self.window, width=7)
        self.k.insert(0, "Val. K")
        self.k.place(x=345, y=92)
        btnkNN = ttk.Button(self.window, text="K - NN", command=self.kNeighborsClassifier)
        btnkNN.place(x=250, y=90)
        
        graphOr = ttk.Button(self.window, text="Graficar", command=self.graphOriginal)
        graphOr.place(x=183, y=150)
    
    def naiveBayes(self):
        trainData = pd.read_csv("train.csv")
        testData = pd.read_csv("test.csv")
        
        classifier = GaussianNB()
        classifier.fit(trainData.iloc[:,:-1], trainData.iloc[:,-1])
        
        predicted_labels = classifier.predict(testData.iloc[:,:-1])
        
        windowResult=tk.Tk()
        windowResult.title("Naive Bayes - Original")
        windowResult.geometry("300x500")
        windowResult.resizable(False, False)
        windowResult.iconbitmap("icon.ico")
        windowResult.deiconify()
        
        score = '{:.5f}'.format(classifier.score(testData.iloc[:,:-1], testData.iloc[:,-1]))

        lbl = ttk.Label(windowResult, text="Precisión en el set de Prueba: "+str(score))
        lbl.place(x=30, y=20)
        lbl = ttk.Label(windowResult, text="Total de Muestras en Test: "+str(testData.shape[0]))
        lbl.place(x=30, y=60)
        lbl = ttk.Label(windowResult, text="Fallos: "+str((testData.iloc[:,-1] != predicted_labels).sum()))
        lbl.place(x=30, y=80)
        lbl = ttk.Label(windowResult, text="Matriz de confusión: ")
        lbl.place(x=30, y=120)
        lbl = ttk.Label(windowResult, text=str(confusion_matrix(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=140)
        lbl = ttk.Label(windowResult, text="Informe de clasificación: ")
        lbl.place(x=30, y=270)
        lbl = ttk.Label(windowResult, text=str(classification_report(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=290)

    def kNeighborsClassifier(self):
        trainData = pd.read_csv("train.csv")
        testData = pd.read_csv("test.csv")
        
        n_neighbors = int(self.k.get())
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(trainData.iloc[:,:-1], trainData.iloc[:,-1])
        
        predicted_labels = knn.predict(testData.iloc[:,:-1])
        
        windowResult=tk.Tk()
        windowResult.title("k-Neighbors N. - Original")
        windowResult.geometry("300x500")
        windowResult.resizable(False, False)
        windowResult.iconbitmap("icon.ico")
        windowResult.deiconify()
        
        score = '{:.5f}'.format(knn.score(testData.iloc[:,:-1], testData.iloc[:,-1]))

        lbl = ttk.Label(windowResult, text="Precisión en el set de Prueba: "+str(score))
        lbl.place(x=30, y=20)
        lbl = ttk.Label(windowResult, text="Total de Muestras en Test: "+str(testData.shape[0]))
        lbl.place(x=30, y=60)
        lbl = ttk.Label(windowResult, text="Fallos: "+str((testData.iloc[:,-1] != predicted_labels).sum()))
        lbl.place(x=30, y=80)
        lbl = ttk.Label(windowResult, text="Matriz de confusión: ")
        lbl.place(x=30, y=120)
        lbl = ttk.Label(windowResult, text=str(confusion_matrix(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=140)
        lbl = ttk.Label(windowResult, text="Informe de clasificación: ")
        lbl.place(x=30, y=270)
        lbl = ttk.Label(windowResult, text=str(classification_report(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=290)
        
        
    def graphOriginal(self):
        data = pd.read_csv("train.csv", header=0)
        colOrd = data.columns.tolist()
        colOrd = colOrd[-1:] + colOrd[:-1]
        data = data[colOrd]
        
        color = ["b","g","r","c","m","y","k"]
        symbol = [".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
        setClass = []
        
        attr1 = int(self.comboxAttr[0].get())
        attr2 = int(self.comboxAttr[1].get())
        
        self.colorGr = []
        self.markerGr = []
        
        for i in range(int(self.nClasses)):
            self.colorGr.append(random.choice(color))
            self.markerGr.append(random.choice(symbol))
        
        plt.figure()
        
        for i in range(int(self.nClasses)):
            setClass.append(data[data['clase'] == i])
            x=setClass[i].iloc[:,attr1]
            y=setClass[i].iloc[:,attr2]
            plt.scatter(x, y, c=self.colorGr[i], marker=self.markerGr[i], label='Clase '+str(i))
     
        plt.title('Gráfica de dispersión - Original')
        plt.xlabel('Atributo '+ str(attr1))
        plt.ylabel('Atributo '+ str(attr2))
        plt.legend(loc='upper center')
        plt.show()
    
    def smoothedDataSet(self):
        self.window=tk.Tk()
        self.window.title("Conjunto de datos suavizados")
        self.window.geometry("445x200")
        self.window.resizable(False, False)
        self.window.iconbitmap("icon.ico")
        self.window.deiconify()
        
        data = pd.read_csv("smoothedTrain.csv", header=0)        
        listAttrib = data.columns[0:int(self.nAttributes)].tolist()
        lbl1 = ttk.Label(self.window, text="Selecciona un atributo")
        lbl1.place(x=50, y=30)
        lbl2 = ttk.Label(self.window, text="Selecciona un atributo")
        lbl2.place(x=250, y=30)
        
        self.comboxAttr = []
        boxAtt1 = ttk.Combobox(self.window, value=listAttrib)
        boxAtt1.place(x=50, y=50)
        self.comboxAttr.append(boxAtt1)
        boxAtt2 = ttk.Combobox(self.window, value=listAttrib)
        boxAtt2.place(x=250, y=50)
        self.comboxAttr.append(boxAtt2)
        
        btnNaive = ttk.Button(self.window, text="Naive Bayes", command=self.naiveBayesS)
        btnNaive.place(x=50, y=90)
        self.k = ttk.Entry(self.window, width=7)
        self.k.insert(0, "Val. K")
        self.k.place(x=345, y=92)
        btnkNN = ttk.Button(self.window, text="K - NN", command=self.kNeighborsClassifierS)
        btnkNN.place(x=250, y=90)
        
        graphOr = ttk.Button(self.window, text="Graficar", command=self.graphSmoth)
        graphOr.place(x=183, y=150)
        
    
    def smoothedData(self):
        trainData = pd.read_csv("train.csv")

        k = int(self.ks.get())
        
        originalArr = trainData.to_numpy()
        copyArr = trainData
        
        sumAttr = 0.0
        neighbors = []
        
        for h in range (int(self.contTr)):
            for i in range (int(self.contTr)):
                for j in range (int(self.nAttributes)):
                    sumAttr += math.pow(originalArr[i][j]-originalArr[h][j], 2);
                dist = math.sqrt(sumAttr)
                sumAttr = 0.0
                
                if(len(neighbors) == k):
                    neighbors.append([dist,originalArr[i][int(self.nAttributes)]])
                    neighbors.sort(key=lambda neighbor: neighbor[0])
                    neighbors.pop()
                else:
                    neighbors.append([dist,originalArr[i][int(self.nAttributes)]])
            
            cont = 0
            contAnt = 0
            for c in range (int(self.nClasses)):
                for x in range(k):
                    if(neighbors[x][1] == c):
                        cont += 1
                if(cont > contAnt):
                    contAnt = cont
                    claseMayor = c
                cont = 0
                
            
            if(claseMayor != originalArr[h][int(self.nAttributes)]):
                copyArr.drop([h], inplace=True)
                
            neighbors.clear()
        
        copyArr.to_csv('smoothedTrain.csv', header=True, index=False)
        
        self.smoothedDataSet()
    
    def graphSmoth(self):
        data = pd.read_csv("smoothedTrain.csv", header=0)
        colOrd = data.columns.tolist()
        colOrd = colOrd[-1:] + colOrd[:-1]
        data = data[colOrd]
        
        setClass = []
        
        attr1 = int(self.comboxAttr[0].get())
        attr2 = int(self.comboxAttr[1].get())
        
        plt.figure()
        
        for i in range(int(self.nClasses)):
            setClass.append(data[data['clase'] == i])
            x=setClass[i].iloc[:,attr1]
            y=setClass[i].iloc[:,attr2]
            plt.scatter(x, y, c=self.colorGr[i], marker=self.markerGr[i], label='Clase '+str(i))
     
        plt.title('Gráfica de dispersión - Suavizado')
        plt.xlabel('Atributo '+ str(attr1))
        plt.ylabel('Atributo '+ str(attr2))
        plt.legend(loc='upper center')
        plt.show()
        
    def naiveBayesS(self):
        trainData = pd.read_csv("smoothedTrain.csv")
        testData = pd.read_csv("test.csv")
        
        classifier = GaussianNB()
        classifier.fit(trainData.iloc[:,:-1], trainData.iloc[:,-1])
        
        predicted_labels = classifier.predict(testData.iloc[:,:-1])
        
        windowResult=tk.Tk()
        windowResult.title("Naive Bayes - Suavizado")
        windowResult.geometry("300x500")
        windowResult.resizable(False, False)
        windowResult.iconbitmap("icon.ico")
        windowResult.deiconify()
        
        score = '{:.5f}'.format(classifier.score(testData.iloc[:,:-1], testData.iloc[:,-1]))

        lbl = ttk.Label(windowResult, text="Precisión en el set de Prueba: "+str(score))
        lbl.place(x=30, y=20)
        lbl = ttk.Label(windowResult, text="Total de Muestras en Test: "+str(testData.shape[0]))
        lbl.place(x=30, y=60)
        lbl = ttk.Label(windowResult, text="Fallos: "+str((testData.iloc[:,-1] != predicted_labels).sum()))
        lbl.place(x=30, y=80)
        lbl = ttk.Label(windowResult, text="Matriz de confusión: ")
        lbl.place(x=30, y=120)
        lbl = ttk.Label(windowResult, text=str(confusion_matrix(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=140)
        lbl = ttk.Label(windowResult, text="Informe de clasificación: ")
        lbl.place(x=30, y=270)
        lbl = ttk.Label(windowResult, text=str(classification_report(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=290)

    def kNeighborsClassifierS(self):
        trainData = pd.read_csv("smoothedTrain.csv")
        testData = pd.read_csv("test.csv")
        
        n_neighbors = int(self.k.get())
        knn = KNeighborsClassifier(n_neighbors)
        knn.fit(trainData.iloc[:,:-1], trainData.iloc[:,-1])
        
        predicted_labels = knn.predict(testData.iloc[:,:-1])
        
        windowResult=tk.Tk()
        windowResult.title("k-Neighbors N. - Suavizado")
        windowResult.geometry("300x500")
        windowResult.resizable(False, False)
        windowResult.iconbitmap("icon.ico")
        windowResult.deiconify()
        
        score = '{:.5f}'.format(knn.score(testData.iloc[:,:-1], testData.iloc[:,-1]))

        lbl = ttk.Label(windowResult, text="Precisión en el set de Prueba: "+str(score))
        lbl.place(x=30, y=20)
        lbl = ttk.Label(windowResult, text="Total de Muestras en Test: "+str(testData.shape[0]))
        lbl.place(x=30, y=60)
        lbl = ttk.Label(windowResult, text="Fallos: "+str((testData.iloc[:,-1] != predicted_labels).sum()))
        lbl.place(x=30, y=80)
        lbl = ttk.Label(windowResult, text="Matriz de confusión: ")
        lbl.place(x=30, y=120)
        lbl = ttk.Label(windowResult, text=str(confusion_matrix(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=140)
        lbl = ttk.Label(windowResult, text="Informe de clasificación: ")
        lbl.place(x=30, y=270)
        lbl = ttk.Label(windowResult, text=str(classification_report(testData.iloc[:,-1], predicted_labels)))
        lbl.place(x=30, y=290)

        
root = tk.Tk()
app = Application(master=root)
app.mainloop()
