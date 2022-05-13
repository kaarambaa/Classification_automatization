# MIT License
#
# Copyright (c) 2022 Konstantin Kovalev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from PyQt5_GUI import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QMessageBox
from utils import *
from Container_utils import *
from ML_utils import models
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.model_selection import train_test_split
import warnings

from PyQt5.QtWidgets import QHeaderView
import joblib

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.container = {}
        self.resampling_alg = 'None'
        self.setupUi(self)
        self.generateButton.clicked.connect(self.EnableCreation)
        self.createButton.clicked.connect(self.CreateDS)
        self.openFileButton.clicked.connect(self.LoadFile)
        self.df_view.setSortingEnabled(True)
        self.encodeButton.clicked.connect(self.performEncoding)
        self.fillDeleteBox.currentTextChanged.connect(self.handleFixevent)
        self.fillDeleteColumnButton.clicked.connect(self.fillDrop)
        self.targetButton.clicked.connect(self.selectTarget)
        self.checkBalanceButton.clicked.connect(self.chekBalance)
        self.trainTestSplitButton.clicked.connect(self.MakeSplit)
        self.balanceButton.clicked.connect(self.fixBalance)
        self.finishPrepocessing.clicked.connect(self.setModelAvaliable)
        self.applyButton.clicked.connect(self.CrossVal)
        self.fitButton.clicked.connect(self.FitModel)
        self.tuneButton.clicked.connect(self.FineTune)
        self.saveData.clicked.connect(self.SaveRF)
        self.saveData_2.clicked.connect(self.SaveGB)
        self.saveData_3.clicked.connect(self.SaveAB)
        self.saveData_4.clicked.connect(self.SaveXGB)
        self.saveData_5.clicked.connect(self.SaveLGBM)

    def popUpMessage(self,titleText,mainText,addText):
        error = QMessageBox()
        error.setWindowTitle(titleText)
        error.setText(mainText)
        error.setInformativeText(addText)
        error.setIcon(QMessageBox.Warning)
        error.setStandardButtons(QMessageBox.Close)
        error.exec_()


    def EnableCreation(self):
        self.hideBox.show()
        self.hideLabel2.show()
        self.hideLabel3.show()
        self.n_features.show()
        self.n_elements.show()
        self.createFileName.show()
        self.createButton.show()
        self.generateButton.setEnabled(False)


    def DisableCreation(self):
        self.hideBox.hide()
        self.hideLabel2.hide()
        self.hideLabel3.hide()
        self.n_features.hide()
        self.n_elements.hide()
        self.createFileName.hide()
        self.createButton.hide()
        self.generateButton.setEnabled(True)


    def CreateDS(self):
        try:
            createFile(n_features=int(self.n_features.text()),n_samples=int(self.n_elements.text()),fname=self.createFileName.text())
            self.generateButton.setEnabled(True)
        except Exception as e:
             self.popUpMessage('Data generation error', 'Dataset cannot be created',
                               'Please check if all fields are filled correctly')


    def AddItem(self, str, isCheckable):
        if isCheckable:
            item = QtWidgets.QListWidgetItem(str)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item = QtWidgets.QListWidgetItem(str)
        return item


    def UpdateData(self,df):
        model = PandasModel(df.round(3))
        self.df_view.setModel(model)
        header = self.df_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.targetColumnList.clear()
        self.fillDeleteColumnList.clear()
        self.encodeList.clear()
        for i in range(df.columns.shape[0]):
            info_encode = f'{df.columns[i]} -- {df.dtypes.values[i]}'
            prcnt_miss = np.mean(df[df.columns[i]].isnull())*100
            info_fill = f'{df.columns[i]} -- missing: {prcnt_miss}%'
            itemTarget = self.AddItem(info_encode, False)
            itemFill = self.AddItem(info_fill, True)
            itemEncode = self.AddItem(info_encode, True)
            self.targetColumnList.addItem(itemTarget)
            self.fillDeleteColumnList.addItem(itemFill)
            self.encodeList.addItem(itemEncode)
        addFile(self.container, 'DF', df)


    def LoadFile(self):
        try:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.MainWidget,   "Open File", "", "CSV Files (*.csv)");
            self.enterFilename.setText(fileName)
            df = pd.read_csv(fileName)
            self.UpdateData(df)
            self.targetButton.setEnabled(True)
            self.fillDeleteColumnButton.setEnabled(True)
            self.encodeButton.setEnabled(True)
            self.checkBalanceButton.setEnabled(True)
            self.trainTestSplitButton.setEnabled(True)
        except Exception as e:
            self.popUpMessage('File upload error','File has not been founded','Please check the file path or its name')


    def selectColumns(self,field):
        checked_items = []
        for index in range(field.count()):
            if field.item(index).checkState() == QtCore.Qt.CheckState.Checked:
                feature = field.item(index).text().split(" -- ")[0]
                checked_items.append(feature)
        return checked_items


    def selectTarget(self):
        df = getFile(self.container, 'DF')
        target = self.targetColumnList.selectedItems()
        if len(target) > 1:
            self.popUpMessage('Target value error','Inappropriate number of features have been selected',
                              'You can select only one target feature')
        elif len(target) == 0:
            self.popUpMessage('Target value error', 'Inappropriate number of features have been selected',
                              'You haven\'t choose any feature')
        else:
            t = target[0].text().split(" -- ")[0]
            Y = df[t]
            addFile(self.container,'Y',Y)
            self.targetButton.setText('Re-select target feature')


    def handleFixevent(self):
        mode = self.fillDeleteBox.currentText()
        if mode == 'Fill with value':
            self.fillValueLine.show()
        else:
            self.fillValueLine.hide()


    def fillDrop(self):
        df = getFile(self.container,'DF')
        mode = str(self.fillDeleteBox.currentText())
        if mode == 'Fill with mean':
            checked_items = self.selectColumns(self.fillDeleteColumnList)
            cols = df[checked_items].columns[df[checked_items].isnull().any()].tolist()
            for c in cols:
                if (df[c].dtype == 'int64') or (df[c].dtype == 'float64'):
                    df[c].fillna(df[c].mean(), inplace = True)
                else:
                    self.popUpMessage('Inplace error','Object type has been selected','You can\'t inplace NaN values with mean in "Object" type Series')
        elif mode == 'Fill with value':
            value = self.fillValueLine.text()
            if len(value) > 0:
                checked_items = self.selectColumns(self.fillDeleteColumnList)
                cols = df[checked_items].columns[df[checked_items].isnull().any()].tolist()
                for c in cols:
                    if (df[c].dtype == 'int64') or (df[c].dtype == 'float64'):
                        df[c].fillna(value, inplace = True)
                    else:
                        df[c].fillna(str(value), inplace = True)
            else:
                self.popUpMessage('Filling error','You haven\'t provide any value','')
        elif mode == 'Delete':
            checked_items = self.selectColumns(self.fillDeleteColumnList)
            df.drop(checked_items, axis = 1, inplace = True)

        self.UpdateData(df)


    def performEncoding(self):
        df = getFile(self.container,'DF')
        mode = str(self.encodeBox.currentText())
        if mode == 'One-Hot encoding (all)':
            dfObj = df.select_dtypes(exclude=['number', 'datetime64'])
            if dfObj.shape[1] == 0:
                self.popUpMessage("Encode error",'Nothing to encode','No non-numeric columns have been found in the dataset')
            else:
                df.drop(labels=dfObj.columns.values, axis=1, inplace=True)
                df = df.merge(pd.get_dummies(dfObj), left_index=True, right_index=True)
                self.UpdateData(df)

        elif mode == 'One-Hot encoding (selected)':
            checked_items = self.selectColumns(self.encodeList)
            dfObj = df[checked_items].select_dtypes(exclude=['number', 'datetime64'])
            if dfObj.shape[1] == 0:
                error = QMessageBox()
                self.popUpMessage("Encode error",'Nothing to encode','No non-numeric columns have been selected')
            else:
                df.drop(labels=dfObj.columns.values, axis=1, inplace=True)
                df = df.merge(pd.get_dummies(dfObj), left_index=True, right_index=True)
                self.UpdateData(df)

        elif mode == 'Label encoding (selected)':
            checked_items = self.selectColumns(self.encodeList)
            if len(checked_items) == 0:
                self.popUpMessage("Encode error", 'Nothing to encode','No non-numeric columns have been found in the dataset')
            else:
                for c in checked_items:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].values)
                    self.UpdateData(df)

        elif mode == 'Label encoding (all)':
            objCols = df.select_dtypes(exclude=['number', 'datetime64']).columns
            if objCols.shape[0] == 0:
                self.popUpMessage("Encode error", 'Nothing to encode','No non-numeric columns have been found in the dataset')
            else:
                for c in objCols:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].values)
                    self.UpdateData(df)


    def recreateDf(self):
        x_train = getFile(self.container,'x_train')
        y_train = getFile(self.container, 'y_train')
        x_test = getFile(self.container, 'x_test')
        y_test = getFile(self.container, 'y_test')
        X = pd.concat([x_train,x_test],axis=0)
        Y = pd.concat([y_train,y_test],axis=0)
        df_new = pd.concat([X,Y],axis=1).sort_index()
        addFile(self.container, 'Y', Y)
        self.UpdateData(df_new)


    def MakeSplit(self):
        df = getFile(self.container,'DF')
        Y = getFile(self.container,'Y')
        X = df.drop(Y.name, axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=self.trainTestSplit.value())
        addFile(self.container,'x_train',x_train)
        addFile(self.container, 'y_train', y_train)
        addFile(self.container, 'x_test', x_test)
        addFile(self.container, 'y_test', y_test)
        addFile(self.container, 'X', X)
        self.dataBalancing.setEnabled(True)
        self.balanceButton.setEnabled(True)


    def chekBalance(self):
        df = getFile(self.container,'DF')
        if len(mathcing(self.container,'Y')) != 0:
            target = getFile(self.container,'Y')
            pos = target.value_counts().values[1]/df.shape[0]
            neg = target.value_counts().values[0]/df.shape[0]
            if pos < 0.35 or neg < 0.35:
                self.isBalancedLabel.setText(f'Data is imbalanced: {pos:0.3f}/{neg:0.3f} ratio')
                self.isBalancedLabel.adjustSize()
                self.balancingLabel.setEnabled(True)
                self.dataBalancing.setEnabled(True)
                self.balanceButton.setEnabled(True)
            else:
                self.isBalancedLabel.setText(f'Data is balanced: {pos}/{neg} ratio')
                self.finishPrepocessing.setEnabled(True)
        else:
            self.popUpMessage('Preprocessing error','No target feature have been selected','Plese, select target feature before cheking data balance')


    def fixBalance(self):
        x_train = getFile(self.container, 'x_train')
        y_train = getFile(self.container, 'y_train')
        self.resampling_alg = str(self.dataBalancing.currentText())
        if len(mathcing(self.container,'x_train')) != 0:
            if  self.resampling_alg == "Oversampling":
                oversample = RandomOverSampler(sampling_strategy='minority', random_state=101)
                x_res, y_res = oversample.fit_resample(x_train, y_train)
                addFile(self.container,'x_train',x_res)
                addFile(self.container, 'y_train', y_res)
            elif  self.resampling_alg == "Oversampling":
                undersample = RandomOverSampler(sampling_strategy='majority', random_state=101)
                x_res, y_res = undersample.fit_resample(x_train, y_train)
                addFile(self.container,'x_train',x_res)
                addFile(self.container, 'y_train', y_res)
            elif self.resampling_alg == "Over + Under":
                oversample = RandomOverSampler(sampling_strategy=0.5, random_state=101)
                undersample = RandomUnderSampler(sampling_strategy= 1, random_state=101)
                x_res, y_res = oversample.fit_resample(x_train, y_train)
                x_res, y_res = undersample.fit_resample(x_res, y_res)
                addFile(self.container, 'x_train', x_res)
                addFile(self.container, 'y_train', y_res)
            elif self.resampling_alg == "SMOTE":
                oversample = SMOTE(sampling_strategy='minority', random_state=101)
                x_res, y_res = oversample.fit_resample(x_train, y_train)
                addFile(self.container, 'x_train', x_res)
                addFile(self.container, 'y_train', y_res)
            elif self.resampling_alg == "NearMiss":
                undersample = NearMiss(sampling_strategy= 'majority', random_state=101)
                x_res, y_res = undersample.fit_resample(x_train, y_train)
                addFile(self.container, 'x_train', x_res)
                addFile(self.container, 'y_train', y_res)
            self.recreateDf()
            self.finishPrepocessing.setEnabled(True)
            print(x_train.shape)
            print(y_train.value_counts())
            print('после ресемплинга:',y_res.value_counts())
        else:
            self.popUpMessage('Preprocessing error','Please split the data into train and test before balancing it','')


    def setModelAvaliable(self):
        self.MainWidget.setTabVisible(2, True)
        self.applyButton.setEnabled(True)


    def CrossVal(self,):

        alg = []
        if self.randomForest.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('RF')
        if self.gradBoosting.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('GB')
        if self.AdaBoost.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('AB')
        if self.XGBoost_2.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('XGB')
        if self.LGBM.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('LGBM')
        n = self.n_folds.value()
        ML = models(algList_=alg, tune_list_ = [],container_=self.container)
        dfCrossVal = ML.crossVal(self.resampling_alg,n,1)
        model_ = PandasModel(dfCrossVal.round(4))
        self.crossValView.setModel(model_)
        header = self.crossValView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.fitButton.setEnabled(True)
        self.tuneButton.setEnabled(True)


    def FineTune(self):
        alg = []
        if self.randomForest_2.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('RF')
        if self.gradBoosting_2.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('GB')
        if self.AdaBoost_2.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('AB')
        if self.XGBoost_3.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('XGB')
        if self.LGBM_2.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('LGBM')
        n = self.n_folds.value()
        ML = models(algList_=[], tune_list_= alg, container_=self.container)
        dfCrossVal = ML.crossVal(self.resampling_alg, n, 2)
        model_ = PandasModel(dfCrossVal.round(4))
        self.crossValView.setModel(model_)
        header = self.crossValView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)


    def DisplayModel(self, metrics, accuracy, precision, recall, roc_auc, f1, ROC_plot, conf_mat):
        while ROC_plot.count():
            child = ROC_plot.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        while conf_mat.count():
            child = conf_mat.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.MainWidget.setTabVisible(3, True)
        acc = metrics['Accuracy']
        prec = metrics['Precision']
        rec = metrics['Recall']
        roc = metrics['ROC_AUC score']
        F1 = metrics['F1']
        accuracy.setText(str(f'Accuracy: {acc:.4f}'))
        precision.setText(str(f'Precision: {prec:.4f}'))
        recall.setText(str(f'Recall: {rec:.4f}'))
        roc_auc.setText(str(f'ROC-AUC score: {roc:.4f}'))
        f1.setText(str(f'F1 score: {F1:.4f}'))
        fig_roc = plotROC_AUC(metrics['ROC_AUC'])
        ROC_AUCView = MplCanvas(fig_roc)
        ROC_plot.addWidget(ROC_AUCView)
        fig_conf = plotConfMat(metrics['Conf'])
        confMatView = MplCanvas(fig_conf)
        conf_mat.addWidget(confMatView)


    def Save(self,model,preds, probas):
        dirname = QtWidgets.QFileDialog.getExistingDirectory(parent=self.MainWidget,caption="Select Folder")
        path_ = dirname+f'/{model.__class__.__name__}_'

        if self.saveModelBox.checkState() == QtCore.Qt.CheckState.Checked:
            path_model = path_ + 'trained_model.sav'
            joblib.dump(model, path_model)
        if self.savePredictionsBox.checkState() == QtCore.Qt.CheckState.Checked:
            path_preds = path_ + 'predictions.csv'
            np.savetxt(path_preds,preds,delimiter=',')
        if self.savePredictionsBox.checkState() == QtCore.Qt.CheckState.Checked:
            path_probas = path_ + 'predicted_probabilities.csv'
            np.savetxt(path_probas,probas,delimiter=',')
        return path_

    def SaveRF(self):
        model = getFile(self.container,'RandomForestClassifier/model')
        preds = getFile(self.container,'RandomForestClassifier/preds')
        probas = getFile(self.container, 'RandomForestClassifier/probas')
        path = self.Save(model,preds,probas)
        self.saveDirLine.setText(path)
        self.saveData.setEnabled(False)


    def SaveGB(self):
        model = getFile(self.container,'GradientBoostingClassifier/model')
        preds = getFile(self.container,'GradientBoostingClassifier/preds')
        probas = getFile(self.container, 'GradientBoostingClassifier/probas')
        path = self.Save(model, preds, probas)
        self.saveDirLine_2.setText(path)
        self.saveData_2.setEnabled(False)


    def SaveAB(self):
        model = getFile(self.container,'AdaBoostClassifier/model')
        preds = getFile(self.container,'AdaBoostClassifier/preds')
        probas = getFile(self.container, 'AdaBoostClassifier/probas')
        path = self.Save(model, preds, probas)
        self.saveDirLine_3.setText(path)
        self.saveData_3.setEnabled(False)


    def SaveXGB(self):
        model = getFile(self.container,'XGBClassifier/model')
        preds = getFile(self.container,'XGBClassifier/preds')
        probas = getFile(self.container, 'XGBClassifier/probas')
        path = self.Save(model, preds, probas)
        self.saveDirLine_4.setText(path)
        self.saveData_4.setEnabled(False)


    def SaveLGBM(self):
        model = getFile(self.container,'LGBMClassifier/model')
        preds = getFile(self.container,'LGBMClassifier/preds')
        probas = getFile(self.container, 'LGBMClassifier/probas')
        path = self.Save(model, preds, probas)
        self.saveDirLine_5.setText(path)
        self.saveData_5.setEnabled(False)


    def FitModel(self):
        # Random forest
        alg = []
        if self.randomForest.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('RF')
        if self.gradBoosting.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('GB')
        if self.AdaBoost.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('AB')
        if self.XGBoost_2.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('XGB')
        if self.LGBM.checkState() == QtCore.Qt.CheckState.Checked:
            alg.append('LGBM')

        ML = models(algList_=alg,container_=self.container)
        metrics = ML.pipeline()

        for m in metrics:
            if m == 'RF':
                self.DisplayModel(metrics[m],self.Accuracy_label,self.Precision_label,self.Recall_label,
                                  self.ROC_AUC_label,self.F1_label,self.HLayoutROC,self.HLayoutConf)
                self.MainWidget.setTabVisible(3, True)
            elif m == 'GB':
                self.DisplayModel(metrics[m], self.Accuracy_label_2, self.Precision_label_2, self.Recall_label_2,
                                      self.ROC_AUC_label_2, self.F1_label_2, self.HLayoutROC_2, self.HLayoutConf_2)
                self.MainWidget.setTabVisible(4, True)
            elif m == 'AB':
                self.DisplayModel(metrics[m], self.Accuracy_label_3, self.Precision_label_3, self.Recall_label_3,
                                      self.ROC_AUC_label_3, self.F1_label_3, self.HLayoutROC_3, self.HLayoutConf_3)
                self.MainWidget.setTabVisible(5, True)
            elif m == 'XGB':
                self.DisplayModel(metrics[m], self.Accuracy_label_4, self.Precision_label_4, self.Recall_label_4,
                                      self.ROC_AUC_label_4, self.F1_label_4, self.HLayoutROC_4, self.HLayoutConf_4)
                self.MainWidget.setTabVisible(6, True)
            elif m == 'LGBM':
                self.DisplayModel(metrics[m], self.Accuracy_label_5, self.Precision_label_5, self.Recall_label_5,
                                      self.ROC_AUC_label_5, self.F1_label_5, self.HLayoutROC_5, self.HLayoutConf_5)
                self.MainWidget.setTabVisible(7, True)



if __name__ == "__main__":
    import sys
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = QtWidgets.QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec_())
