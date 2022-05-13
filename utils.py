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

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QSizePolicy
import pandas as pd
from sklearn.datasets import make_classification
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt
import itertools

class MplCanvas(FigureCanvas):
    def __init__(self,fig,parent = None):
        self.fig = fig
        FigureCanvas.__init__(self,self.fig)
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

def plotROC_AUC(roc_auc):
    fig, axes = plt.subplots()
    fpr,tpr,_ = roc_auc
    line1 = axes.plot(fpr, tpr, label='ROC curve')
    line2 = axes.plot(fpr, fpr, '--', label='Random classifier')
    axes.set_ylabel('True Positive Rate')
    axes.set_xlabel('False Positive Rate')
    axes.legend(loc=4)

    axes.set_ylabel('True Positive Rate')
    axes.set_xlabel('False Positive Rate')
    return fig

def plotConfMat(cnf_matrix):
    fig, axes = plt.subplots()
    axes.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    axes.set_title('Confusion matrix', fontsize=12)
    tick_marks = np.arange(len(['1', '2']))
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        axes.text(j, i, format(cnf_matrix[i, j]),
                  horizontalalignment="center",
                  color="white" if cnf_matrix[i, j] > thresh else "black")
    return fig

def createFile(n_features, n_samples, fname):
    clf = make_classification(n_samples=n_samples, n_features=n_features)
    columns = []
    for x in range(0, clf[0].shape[1]):
        columns.append('F' + str(x))
    dfx = pd.DataFrame(clf[0], columns=columns)
    dfy = pd.DataFrame(clf[1], columns=['target'])
    df = pd.merge(dfx, dfy, left_index=True, right_index=True)
    if '.csv' in fname:
        df.to_csv(fname, sep=',', index=False, float_format="%.3f")
    else:
        df.to_csv(fname+'.csv', sep=',',index=False, float_format="%.3f")



class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()



