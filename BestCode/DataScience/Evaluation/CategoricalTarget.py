from matplotlib import pyplot as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def confusion(y_test, pred, labels):
    ''' Predict
    labels = list(set(y_train))  
    '''
    conf_matrix = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix for Best Model')
    plt.show()

def ROC(y_test, pred, labels):
    ''' Predict Proba '''
    if len(labels) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_test, pred[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Best Model')
        plt.legend(loc="lower right")
        plt.show()
    elif len(labels) > 2:  # Multi-class classification
        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(len(labels)):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot ROC curves
        plt.figure()
        for i, color in zip(range(len(labels)), ['blue', 'green', 'red', 'cyan', 'magenta']):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for Best Model')
        plt.legend(loc="lower right")
        plt.show()
        