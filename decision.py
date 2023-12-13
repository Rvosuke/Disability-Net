import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    """
    Calculate the benefit obtained from the model.

    Return: 
    net_benefit_model : Benefits obtained at different thresholds.

    Parameters
    ----------
    thresh_group : Different thresholds for comparison with y_pred_score to obtain predicted labels

    y_pred_score : Predictive probability of positive class

    y_label : True labels
    """
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model

def calculate_net_benefit_all(thresh_group, y_label):
    """
    Calculate the benefit of treating all samples

    Return : 
    net_benefit_all : Benefits obtained for the entire population at different thresholds


    Parameters
    ----------
    thresh_group : Different thresholds

    y_label : True labels
    """
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all):
    """
    Draw the clinical benefit curve.

    Parameters
    ----------
    thresh_group : Different thresholds
    net_benefit_model : The benefits obtained by the model, returned by the calculate_net_benefit_model() function
    net_benefit_all : Benefits obtained for the entire population at different thresholds, returned by the calculate_net_benefit_all() function
    """
    #Plot
    ax.plot(thresh_group, net_benefit_model, color = 'crimson', label = 'Model')
    ax.plot(thresh_group, net_benefit_all, color = 'black',label = 'Treat all')
    ax.plot((0, 1), (0, 0), color = 'black', linestyle = ':', label = 'Treat none')

    # Display the part where the model outperforms treat all and treat none 
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresh_group, y1, y2, color = 'crimson', alpha = 0.2)

    #Figure Configuration
    ax.set_xlim(0,1)
    ax.set_ylim(net_benefit_model.min() - 0.02, net_benefit_model.max() + 0.15)#adjustify the y axis limitation
    ax.set_xlabel(
        xlabel = 'Threshold Probability', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.set_ylabel(
        ylabel = 'Net Benefit', 
        fontdict= {'family': 'Times New Roman', 'fontsize': 15}
        )
    ax.legend()

    return ax
