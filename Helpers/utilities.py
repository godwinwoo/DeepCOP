from sklearn.metrics import roc_curve as roc, auc, precision_recall_curve
from scipy.stats import linregress
import numpy as np

def all_stats(labels,scores,cutoff=None):
    if np.unique(labels).shape[0]>1:
      #print np.unique(labels)
      if np.unique(labels).shape[0]==2:
       #print len(np.unique(labels))
       fpr, tpr, thresholds_roc = roc(labels,scores)
       precision, recall, thresholds = precision_recall_curve(labels,scores)
       precision[np.where(precision==0)]=0.000000001
       recall[np.where(recall==0)]=0.000000001
       if len(thresholds)>1:
        F_score=2*(precision*recall)/(precision+recall)
        try:
            if cutoff == None:
                #cutoff=round(thresholds_roc[np.where(abs(tpr-0.95)==min((abs(tpr-0.95))))][0],5)
                #print "Calculation cutoff of maximum F-score"
                cutoff=round(thresholds[np.where(F_score==max(F_score))][0],5)
            else:
                print ("Using cutoff from previous calculations",cutoff)
            aucvalue=round(auc(fpr, tpr),3)
            cutoff_id = np.where(abs(thresholds_roc-cutoff)==min(abs(thresholds_roc-cutoff)))
            cutoff_pre_id = np.where(abs(thresholds-cutoff)==min(abs(thresholds-cutoff)))
            TPR=round(tpr[cutoff_id][0],5)
            FPR=round(fpr[cutoff_id][0],5)
            PRE=round(precision[cutoff_pre_id][0],5)
            stats=aucvalue,TPR,1-FPR,len(labels),PRE,cutoff, max(F_score)
        except:
            stats=float('NaN'),float('NaN'),float('NaN'),len(labels),float('NaN'),float('NaN')
       else:
        stats=float('NaN'),float('NaN'),float('NaN'),len(labels),float('NaN'),float('NaN')
      else:
            gradient, intercept, r_value, p_value, std_err = linregress(labels,scores)
            std_err=np.std((labels-scores))
            stats=r_value**2,std_err,gradient,len(labels),p_value,float('NaN')

    else:
        stats=[float('NaN'),float('NaN'),float('NaN'),len(labels),float('NaN'),float('NaN')]
    return np.array(stats)
