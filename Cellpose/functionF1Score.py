import numpy as np

def functionF1Score(imgScore, imgLabels): 
    
    # imgScore = imgSeg
    # imgLabels = imgGT

    #F1 Score
    positives = np.where(imgScore > 0, 1,0)
    negatives = np.where(imgScore <= 0, 1, 0)
    trues = np.where(imgLabels > 0, 1, 0)
    falses = np.where(imgLabels <= 0, 1, 0)

    tp = np.where((imgScore == 1) & (imgLabels==1), 1, 0)
    tn = np.where((imgScore == 0) & (imgLabels==0), 1, 0)
    fp = np.where((imgScore == 1) & (imgLabels==0), 1, 0)
    fn = np.where((imgScore == 0) & (imgLabels==1), 1, 0)

    # if totalTP != totaltp2:
    #     print("not equivalent")
    # else:
    #     print("equivalent")
    
    totalTP = np.sum(tp)
    totalTN = np.sum(tn)
    totalFP = np.sum(fp)
    totalFN = np.sum(fn)

    precision = np.float64(totalTP/(totalTP + totalFP + 0.000001)) # adding 0.000001 avoids divide by zero error
    
    recall = np.float64(totalTP/(totalTP + totalFN + 0.000001))
    f1 = np.float64(2 * ( (precision * recall)/(precision + recall + 0.000001)))
    # print(f1)

    # accuracyOld = (totalTP + totalTN)/len(imgLabels.flatten()) #flatten or along len(imgLabels); flatten is total number of pixels
    accuracy = (totalTP + totalTN)/(totalTP + totalTN + totalFP + totalFN)

    # if accuracy != accuracyOld:
    #     print("not equivalent")
    # else:
    #     print("equivalent")

    beta = 2 #Favorezo recall
    f1beta = np.float64(((1 + beta * beta) * precision * recall)/((beta * beta * precision) + recall + 0.000001))

    beta = 0.5 #Favorezo precision
    f1beta2 = np.float64(((1 + beta * beta) * precision * recall) / ((beta * beta * precision) + recall + 0.000001))

    dice = np.float64((2 * (np.count_nonzero(np.logical_and(positives, trues)))/(np.count_nonzero(positives) + np.count_nonzero(trues))))

    intersecIm = np.logical_and(positives, trues)
    sumIntersec = np.double(np.sum(intersecIm))
    unionIm = np.logical_or(positives, trues)
    sumUnion = np.double(np.sum(unionIm))
    IoU = 0

    if sumUnion > 0:
        IoU = np.float64(sumIntersec/sumUnion)
    else:
        print("union is 0")

    
    # MCC_numerator = np.double((totalTP*totalTN) - (totalFP*totalFP)) # i think this is wrong
    # MCC_denominator =np.sqrt(np.double((totalTP + totalFP) * (totalTP + totalFN) * (totalTN+totalFP) * (totalTN +totalFN)))
    # mcc = MCC_numerator/MCC_denominator
    return f1, precision, recall, accuracy, f1beta, f1beta2, dice, IoU