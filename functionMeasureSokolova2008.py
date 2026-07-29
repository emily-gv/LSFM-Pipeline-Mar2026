import numpy as np
import os
from PIL import Image

def functionMeasureSokolova2008_counts(imgPredictionMulti, imgLabelsMulti, classes):
    numClasses = len(classes)
    tps = np.zeros(numClasses)
    tns = np.zeros(numClasses)
    fps = np.zeros(numClasses)
    fns = np.zeros(numClasses)

    for iterClass in range(numClasses):
        iClass = classes[iterClass]

        tp = np.where(np.logical_and(imgPredictionMulti == iClass, imgLabelsMulti == iClass), 1, 0)
        tn = np.where(np.logical_and(imgPredictionMulti != iClass, imgLabelsMulti != iClass), 1, 0)
        fp = np.where(np.logical_and(imgPredictionMulti == iClass, imgLabelsMulti != iClass), 1, 0)
        fn = np.where(np.logical_and(imgPredictionMulti != iClass, imgLabelsMulti == iClass), 1, 0)

        tps[iterClass] = np.sum(tp)
        tns[iterClass] = np.sum(tn)
        fps[iterClass] = np.sum(fp)
        fns[iterClass] = np.sum(fn)
        return tps, tns, fps, fns

def functionMeasureSokolova2008FromCounts(tps, tns, fps, fns):
    """
    Turns accumulated (or single-image) TP/TN/FP/FN arrays into the
    Sokolova & Lapalme (2009) macro/micro metrics.
    """
    numClasses = len(tps)
    eps = 1e-10
 
    numAvgAcc = 0
    numPrecM = 0
    numRecaM = 0
    for iterClass in range(numClasses):
        numAvgAcc += (tps[iterClass] + tns[iterClass]) / (
            tps[iterClass] + fps[iterClass] + tns[iterClass] + fns[iterClass] + eps)
        numPrecM += tps[iterClass] / (tps[iterClass] + fps[iterClass] + eps)
        numRecaM += tps[iterClass] / (tps[iterClass] + fns[iterClass] + eps)
 
    avgAcc = numAvgAcc / numClasses
    precM = numPrecM / numClasses
    RecaM = numRecaM / numClasses
 
    beta = 1
    FscoreM = ((1 + beta * beta) * precM * RecaM) / ((beta * beta * precM) + RecaM + eps)
 
    beta = 2
    FscoreMBeta2 = ((1 + beta * beta) * precM * RecaM) / ((beta * beta * precM) + RecaM + eps)
 
    precisionMu = np.sum(tps) / (np.sum(tps) + np.sum(fps) + eps)
    recallMu = np.sum(tps) / (np.sum(tps) + np.sum(fns) + eps)
 
    beta = 1
    FScoreMu = ((1 + beta * beta) * precisionMu * recallMu) / ((beta * beta * precisionMu) + recallMu + eps)
 
    beta = 2
    FscoreMuBeta2 = ((1 + beta * beta) * precisionMu * recallMu) / ((beta * beta * precisionMu) + recallMu + eps)
 
    return avgAcc, FScoreMu, FscoreM, FscoreMuBeta2, FscoreMBeta2

def functionMeasureSokolova2008(imgPredictionMulti, imgLabelsMulti):
    classes = np.union1d(np.unique(imgLabelsMulti), np.unique(imgPredictionMulti))
    numClasses = len(classes)
    tps = np.zeros(numClasses)
    tns = np.zeros(numClasses)
    fps = np.zeros(numClasses)
    fns = np.zeros(numClasses)
    totalClass = np.zeros(numClasses)

    for iterClass in range(numClasses):
        iClass = classes[iterClass]

        tp = np.where(np.logical_and(imgPredictionMulti == iClass, imgLabelsMulti == iClass), 1, 0)
        tn = np.where(np.logical_and(imgPredictionMulti != iClass, imgLabelsMulti != iClass), 1, 0)
        fp = np.where(np.logical_and(imgPredictionMulti == iClass, imgLabelsMulti != iClass), 1, 0)
        fn = np.where(np.logical_and(imgPredictionMulti != iClass, imgLabelsMulti == iClass), 1, 0)

        tps[iterClass] = np.sum(tp)
        tns[iterClass] = np.sum(tn)
        fps[iterClass] = np.sum(fp)
        fns[iterClass] = np.sum(fn)
        totalClass[iterClass] = np.sum(np.where(imgLabelsMulti == iClass, 1, 0))

    numAvgAcc = 0
    numPrecM = 0
    numRecaM = 0
    
    for iterClass in range(numClasses):
        numAvgAcc = np.float64(numAvgAcc + (tps[iterClass] + tns[iterClass])/(
            tps[iterClass] + fps[iterClass] + tns[iterClass] +fns[iterClass]))
        numPrecM = np.float64(numPrecM + tps[iterClass]/(tps[iterClass] + fps[iterClass] + 0.0000000001))
        numRecaM = np.float64(numRecaM + tps[iterClass]/(tps[iterClass] + fns[iterClass] + 0.0000000001))

    avgAcc = numAvgAcc/numClasses
    precM = numPrecM/numClasses
    RecaM = numRecaM/numClasses

    beta = 1 #favorezco precision
    FscoreM = ((1+beta*beta) * precM * RecaM) / ((beta*beta*precM) + RecaM )

    beta = 2
    FscoreMBeta2 = ((1+beta*beta) *precM *RecaM) / ((beta*beta*precM) + RecaM)

    precisionMu = np.float64(np.sum(tps) / (np.sum(tps) + np.sum(fps)))
    recallMu = np.float64(np.sum(tps) / (np.sum(tps) + np.sum(fns)))

    beta = 1
    FScoreMu = ((1+beta*beta) * precisionMu * recallMu) / ((beta*beta*precisionMu) + recallMu)

    beta = 2
    FscoreMuBeta2 = ((1+beta*beta) * precisionMu * recallMu) / ((beta*beta*precisionMu) + recallMu)
    return avgAcc, FScoreMu, FscoreM, FscoreMuBeta2, FscoreMBeta2

def load_label_mask(fullpath):
    return np.asarray(Image.open(fullpath))

if __name__ == "__main__":
    folder_a = r"/home/emilygv/Desktop/INTER_OBS_ANALYSIS/TestSet_Obs1_3000"
    folder_b = r"/home/emilygv/Desktop/INTER_OBS_ANALYSIS/TestSet_Obs2_3000"

    files_a = sorted(f for f in os.listdir(folder_a) if f.endswith("_mask.png"))
    files_b = sorted(f for f in os.listdir(folder_b) if f.endswith("_mask.png"))
    assert len(files_a) == len(files_b), "Annotator A/B mask lists must match"

    classes = np.array([0,1,2])

    tps_total = np.zeros(len(classes))
    tns_total = np.zeros(len(classes))
    fps_total = np.zeros(len(classes))
    fns_total = np.zeros(len(classes))

    for fa, fb in zip(files_a, files_b):
        labels_a = load_label_mask(os.path.join(folder_a, fa))
        labels_b = load_label_mask(os.path.join(folder_b, fb))
        tps, tns, fps, fns = functionMeasureSokolova2008_counts(labels_a,labels_b, classes)

        tps_total += tps
        tns_total += tns
        fps_total += fps
        fns_total += fns

        # Optional: per-image metrics too, if you want to inspect individual files
        avgAcc_i, FScoreMu_i, FscoreM_i, FscoreMuBeta2_i, FscoreMBeta2_i = functionMeasureSokolova2008FromCounts(tps, tns, fps, fns)
        print(f"{fa}: avgAcc={avgAcc_i:.4f}, F1_micro={FScoreMu_i:.4f}, F1_macro={FscoreM_i:.4f}")

        # Cumulative (pooled) metrics across the whole folder
    avgAcc, FScoreMu, FscoreM, FscoreMuBeta2, FscoreMBeta2 = functionMeasureSokolova2008FromCounts(
        tps_total, tns_total, fps_total, fns_total
    )
 
    print("\n--- Cumulative metrics across folder ---")
    print(f"Average Accuracy: {avgAcc:.4f}")
    print(f"F1 (micro): {FScoreMu:.4f}")
    print(f"F1 (macro): {FscoreM:.4f}")
    print(f"F-beta=2 (micro): {FscoreMuBeta2:.4f}")
    print(f"F-beta=2 (macro): {FscoreMBeta2:.4f}")