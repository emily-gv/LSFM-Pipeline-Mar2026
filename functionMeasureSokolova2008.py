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
    Turns accumulated TP/TN/FP/FN arrays into the
    Sokolova macro/micro metrics.
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
    classes = np.array([0, 1, 2])
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

def load_gt_mask(path):
    """
    Load RGB ground truth.
    Assumes RGB channels contain class values 0,1,2.
    """
    img = np.array(Image.open(path))

    # If RGB, take one channel
    if img.ndim == 3:
        img = img[:, :, 0]

    return img.astype(np.uint8)


def load_prediction_mask(path):
    """
    Load prediction grayscale PNG.
    Converts:
    0   -> background
    50  -> mesenchyme
    100 -> neural ectoderm
    """

    img = np.array(Image.open(path))

    mask = np.zeros_like(img, dtype=np.uint8)

    mask[img == 0] = 0
    mask[img == 50] = 1
    mask[img == 100] = 2

    return mask

if __name__ == "__main__":
"""
I last used this to test the new low intensity models, so modify it to suit whatever youre doing. 
Here the GT images were the label images (0,1,2) and they were being compared to the segmentations (black, light grey, dark grey) 
which first had to be converted into label images.
"""
    folder_a = r"/home/emily/Desktop/Validation_Images/Low_Intensity_GT" # GROUND TRUTH 
    folder_b = r"/home/emily/Desktop/Validation_Images/Low_Intensity_Segmented_NewModel" # LOW INTENSITY MODEl

    files_a = sorted(f for f in os.listdir(folder_a) if f.endswith("_cropped.png"))
    files_b = sorted(f for f in os.listdir(folder_b) if f.endswith("_cropped.png"))
    assert len(files_a) == len(files_b), "Annotator A/B mask lists must match"

    classes = np.array([0,1,2])

    tps_total = np.zeros(len(classes))
    tns_total = np.zeros(len(classes))
    fps_total = np.zeros(len(classes))
    fns_total = np.zeros(len(classes))

    for fa, fb in zip(files_a, files_b):
        labels_a = load_gt_mask(os.path.join(folder_a, fa))
        labels_b = load_prediction_mask(os.path.join(folder_b, fb))
        tps, tns, fps, fns = functionMeasureSokolova2008_counts(labels_a,labels_b, classes)

        tps_total += tps
        tns_total += tns
        fps_total += fps
        fns_total += fns

        # Optional: per-image metrics too, if you want to inspect individual files
        avgAcc_i, FScoreMu_i, FscoreM_i, FscoreMuBeta2_i, FscoreMBeta2_i = functionMeasureSokolova2008FromCounts(tps, tns, fps, fns)
        print(f"{fa}: avgAcc={avgAcc_i:.4f}, F1_micro={FScoreMu_i:.4f}, F1_macro={FscoreM_i:.4f}")

        # Cumulative (pooled) metrics across the whole folder
    avgAcc, FScoreMu, FscoreM, FscoreMuBeta2, FscoreMBeta2 = functionMeasureSokolova2008FromCounts(tps_total, tns_total, fps_total, fns_total)
 
    print("\n--- Cumulative metrics across folder ---")
    print(f"Average Accuracy: {avgAcc:.4f}")
    print(f"F1 (micro): {FScoreMu:.4f}")
    print(f"F1 (macro): {FscoreM:.4f}")
    print(f"F-beta=2 (micro): {FscoreMuBeta2:.4f}")
    print(f"F-beta=2 (macro): {FscoreMBeta2:.4f}")

    print("\nPer-class metrics")

    class_names = ["Background", "Mesenchyme", "Neural"]

    for i, name in enumerate(class_names):
        precision = tps_total[i] / (tps_total[i] + fps_total[i] + 1e-10)
        recall = tps_total[i] / (tps_total[i] + fns_total[i] + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        print(
            f"{name:12s}"
            f" Precision={precision:.4f}"
            f" Recall={recall:.4f}"
            f" F1={f1:.4f}"
        )