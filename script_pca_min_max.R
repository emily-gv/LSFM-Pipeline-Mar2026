# Authorship of this code belongs to Marta, I just repurposed it for the Nosip analysis

setwd('/home/emily/Desktop/SUMMER_2026/Sim/Sim/Affine/SyN')

library(Morpho)

# Load data >> this is your output from script_pca_on_proliferation.py
apoptosis <- (as.matrix(read.csv("Nosip_10.5_Mask_WholeHead_X_matrix_down3.csv", row.names=1)))

### MIX / MAX PCA ###
PCA_apoptosis <- (prcomp(apoptosis, center = TRUE, scale.=FALSE))
scores <- PCA_apoptosis$x
loadings <- PCA_apoptosis$rotation
mean_vector <- colMeans()

# Change number to whichever PC you're looking at
PC2_min <- min(scores[,2]) 
PC2_max <- max(scores[,2])

synthetic_scores <- matrix(rep(colMeans(scores), each = 1), nrow = 1)
synthetic_scores[2] <- max(scores[,2]) # Re: change number to whichever PC you're looking at
synthetic_data <- synthetic_scores %*% t(loadings) + mean_vector
# write.csv(synthetic_data, "PCA_apoptosis_nosips_PC2_max_wholeHead_down3.csv")
write.table(synthetic_data, file="PCA_apoptosis_nosips_PC2_max_wholeHead_down3.csv", sep  =",", row.names=FALSE, col.names=FALSE)

synthetic_scores <- matrix(rep(colMeans(scores), each = 1), nrow = 1)
synthetic_scores[2] <- min(scores[,2]) # Re: change number to whichever PC you're looking at
synthetic_data <- synthetic_scores %*% t(loadings) + mean_vector
write.table(synthetic_data, file="PCA_apoptosis_nosips_PC2_min_wholeHead_down3.csv", sep  =",", row.names=FALSE, col.names=FALSE)

### CVA ###
# Was testing this but didn't follow up on it

# genotype <- c("WT", "Het", "Het", "Null", "Het", "Null", "Null", "WT", "Het", "Het", "Het", "WT", "Het", "WT", "Het", "Het", "Null", "Het", "Null", "Null", "WT", "Het", "Het", "Het", "WT", "Het")
# severity <- c('Mild', 'Mild', 'Mild', 'Mild', "Mild", "Severe", "Severe", "Mild", "Mild", "Severe", "Mild", "Mild", "Mild", 'Mild', 'Mild', 'Mild', 'Mild', "Mild", "Severe", "Severe", "Mild", "Mild", "Severe", "Mild", "Mild", "Mild")

# groups <- as.factor(interaction(genotype, severity))
# groups <- droplevels(groups)

# groupings <- CVA(apoptosis, groups, plots=TRUE, rounds=10000, cv=TRUE)
