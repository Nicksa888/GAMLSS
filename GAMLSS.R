# Install and load the gamlss package
install.packages("gamlss")
library(gamlss)

#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all data sets, functions and so on.

####################
####################
# Source Documents #
####################
####################

source("C:/R Portfolio/Global_Terrorism_Prediction/Functions 29 06 23.R")
source("C:/R Portfolio/Global_Terrorism_Prediction/MENA_Data_Object_Locations_13_04_2023.R")

# Data Preparation

# Here we select the Middle East and North Africa Region fro 

# Middle East & North Africa ---------------------------------------------

Region_Name <- "Middle East & North Africa"
MENA <- Region_Prep(GTD_WD, Region_Name)
glimpse(MENA)
MENA$Lethal <- as.integer(MENA$Lethal)
MENA$Lethal[MENA$Lethal %in% "1"] <- "0"
MENA$Lethal[MENA$Lethal %in% "2"] <- "1"
MENA$Lethal <- as.integer(MENA$Lethal)

####################
####################
# One Hot Encoding # 
####################
####################

# Here we convert the categorical variables into binary variables

MENA_Binary <- one_hot(as.data.table(MENA))
glimpse(MENA_Binary)

# Recode some variables #

MENA_Binary <- MENA_Binary %>% 
  rename(
    Islamic_State = `Group_Islamic State of Iraq and the Levant (ISIL)`,
    OtherGroup = Group_OtherGroup,
    Iraq = Country_Iraq,
    Syria = Country_Syria,
    Turkey = Country_Turkey,
    Yemen = Country_Yemen,
    OtherCountry = Country_OtherCountry,
    Iraq_Nationality = Nationality_Iraq,
    Israel_Nationality = Nationality_Israel,
    Turkey_Nationality = Nationality_Turkey,
    Yemen_Nationality = Nationality_Yemen,
    OtherNationality = Nationality_OtherNationality,
    Al_Anbar_Province = Province_Al_Anbar,
    Baghdad_Province = Province_Baghdad,
    Diyala_Province = Province_Diyala,
    Nineveh_Province = Province_Nineveh,
    Saladin_Province = Province_Saladin,
    OtherProvince = Province_OtherProvince,
    Baghdad_City = City_Baghdad,
    OtherCity = City_OtherCity
  )

# Remove unneeded parts of column names

names(MENA_Binary) = gsub(pattern = "Quarter_", 
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Week", 
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Group_", 
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Target_",
                          replacement = "", 
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Attack_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Weapon_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Nationality_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "Province_",
                          replacement = "",
                          x = names(MENA_Binary))
names(MENA_Binary) = gsub(pattern = "City_",
                          replacement = "",
                          x = names(MENA_Binary))

write.csv(MENA_Binary, file = "MENA_Binary.csv", row.names = F)
MENA_Binary <- read.csv("MENA_Binary.csv")

# Final Prediction Data Set #

MENA_Binary_Initial <- dplyr::select(MENA_Binary, -c(all_of(Intial_Columns_Remove)))
MENA_Binary_Initial <- transform(MENA_Binary_Initial, Lethal = as.numeric(Lethal))

# Count NA's

sapply(MENA_Binary_Initial, function(x) sum(is.na(x)))
# Remove NA values

MENA_Binary_Initial <- MENA_Binary_Initial[complete.cases(MENA_Binary_Initial), ]

# zero and near - zero variance features #

# Zero and near-zero variance features refer to variables in a dataset that have either zero or very little variance. These features typically do not provide useful information for predictive modeling and may even hinder the performance of the model. Here's an explanation of both:

# Zero Variance Features:
# These are variables that have the same value for all observations in the dataset. Because the variable does not vary at all, it cannot help differentiate between different observations.
# For example, if a variable has a value of 0 for all observations, it has zero variance.
# Near-Zero Variance Features:
# These are variables that have very little variation in their values across the dataset. While they may have more than one unique value, one value might dominate the majority of the observations, making it nearly constant.
# Near-zero variance features are often undesirable because they don't provide enough information to the model and can lead to overfitting.
# Identifying and removing near-zero variance features can help simplify the model and improve its performance.
# In predictive modeling, it's essential to identify and handle zero and near-zero variance features appropriately. Removing these features can streamline the model, reduce complexity, and improve its ability to generalize to new data. Techniques such as univariate analysis, variance thresholds, and correlation analysis are commonly used to detect and address these issues.

set.seed(555)
feature_variance <- caret::nearZeroVar(MENA_Binary_Initial, saveMetrics = T)
head(feature_variance)
which(feature_variance$zeroVar == 'TRUE')

# There is no near zero or zero variance

# Correlation Test on Runs Train Data #

MENA_Binary_Initial_corr <- cor(MENA_Binary_Initial, method = "spearman")

high_corr_MENA_Binary_Initial <- caret::findCorrelation(MENA_Binary_Initial_corr, cutoff = 0.70)
high_corr_MENA_Binary_Initial

# MENA_Binary_Initial_corr <- cor(MENA_Binary_Initial, method = "spearman"):
#   This line calculates the Spearman correlation coefficient matrix for the dataset MENA_Binary_Initial.
# The method = "spearman" argument specifies that Spearman's rank correlation coefficient is used. Spearman correlation measures the strength and direction of association between two variables, and it's particularly useful for assessing monotonic relationships between variables.
# high_corr_MENA_Binary_Initial <- caret::findCorrelation(MENA_Binary_Initial_corr, cutoff = 0.70):
#   This line applies the findCorrelation function from the caret package to identify highly correlated variables in the correlation matrix MENA_Binary_Initial_corr.
# The cutoff = 0.70 argument specifies the threshold for correlation. Here, any pair of variables with a correlation coefficient greater than or equal to 0.70 (in absolute value) will be considered highly correlated.
# The function returns a logical vector indicating which variables are highly correlated with each other. If two variables are highly correlated, one of them might be redundant for modeling purposes, and removing one can help avoid multicollinearity issues.

# These are the variables that are highlu correlated

#  18 23 29 34 12 15 21  3 20
names(MENA_Binary_Initial)

# "Iraq", "Iraq_Nationality", "Baghdad_Province", "Baghdad_City", "BombAttack", "Explosives", "Yemen", "Islamic_State", "Turkey" 

MENA_Binary_Final <- dplyr::select(MENA_Binary_Initial, -c(all_of(Corr_Columns_Remove)))

# Predict for 2020

# For Training Data, select all data upto and including 2019.
MENA_Train_Year <- MENA_Binary_Final %>% dplyr::filter(Year %in% c(1970:2019)) %>% dplyr::select(-c(Year))
str(MENA_Train_Year)
MENA_Test_Year <- dplyr::filter(MENA_Binary_Final, Year == 2020) %>% dplyr::select(-c(Year))
MENA_Train_Year <- transform(MENA_Train_Year, Lethal = as.numeric(Lethal))
MENA_Test_Year <- transform(MENA_Test_Year, Lethal = as.numeric(Lethal))

# Assuming your data frame is named 'df'
# Define your model formula
formula <- Lethal ~ OtherGroup + Business + GovtGen + OtherTarget + Police + Private + ArmedAssaultAttack + Assassination + HostageKidnapAttack + OtherAttack + Firearms + OtherWeapon + OtherCountry + Syria + Israel_Nationality + OtherNationality + Turkey_Nationality + Yemen_Nationality + Al_Anbar_Province + Diyala_Province + Nineveh_Province + OtherProvince + Saladin_Province + OtherCity

# Fit the GAMLSS model
model <- gamlss(formula = formula, data = MENA_Train_Year, family = "NO")

# Summarize the model
summary(model)

# ******************************************************************
#   Family:  c("NO", "Normal") 
# 
# Call:  gamlss(formula = formula, family = "NO", data = MENA_Train_Year) 
# 
# Fitting method: RS() 
# 
# ------------------------------------------------------------------
#   Mu link function:  identity
# Mu Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)          0.805922   0.009430  85.466  < 2e-16 ***
#   OtherGroup          -0.105953   0.008204 -12.915  < 2e-16 ***
#   Business            -0.029971   0.008872  -3.378  0.00073 ***
#   GovtGen             -0.150323   0.008789 -17.103  < 2e-16 ***
#   OtherTarget         -0.114156   0.006161 -18.530  < 2e-16 ***
#   Police              -0.003639   0.007487  -0.486  0.62691    
# ArmedAssaultAttack   0.008951   0.015146   0.591  0.55453    
# Assassination        0.085453   0.013795   6.194 5.91e-10 ***
#   HostageKidnapAttack -0.360540   0.017837 -20.213  < 2e-16 ***
#   OtherAttack         -0.250433   0.019170 -13.064  < 2e-16 ***
#   Firearms             0.277150   0.014352  19.311  < 2e-16 ***
#   OtherWeapon          0.155033   0.016875   9.187  < 2e-16 ***
#   OtherCountry         0.007384   0.017533   0.421  0.67366    
# Syria                0.220502   0.020166  10.935  < 2e-16 ***
#   Israel_Nationality  -0.284796   0.020522 -13.878  < 2e-16 ***
#   OtherNationality    -0.134830   0.017419  -7.741 1.01e-14 ***
#   Turkey_Nationality  -0.173178   0.012943 -13.380  < 2e-16 ***
#   Yemen_Nationality   -0.103605   0.012508  -8.283  < 2e-16 ***
#   Al_Anbar_Province   -0.118762   0.058948  -2.015  0.04394 *  
#   Diyala_Province     -0.035967   0.058872  -0.611  0.54125    
# Nineveh_Province    -0.132806   0.058957  -2.253  0.02429 *  
#   OtherProvince       -0.128534   0.058734  -2.188  0.02865 *  
#   Saladin_Province    -0.042059   0.058955  -0.713  0.47560    
# OtherCity           -0.014844   0.058397  -0.254  0.79935    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# ------------------------------------------------------------------
#   Sigma link function:  log
# Sigma Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept) -0.773248   0.003651  -211.8   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# ------------------------------------------------------------------
#   No. of observations in the fit:  37520 
# Degrees of Freedom for the fit:  25
# Residual Deg. of Freedom:  37495 
# at cycle:  2 
# 
# Global Deviance:     48452.65 
# AIC:     48502.65 
# SBC:     48715.96 

# This output represents the results of a GAMLSS (Generalized Additive Models for Location Scale and Shape) regression model fitted to your data. Here's how to interpret the key parts:
# 
# Mu (Mean) Model:
# Mu Link Function: Identity
# This means that the mean response variable (Lethal) is modeled directly, without any transformation.
# Mu Coefficients:
# Each coefficient represents the effect of the corresponding predictor variable on the mean response variable.
# The Estimate column shows the estimated effect size.
# The Std. Error column indicates the standard error of the estimate.
# The t value column represents the t-statistic for each coefficient, which measures the significance of the effect.
# The Pr(>|t|) column shows the p-value associated with each coefficient, indicating its significance.
# For example, Assassination has a positive coefficient of 0.085453, indicating that an increase in the occurrence of Assassination events is associated with an increase in the mean response variable Lethal.
# Sigma (Scale) Model:
# Sigma Link Function: Log
# This indicates that the scale parameter (standard deviation) of the response variable is modeled using the log function.
# Sigma Coefficients:
# There's only one coefficient estimated for the scale model, which represents the intercept.
# The Estimate column shows the estimated intercept for the scale model.
# The Std. Error, t value, and Pr(>|t|) columns follow the same interpretation as for the Mu coefficients.
# Model Fit:
#   No. of observations in the fit: 37520
# This indicates the number of observations used in the model fitting process.
# Degrees of Freedom for the fit: 25
# This represents the degrees of freedom associated with the model.
# Residual Degrees of Freedom: 37495
# These are the degrees of freedom associated with the residuals of the model.
# Global Deviance: 48452.65
# This is a measure of the goodness of fit of the model. Lower values indicate better fit.
# AIC (Akaike Information Criterion): 48502.65
# AIC is a measure of the relative quality of a statistical model for a given set of data. Lower AIC values indicate better models.
# SBC (Schwarz Bayesian Criterion): 48715.96
# SBC is similar to AIC but includes a penalty for the number of parameters in the model.
# Interpretation:
#   The coefficients with *** are highly significant (p < 0.001), ** are significant (p < 0.01), and * are marginally significant (p < 0.05).
# For example, variables like OtherGroup, GovtGen, OtherTarget, HostageKidnapAttack, OtherAttack, Firearms, OtherWeapon, Syria, Israel_Nationality, OtherNationality, Turkey_Nationality, Yemen_Nationality, Al_Anbar_Province, Nineveh_Province, OtherProvince are highly significant predictors of Lethal.
# Police, ArmedAssaultAttack, OtherCountry, Diyala_Province, Saladin_Province, OtherCity do not appear to be significant predictors of Lethal as their p-values are above the significance threshold.

# Generalized Additive Models for Location Scale and Shape (GAMLSS) offer several benefits over traditional regression models:
#   
# Flexibility: GAMLSS allow for modeling not only the mean of the response variable but also its variance, skewness, and kurtosis. This flexibility allows for a more comprehensive understanding of the distributional properties of the response variable.
# Accommodates Various Distributions: GAMLSS can handle a wide range of response variable distributions, including but not limited to normal, gamma, beta, and Poisson distributions. This versatility makes GAMLSS suitable for a diverse range of data types and research questions.
# Non-linear Relationships: GAMLSS can capture non-linear relationships between predictors and the response variable through the use of smooth functions, such as splines or polynomial terms. This enables the modeling of complex relationships that may not be adequately captured by linear models.
# Model Interpretability: GAMLSS provide interpretable parameters for each component of the distribution (location, scale, shape), allowing for a deeper understanding of how predictors influence different aspects of the response variable's distribution.
# Robustness: GAMLSS are robust to outliers and heteroscedasticity since they model the entire distribution rather than just the mean. This can lead to more reliable estimates and inferences, especially in the presence of data with non-constant variance.
# Diagnostic Tools: GAMLSS offer a variety of diagnostic tools for model assessment, including residual plots, Q-Q plots, and goodness-of-fit tests for each component of the distribution. These tools help ensure that the model adequately captures the underlying patterns in the data.
# Prediction and Inference: GAMLSS can be used for both prediction and inference tasks. They provide estimates of not only the mean response but also its variability and shape, allowing for more informed predictions and uncertainty quantification.
# Software Support: GAMLSS are supported by various statistical software packages, including R, which provides comprehensive functions for model fitting, visualization, and interpretation.
# Overall, GAMLSS offer a powerful framework for modeling complex data with non-normal distributions and non-linear relationships, providing researchers with valuable insights into the underlying processes driving their data.