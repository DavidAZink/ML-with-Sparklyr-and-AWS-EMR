install.packages('plyr', dependencies = T)
install.packages('dplyr', dependencies = T)
install.packages('devtools')
install.packages('sparklyr')
devtools::install_github("rstudio/sparklyr")
devtools::dev_mode(on=T)

library(sparklyr)
library(dplyr)
Sys.setenv(SPARK_HOME="/usr/lib/spark")

#connect to the cluster in yarn-client mode. I launched a cluster in Amazon EMR consisting of 1 m5.xlarge master node and 10 c5.4xlarge core nodes. 
config <- spark_config()
sc <- spark_connect(master = "yarn-client", config = config, version = '2.4.5')



#All data for this project is stored in AWS S3. Below I specify the paths for each of the datasets used to train the ML model. 
#The final dataset has about 7 million observations and is too large for me to store on GitHub. Please email me if you would like access to the
#data.

lar2018_path="s3://aws-logs-764479432355-us-east-2/load/lar2018_loanlevel.csv"
census2018_path="s3://aws-logs-764479432355-us-east-2/load/census_lar2018.csv"
lar2019_path="s3://aws-logs-764479432355-us-east-2/load/lar2019_loanlevel.csv"
census2019_path="s3://aws-logs-764479432355-us-east-2/load/census_lar2019.csv"
countydata_path="s3://aws-logs-764479432355-us-east-2/load/county_level_data.csv"


#read in data from amazon s3 and join dataframes together. Store in object called df_tbl. The final dataset consists of loan level data from 2018 and 2019. Loan level data is combined with census
#tract and county level data. Loan level data is indicative of borrower specific characteristics, while census tract and county level data are indicative of the local economic and demographic characteristics
# of the area in which the borrower is located. 

spark_read_csv(sc, path=lar2018_path)%>% 
  inner_join(spark_read_csv(sc, path=census2018_path), by='census_tract') %>% 
  select(-c(`_c0_x`, `_c0_y`)) %>% mutate(FIP=as.numeric(County), Year=2018)->lar2018

spark_read_csv(sc, path=lar2019_path)  %>%
  inner_join(spark_read_csv(sc, path=census2019_path), by=c('Year', 'census_tract')) %>% 
  select(-c(`_c0_x`, `_c0_y`)) %>% mutate(FIP=as.numeric(County)) %>% union_all(lar2018) %>%
  inner_join(spark_read_csv(sc, path=countydata_path), by=c('FIP', 'Year'), suffix=c("_x", "_y")) ->df_tbl


#delete observations with missing labels. Drop some features that I don't need (these features vary only at the year level so are not useful)
df_tbl %>% filter(!is.na(as.numeric(high_risk))) %>% mutate_all(as.numeric)  %>% 
  select(-c(census_tract, FSI, County, FIP, GDP, qtr, OUT00, emi_ffr, emp_pn, d_rate, g_tradable, tradable, tradable_share, CPI, ebp_oa, gz_spr, g_GDP, g2_GDP, 
            g_CPI, g2_CPI, g_ebp_oa, g2_ebp_oa, g_gz_spr, g2_gz_spr, g_FSI, g2_FSI, `_c0`, 'Year')) -> df_tbl


#create a different factor level for categorical features with missing values
df_tbl %>% mutate(Race1=ifelse(is.na(Race1), max(Race1)+1, Race1)) ->df_tbl
df_tbl %>% mutate(PurchaseType=ifelse(is.na(PurchaseType), max(PurchaseType)+1, PurchaseType)) ->df_tbl
df_tbl %>% mutate(LoanType=ifelse(is.na(LoanType), max(LoanType)+1, LoanType)) ->df_tbl
df_tbl %>% mutate(Sex=ifelse(is.na(Sex), max(Sex)+1, Sex)) ->df_tbl
df_tbl %>% mutate(HOEPA=ifelse(is.na(HOEPA), max(HOEPA)+1, HOEPA)) ->df_tbl
df_tbl %>% mutate(Ethnicity1=ifelse(is.na(Ethnicity1), max(Ethnicity1)+1, Ethnicity1)) ->df_tbl


#one hot encode factor variables
for ( i in c('PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1')){ 
  df_tbl %>% ft_one_hot_encoder(input_cols=i, output_cols=paste0(i, '_encoded')) 
}


#split data into test and training sets
data_splits=sdf_random_split(df_tbl, training = 0.8, testing = 0.2, seed = 42)
df_train=data_splits$training
df_test=data_splits$testing

#create observations weights for training data
df_train %>% mutate(weight=ifelse(high_risk==1, sum(as.numeric(high_risk==0))/(sum(as.numeric(!is.na(high_risk)))), sum(as.numeric(high_risk==1))/(sum(as.numeric(!is.na(high_risk)))))) -> df_train

#register training data and load into cached memory
sdf_register(df_train, 'df_train')
tbl_cache(sc, "df_train")
#repartition training data
df_train<-sdf_repartition(df_train, 20)


#pipline object
pipeline <- ml_pipeline(sc) %>%
  #replace missing values with means for non-categoricial features
  ft_imputer(input_cols=colnames(df_train)[! colnames(df_train) %in% c('weight', 'PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 
                                                                       'PurchaseType_encoded', 'LoanType_encoded', 'Sex_encoded', 'Race1_encoded', 'HOEPA_encoded', 'Ethnicity1_encoded')], 
             output_cols=colnames(df_train)[! colnames(df_train) %in% c('weight', 'PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 
                                                                        'PurchaseType_encoded', 'LoanType_encoded', 'Sex_encoded', 'Race1_encoded', 'HOEPA_encoded', 'Ethnicity1_encoded')], 
             strategy="mean") %>%
  #assemble features into a vector, ignoring the label column (high_risk) and all the all the categorical variables (because otherwise they would be included twice since there already is a
  #one hot encoded version of them in the dataframe) 
  ft_vector_assembler(
    input_cols = colnames(df_train)[! colnames(df_train) %in% c('PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 'high_risk', 'weight')], 
    output_col = "features") %>%
  #standardize features to have mean zero and unit variance
  ft_standard_scaler(input_col = "features", output_col = "features_scaled", 
                     with_mean = TRUE)%>%
  #fit weighted logistic regression
  ml_logistic_regression(features_col = "features_scaled", 
                         label_col = "high_risk", weight='weight')


#cross validation
cv <- ml_cross_validator(
  sc,
  estimator = pipeline,
  estimator_param_maps = list(
    standard_scaler = list(with_mean = c(TRUE)),
    #create grid of regulatization parameters for elasticnet logistic regression
    logistic_regression = list(
      elastic_net_param = c(0, 0.15, 0.25, 0.5, 0.75),
      reg_param = c(0.5, 1e-1, 1e-2, 1e-3, 0)
    )
  ),
  evaluator = ml_binary_classification_evaluator(sc, label_col = "high_risk"),
  #10 folds
  num_folds = 10)

#cross validate with training data
cv_model <- ml_fit(cv, df_train)

#sort cross validation models by best to worst performance based on the AUROC
ml_validation_metrics(cv_model) %>% arrange(-areaUnderROC)->performance


#pipeline for full dataset. Everything is the same as the initial pipeline object except this time I define the regulatization parameters in the 
#logistic regression stage to be those selected from the cross validation
pipeline_full <- ml_pipeline(sc) %>%
  ft_imputer(input_cols=colnames(df_train)[! colnames(df_train) %in% c('weight', 'PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 
                                                                       'PurchaseType_encoded', 'LoanType_encoded', 'Sex_encoded', 'Race1_encoded', 'HOEPA_encoded', 'Ethnicity1_encoded')], 
             output_cols=colnames(df_train)[! colnames(df_train) %in% c('weight', 'PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 
                                                                        'PurchaseType_encoded', 'LoanType_encoded', 'Sex_encoded', 'Race1_encoded', 'HOEPA_encoded', 'Ethnicity1_encoded')], 
             strategy="mean") %>%
  ft_vector_assembler(
    input_cols = colnames(df_train)[! colnames(df_train) %in% c('PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 'high_risk', 'weight')], 
    output_col = "features") %>%
  ft_standard_scaler(input_col = "features", output_col = "features_scaled", 
                     with_mean = TRUE)%>%
  ml_logistic_regression(features_col = "features_scaled", 
                         label_col = "high_risk", elastic_net_param=performance$elastic_net_param_1[1], 
                         reg_param=performance$reg_param_1[1], weight_col='weight')


#fit the model with the full training data
optimal_model<-ml_fit(pipeline_full, df_train)


#get predictions with test data
df_test %>%
  ft_vector_assembler(
    input_cols = colnames(df_test)[! colnames(df_test) %in% c('PurchaseType', 'LoanType', 'Sex', 'Race1', 'HOEPA', 'Ethnicity1', 'high_risk', 'weight')], 
    output_col = "features") %>%
  ft_standard_scaler(input_col = "features", output_col = "features_scaled", 
                     with_mean = TRUE) %>%
  ml_evaluate(ml_stage(optimal_model, 'logistic_regression'))->metrics


metrics$area_under_roc()
#AUC is 0.7317513#