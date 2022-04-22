# Jorrit van Gils 2021
# Thesis, Wageningen University
# Download preparation
# 10/10/2019

#Scripts 01_Data, 02_download, 03_collect_and_preprocess
#04_labeling, and 05_bind_labels have been run for you.
#The final result is in the code below (line 10/13) a tibble/rds 'labels':

{
  # obtaining the data from file location
  labels <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/processed/labels_behaviour/labels_Reddeer_JG.rds")
}

# You could also follow the scripts 01,02,03,04,05 to obtain the same labels


############################################
############# start of script   ############ 



######### install & load packages  ######### 
{
  if(!"tidyverse" %in% rownames(installed.packages())){install.packages("tidyverse")}
  if(!"lubridate" %in% rownames(installed.packages())){install.packages("lubridate")}
  if(!"rsample" %in% rownames(installed.packages())){install.packages("rsample")} #group_vfold_cv
  library(tidyverse)
  library(lubridate)
  library(rsample) 
}

 
########## set working directory ###########
{
  main_dir = "C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils"
  #create sub-folder for downloading
  sub_dir<-"downloads"
  setwd(main_dir) 
}

 
############## load the data  ##############
{
  obs_dat <- read_csv("data/raw/hoge-veluwe-wildlife-monitoring-project-20210722055531/observations.csv")
  assets_dat <- read_csv("data/raw/hoge-veluwe-wildlife-monitoring-project-20210722055531/multimedia.csv")
  dep_dat <- read_csv("data/raw/hoge-veluwe-wildlife-monitoring-project-20210722055531/deployments.csv")
}


######### filter observations.csv ##########
{
  base_url = "https://www.agouti.eu/#/project/e1730e39-e15d-41b4-bfeb-4a65912e5553/annotate/sequence/"
  
  #From 454904 obs. and 20 var. to 6895 obs. of 9 var.
  obs_deer <- obs_dat %>% 
    #Select desired attributes 
    select(timestamp, deployment_id, sequence_id, scientific_name, count) %>% 
    
    #Filter for "Red deer'
    filter(scientific_name == "Cervus elaphus") %>% 
    
    #Add attribute year 
    mutate(year = year(timestamp)) %>%
    
    #Filter =out duplicate rows, smaller data set with same amount of info
    unique() %>% 
    
    #Filter observations with 1 animal and 1 unique value of count per sequence_id
    group_by(sequence_id) %>% 
    mutate(n=n(),#records in group
           n_count_unique = length(unique(count))) %>% #length unique values count
    ungroup() %>% 
    filter(count == 1, n_count_unique == 1)
  
  #join column "location_id" from dep_dat to obs_deer with join keys "deployments_id"
  obs_deer <- left_join(obs_deer, select(dep_dat, deployment_id, location_name), 
                        by="deployment_id")
  
  #Join columns "multimedia_id", "sequence_id", "file_path" and "file_name" from 
  #assets_dat to obs_deer with join keys "sequence_id"
  assets_dat_filter <- filter(assets_dat, sequence_id %in% obs_deer$sequence_id)
  obs_deer <- assets_dat_filter %>% 
    select(multimedia_id, sequence_id, file_path, file_name) %>% 
    left_join(obs_deer, by = "sequence_id")
  
  #Drop collumns that are not needed anymore
  drops <- c("n","n_count_unique")
  obs_deer[ , !(names(obs_deer) %in% drops)]
  
}


####### prepare train- and test set ######## INSERT!
{
  
  #Set seed to allow for exact repetitive image results
  set.seed(1234567)
  #perform the split, V=2, meaning 50/50 split. 
  #group by location_name: dat_train and dat_test contain data from different cameras 
  dat_split <- group_vfold_cv(obs_deer, group = location_name, v = 5)
  #first row dat_split:[173607/52117] analysis = left-, assessment = right value
  dat_train <- analysis(dat_split$splits[[1]])
  dat_test <- assessment(dat_split$splits[[1]])
  
  #downsize the chosen set
  dat_train <- dat_train %>%
    
    #select collumns of interest
    select(sequence_id, deployment_id, timestamp, location_name) %>%
    
    #Filter for 1 value per sequence_id
    group_by(sequence_id) %>%
    slice_head(n=1) %>%
    ungroup() %>%
    
    #filter for data before 2019
    filter(year(timestamp) < 2019) %>%
    
    #per year, per location max 10 sequences
    group_by(year(timestamp), location_name) %>%
    slice_sample(n=10) %>%
    ungroup() %>%
    
    #----CHOOSE HERE THE SAMPLE SIZE OF dat_train (n = ...) ----
  slice_sample(n=700)
  
  dat_test <- dat_test %>%
    select(sequence_id, deployment_id, timestamp, location_name) %>%
    group_by(sequence_id) %>%
    slice_head(n=1) %>%
    ungroup() %>%
    filter(year(timestamp) >= 2019) %>%
    group_by(year(timestamp), location_name) %>%
    slice_sample(n=10) %>%
    ungroup() %>%
    
    #----CHOOSE HERE THE SAMPLE SIZE OF dat_test (n = ...) ----
  slice_sample(n=175)
}


########## download preparations  ########## INSERT!
{
  # CHOOSE 1: WHICH SET DO YOU WANT TO DOWNLOAD: dat_train(1) or dat_test (2) 
  seq_unique  <- unique(dat_train$sequence_id); mainFolder = "data/images/downloads/train" #(1)
}

{
  seq_unique  <- unique(dat_test$sequence_id); mainFolder = "data/images/downloads/test" #(2)
}

############## end of script  ############## 
