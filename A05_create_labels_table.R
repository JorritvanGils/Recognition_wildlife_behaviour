# Jorrit van Gils 2021
# Thesis, Wageningen University
# BindRDS train
# 17/09/2021


######### install & load packages  ######### 
{
  if(!"tidyverse" %in% rownames(installed.packages())){install.packages("tidyverse")}
  library(tidyverse)
}


######## read test- and train data  ######## 
{
  #original destination of labels
  labelsTrain <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/images/processed/train/imgPose.rds")
  labelsTest <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/images/processed/test/imgPose.rds")
}


######## correct for label mistakes ########
{
  #I made 1 annotation error on the  80th image of labelsTest
  labelsTest$behaviour[80] <- "f"
  #I made 1 annotation error on the 380th image of labelsTrain
  labelsTrain$behaviour[380] <- "f"
  #Not needed here but handy to know: change multiple values
  #labelsTest$behaviour[c(80,??,??)] <- "f"
}


######## bind train and test labels ########
{
  labelsTrain$train=FALSE
  labelsTest$train=TRUE
  labels <- bind_rows(labelsTrain, 
                      labelsTest)
  #labels %>% drop_na()
}


###### value changes and columnames   ###### 
{
  labels <- labels %>% 
    # Remove " - Copy" in name  path to obtain same structured file_names
    mutate(image = gsub(image, pattern=" - Copy", replacement="")) %>% 
    # column name 'image' to 'file_name' 
    rename(file_name = image) %>% 
    mutate(multimedia_id = file_name) %>% 
    # Remove ".jpg" in values column 'multimedia_id'
    mutate(multimedia_id = gsub(multimedia_id, pattern=".jpg", replacement="")) %>% 
    rename(in_validation_set = train)
  
}


####### left_join() with obs_deer  ####### WARNING!
#!!!run variable obs_deer from '01_DataV10_JG.R'!!! 
{
  labels <- left_join(labels, select(obs_deer, multimedia_id, file_path, deployment_id, sequence_id, location_name, timestamp), 
                      by="multimedia_id")
  labels <- labels %>% 
    rename(path = file_path) 
}


########### Output folder labels ###########
{
  # define a new output folder for the merged (train+test) labels 
  labelFolder = "data/processed/labels_behaviour"
  
  # Create the folder 'labels_behaviour' in path 'data/processed/' if it does not exist
  if(!dir.exists(file.path(main_dir, labelFolder))){dir.create(file.path(main_dir, labelFolder), recursive=TRUE)}
}


########## save labels tibble as rds/csv ##########
{
  # saving labels as RDS
  saveRDS(labels, file=file.path(main_dir, labelFolder, "labels_Reddeer_JG.rds"))
  # saving labels as csv
  write.csv(labels, file=file.path(main_dir, labelFolder, "labels_Reddeer_JG.csv"))
}


########## open labels tibble as rds/csv ##########

{
  # obtaining the data from file location
  labels <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/processed/labels_behaviour/labels_Reddeer_JG.rds")
}


############## end of script  ############## 
