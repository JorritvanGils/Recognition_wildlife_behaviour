# Jorrit van Gils 2021
# Thesis, Wageningen University
# Box21_Reddeer_JG_RH
# 28/09/2019

################# Chose directory #################
{
  #Set directory variables (or change to own preferences)
  main_dir = "C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils"
  labelFolder = "data/processed/labels_behaviour"
}


########## open labels tibble as rds/csv ##########

{
  # obtaining the data from file location
  labels <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/processed/labels_behaviour/labels_Reddeer_JG.rds")
}



########## change to needs of RH ##########
{
  labels_rh <- labels %>%
    rename(original_category = behaviour) %>% 
    #add column 'meta'
    add_column(meta = NA) %>%
    select(path, original_category, meta, sequence_id, deployment_id, in_validation_set, multimedia_id) %>% 
    #build the string for 'meta'
    mutate(meta = str_c('{"seq_id": "',sequence_id,'", "depl_id": "',deployment_id,'", "filename": "',multimedia_id,'.JPG"}')) %>% 
    # select the desired columns
    select(path, original_category, meta,  in_validation_set)
}


########## save labels_rh tibble as rds/csv ##########
{
  # saving labels_rh as RDS
  saveRDS(labels_rh, file=file.path(main_dir, labelFolder, "labels_Box21_RH.rds"))
  # saving labels_rh as CSV
  write.csv(labels_rh, file=file.path(main_dir, labelFolder, "labels_Box21_RH.csv"))
}

########## open labels_rh tibble as rds/csv ##########
{
  
  labels_rh <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/processed/labels_behaviour/labels_Box21_RH.rds")
  
}

###############################################
########## Repeat for: behaviour_sub ##########
{
  labels_rh_sub <- labels %>%
    rename(original_category = behaviour_sub) %>% 
    #add column 'meta'
    add_column(meta = NA) %>%
    select(path, original_category, meta, sequence_id, deployment_id, in_validation_set, multimedia_id) %>% 
    #build the string for 'meta'
    mutate(meta = str_c('{"seq_id": "',sequence_id,'", "depl_id": "',deployment_id,'", "filename": "',multimedia_id,'.JPG"}')) %>% 
    # select the desired columns
    select(path, original_category, meta,  in_validation_set)
}


########## save labels_rh_sub tibble as rds/csv ##########
{
  # saving labels_rh_sub as RDS
  saveRDS(labels_rh_sub, file=file.path(main_dir, labelFolder, "labels_sub_Box21_RH.rds"))
  # saving labels_rh_sub as CSV
  write.csv(labels_rh_sub, file=file.path(main_dir, labelFolder, "labels_sub_Box21_RH.csv"))
}

########## open labels_rh tibble as rds/csv ##########
{
  
  labels_rh_sub <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/processed/labels_behaviour/labels_sub_Box21_RH.rds")
  
}


############## end of script  ############## 