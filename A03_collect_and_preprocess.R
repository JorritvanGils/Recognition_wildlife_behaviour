# Jorrit van Gils 2021
# Thesis, Wageningen University
# Collect_and_preprocess
# 10/10/2019

######### install & load packages  ######### 
{
  if(!"imager" %in% rownames(installed.packages())){install.packages("imager")} #load.image + image crop
  library(imager) 
}


############# select folder train (1) or test (2) ####### Choose here!
{
  # From here, run this script twice, once for train resulting in: 
  # -data/images/processed/train (1)
  seq_unique  <- unique(dat_train$sequence_id); mainFolder = "data/images/downloads/train" #(1)
}

{
  # and once for test resulting in:
  # -data/images/processed/test (2)
  seq_unique  <- unique(dat_test$sequence_id); mainFolder = "data/images/downloads/test" #(2)
}

######## collect and crop  images  #########
{
  #obtain all images that 
  #change here the folder (train or test) from downloads to processed
  save_image_folder <- gsub(mainFolder, pattern="downloads", replacement = "processed")
  
  #Create the folder 'processesed(/train or /test) if it does not exist
  if(!dir.exists(file.path(main_dir, save_image_folder))){dir.create(file.path(main_dir, save_image_folder), recursive=TRUE)}
  
  # list folders in  main folder
  seqFolders <- list.files(file.path(main_dir, mainFolder), recursive=FALSE)
  
  # Loop thought the folders and load/process images
  for(iseq in seqFolders)
  {
    # list files in this sequences folder
    iseqImages <- list.files(file.path(main_dir, mainFolder, iseq), recursive=FALSE)
    
    # iterate through the images and do stuff
    for(iseqimg in iseqImages)
    {
      # load image
      aimg <- load.image(file.path(main_dir, mainFolder, iseq, iseqimg))
      
      # crop image
      cropTop = 40
      cropBottom = 70
      aimgsub <- imsub(aimg,
                       x %inr% c(0,dim(aimg)[1]), # left-right
                       y %inr% c(cropTop, dim(aimg)[2]-cropBottom)) # top-bottom
      
      # save image (all in 1 folder?)
      save.image(aimgsub, file=file.path(main_dir, save_image_folder , iseqimg), quality=0.7)
    }
  }
}


############## end of script  ############## 