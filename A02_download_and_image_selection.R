# Jorrit van Gils 2021
# Thesis, Wageningen University
# Download images
# 10/10/2019


# *************************WARNING!**************************************
# RUN THIS STEP ONLY ONCE. AFTER THIS STEP YOU WILL SELECT USEFULL IMAGES 
# MANUALLY. RUNNING THIS SCRIPT AGAIN WILL OVERWRITE YOUR SELECTION


######### install & load packages  ######### 
{
  if(!"utils" %in% rownames(installed.packages())){install.packages("utils")} #download.file
  library(utils) 
}


####### download images to folders  ######## 
{
  
  #Create the folder 'downloads(/train or /test) if it does not exist
  #recursive=TRUE: also list files in subfolders
  if(!dir.exists(file.path(main_dir, mainFolder))){dir.create(file.path(main_dir, mainFolder), recursive=TRUE)}
  
  #Time the for-loop
  tstart <- Sys.time()
  
  # first loop: Per "sequence_id" create a folder 
  for (seq_id in seq_unique){
    # seq_id = seq_unique[1]
    
    # Keep track of progress in console
    cat("\n", match(seq_id, seq_unique), "of", length(seq_unique))
    
    # get subset (focal) of obs_deer
    obs_deer_focal <- obs_deer %>%
      filter(sequence_id == seq_id)
    
    # Create sub/-folders with name "sequence_id"
    if(!dir.exists(file.path(main_dir, mainFolder, seq_id))){dir.create(file.path(main_dir, mainFolder, seq_id))}
    
    # second loop: Download images to folders corresponding with "sequence_id"
    for(seq_img in obs_deer_focal$file_path)
    {
      y <- strsplit(seq_img, split = "/")[[1]]
      download.file(url = seq_img, destfile = file.path(main_dir, mainFolder, seq_id, paste0(y[5],".jpg")),  method ="curl", quiet = TRUE)
    }
  }
  tend <- Sys.time()
  tend - tstart
  
  dt = (tend - tstart)
  Sys.time() + dt/10*6000
  
}

{
  
  ######### manual image selection   ######### 
  #
  # After downloading the images go to the folders:  
  #   -data/images/downloads/train  
  #   -data/images/downloads/test  
  # In order to create a list with images that can be used, select maximum 1 image 
  # per sequence_id folder  
  # Try to select images based on behavior of interest  
  # ideally an equal amount of training images per category  
  #   -E.g. train: +-150 moving, +- 150 foraging +- 150 other  
  #   -E.g. test : +- 50 moving, +-  50 foraging +-  50 other  
  # 
  # Most images are category 'moving', also 'foraging' is abundant.  
  # The category 'other' is more rare. Always choose images from 'other' category  
  # and sometimes prefer 'foraging' over 'moving' to end up with a balanced dataset! 
  #   
  # Delete:
  #   -images that despite the selection for count = "1",still contain multiple animals
  #   -images that are blurry
  #   -images in which only a small part of the animal is visible
  #   -images in which the animal is partly out of the screen
  # 
  # Sometimes a sequence_id folder becomes empty (no worries!)
  # 
  # Tests of the remaining images with DL object detection model YOLO showed
  # some images contained multiple animals that where hard to detect with the
  # human eye.
  # 
  # Therefore these sequence_id images have also been excluded:
  #   
  # *  055e2d5d-85ac-4006-b1ca-9cb9dd586c77
  # *  4ebcb62a-68dd-4062-b4f6-81e52d6df094
  # *  d257e9bb-2aa8-4382-923e-adcc4d33a58f
  # *  ee30e8d0-38e4-4f94-a375-98947531edfd
  # *  e421d01a-d5d7-474e-a320-d00a7cb67587
  # *  4c6ea438-4060-4724-b06f-66413fe5613a
  # *  494ba3fc-d5f7-4767-bd6d-1dfcd26902ce
  # *  f23bdc85-8b0c-4c47-8163-897adc3f4171
  # *  4ab11f57-8142-4184-8110-fa7734ed6fc3
  # *  06730a7a-30bc-429d-8892-8f7b01b61be9
  # *  f8430bf3-7a81-4f32-950a-b5134770e79b
  # *  1ef1938f-1e72-48f4-8d8c-543745fe1c12
  # *  6791100f-9162-47c8-b5d0-703f47c213e4
  # *  917c37e3-9652-44fd-bf46-b99d1ad1b25c
  # *  9454de69-71bb-41dd-b8f4-fad9c9b38dbe
  # *  9c55c28a-45e1-4a3d-8262-f5994c28d67d
  # *  30f50b5c-2114-4f21-8e81-39e53bfe9dbc
  # *  af8ed74a-a089-4c39-971d-ee579d85d30d
  # *  b8b4060b-3341-4aa1-a633-e12cd04b918a
  # 
  # now that you have finalized your selection, the selected images from the 
  # folder train or test are collected and preprocessed 
  
}

############## end of script  ############## 