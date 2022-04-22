# Jorrit van Gils 2021
# Thesis, Wageningen University
# Data collection
# 10/10/2019


######### install & load packages  ######### 
{
  if(!"imager" %in% rownames(installed.packages())){install.packages("imager")} #load.image + image crop
  library(imager) 
}


############# select folder train (1) or test (2) ####### Choose here!

# run this script twice, once for train resulting in: 
# data/images/processed/train/imgPose.rds
# and once for test resulting in:
# data/images/processed/test/imgPose.rds

{
  #train(1)
  seq_unique  <- unique(dat_train$sequence_id); mainFolder = "data/images/downloads/train" #(1)
  save_image_folder <- gsub(mainFolder, pattern="downloads", replacement = "processed")
  
}

{
  #test(2)
  seq_unique  <- unique(dat_test$sequence_id); mainFolder = "data/images/downloads/test" #(2)
  save_image_folder <- gsub(mainFolder, pattern="downloads", replacement = "processed")
}


#### retrieve, plot and annotate images ####

#loop over the folder data/images/processed/train or data/images/processed/test
# per image chose a behaviour and behaviour_sub
# 05Bind_labels will merge these tibbles together

{
  
  # define labels
  labelOptions <- c(moving = "m", foraging = "f", other = "o")
  
  #ADD
  labelOptionsSub <- c(running = "ru", walking = "w", scanning = "sc",
                       browsing = "b", grazing = "gra", roaring = "ro",
                       sitting = "si", grooming = "gro", standing = "st",
                       vigilance = "v", camera_watching = "c")
  
  # list all image names in folder
  imglist <- list.files(save_image_folder)
  
  # create empty data.frame/tibble with pose per image
  imgPose <- tibble(i = seq_along(imglist), 
                    image = imglist,
                    behaviour = factor(NA_character_, levels = as.character(labelOptions)),
                    behaviour_sub = factor(NA_character_, levels = as.character(labelOptionsSub)))
  
  imgPose
  levels(imgPose$behaviour)
  
  #ADD
  # imgPose
  # levels(imgPose$behaviour_sub)
  

  
  # Close all plotting devices
  while (!is.null(dev.list()))  dev.off()

  
  # loop over all images in save_image_folder
  for(i in imgPose$i[is.na(imgPose$behaviour)]) # seq_along generates numerical values for images in imglist
  {
    # load image
    aimg <- load.image(file.path(main_dir, save_image_folder, imglist[i]))
    
    # plot image (resize to make it faster?)
    thmb <- resize(aimg, -100, -100) # negative numbers are percentages
    plot(thmb, main=i, axes=FALSE, interpolate=FALSE)
    # display(aimg) # faster but in separate window
    # plot(as.raster(aimg))
    
    # Annotate
    correctAnnotation <- FALSE
    while(correctAnnotation == FALSE)
    {
      annotation <- readline(prompt = paste0("image ",i," -- enter label: \n", paste(paste0(names(labelOptions), " = ", labelOptions), collapse = ", "),"\n"))
      if(annotation %in% labelOptions)
      {
        annotation <- factor(annotation, levels = as.character(labelOptions))
        correctAnnotation <- TRUE
      }else
      {
        cat("INCORRECT label, enter correct label!")
      }
    }
    
    # Store annotation in column behaviour
    imgPose$behaviour[i] <- annotation
    
    #ADD behaviour_sub info
    # Annotate
    correctAnnotation <- FALSE
    while(correctAnnotation == FALSE)
    {
      annotation <- readline(prompt = paste0("imgage ",i," -- enter sub_label: \n", paste(labelOptionsSub, collapse = ", "),"\n"))
      if(annotation %in% labelOptionsSub)
      {
        annotation <- factor(annotation, levels = as.character(labelOptionsSub))
        correctAnnotation <- TRUE
      }else
      {
        cat("INCORRECT label, enter correct label!")
      }
    }
    
    # Store annotation in column behaviour_sub
    imgPose$behaviour_sub[i] <- annotation
    
    
    
    # Close all plotting devices
    while (!is.null(dev.list()))  dev.off()
  }
  
  # SAVE imgPose
  saveRDS(imgPose, file=file.path(main_dir, save_image_folder, "imgPose.rds"))
  
}


############## end of script  ############## 
