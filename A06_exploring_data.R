# Jorrit van Gils 2021
# Thesis, Wageningen University
# Exploring data
# 29/09/2021

######### install & load packages  ######### 
{
  if(!"tidyverse" %in% rownames(installed.packages())){install.packages("tidyverse")}
  library(tidyverse)
}

########## open labels tibble as rds/csv ##########

{
  # obtaining the data from file location
  labels <- readRDS("C:/Users/jorri/OneDrive - Wageningen University & Research/02Thesis/Project_Thesis_JorritvanGils/data/processed/labels_behaviour/labels_Reddeer_JG.rds")
}


###############  Inspecting general  ##############
head(labels, n=3)
class(labels) #type of data object
class(labels$behaviour)
class(labels$file_name)
nrow(labels)
ncol(labels)
length(labels)
length(labels[1 , ])#first row
names(labels)
str(labels)
glimpse(labels) #dyplyr


############## Inspecting categories  #############
levels(labels$behaviour) #behaviour
table(labels$behaviour)
table(labels$behaviour_sub) # behaviour_sub
table(labels$in_validation_set)
labels_table_sub <- table(labels$behaviour_sub, labels$in_validation_set) #USE THIS
labels_table <- table(labels$behaviour, labels$in_validation_set) #USE THIS
xtabs(~labels$behaviour_sub + labels$in_validation_set) #alternatively
xtabs(~labels$in_validation_set + labels$behaviour_sub) #alternatively


####################  Barchart  ##################


# Use behaviour and behaviour_sub to display the abbreveations
# Use behaviour_graph and behaviour_sub_graph to display full names

labels <- labels %>%
  mutate(behaviour_graph = case_when(behaviour == "f" ~ "foraging",
                                 behaviour == "m" ~ "moving",
                                 behaviour == "o" ~ "other",
                                 TRUE ~ "NONE"))
labels <- labels %>%
  mutate(behaviour_sub_graph = case_when(behaviour_sub == "ru" ~ "running",
                                     behaviour_sub == "w" ~ "walking",
                                     behaviour_sub == "sc" ~ "scanning",
                                     behaviour_sub == "b" ~ "browsing",
                                     behaviour_sub == "gra" ~ "grazing",
                                     behaviour_sub == "ro" ~ "roaring",
                                     behaviour_sub == "si" ~ "sitting",
                                     behaviour_sub == "gro" ~ "grooming",
                                     behaviour_sub == "st" ~ "standing",
                                     behaviour_sub == "v" ~ "vigilance",
                                     behaviour_sub == "c" ~ "camera_watching",
                                     TRUE ~ "NONE"))

# display behaviour
#nice but: -behaviour categories x axis are grey
#         - colours are not always clear. Alternative?
labels %>%
  ggplot(mapping = aes(x = behaviour_graph, fill = behaviour_graph)) +
  geom_bar()



ggplot(data = labels,
       mapping = aes(x = behaviour_sub)) +
  geom_bar()


# behaviour most simple form without legend
labels %>%
  ggplot(mapping = aes(x = behaviour_graph)) +
  geom_bar()



# try
labels %>%
  ggplot(mapping = aes(x = behaviour, fill = behaviour_graph)) +
  geom_bar(stat = "identity")

# display behaviour_sub within behaviour bar
labels %>%
  #ggplot old package, + is same as %>%, but not supported
  ggplot(mapping = aes(x = behaviour_graph, fill = behaviour_sub_graph)) +
  geom_bar()

# display behaviour_sub as separated bars
labels %>%
  ggplot(mapping = aes(x = behaviour_sub, fill = behaviour_sub_graph)) + 
  geom_bar() +
  #try to change scales to 'fixed'
  facet_wrap(vars(behaviour_graph), scales = "free")

# display behaviour_sub as separated bars, not colour them
labels %>%
  #ggplot old package, + is same as %>%, but not supported
  ggplot(mapping = aes(x = behaviour_sub_graph)) + 
  geom_bar() +
  #try to change scales to 'fixed'
  facet_wrap(vars(behaviour_graph), scales = "free")





#bar chart
agg <- count(labels, behaviour)
head(agg)

agg <- count(labels, behaviour, behaviour_sub)
head(agg)

agg <- count(labels, behaviour, behaviour_sub, in_validation_set)
head(agg)



ggplot(labels)


labels_table_sub <- table(labels$behaviour_sub)
data_frame <- as.data.frame(labels_table_sub)
y_lim <- max(data_frame$Freq)
ylim=c(0,50)

ggplot(data_frame) + geom_bar(aes(x = levels(labels$behaviour_sub))) + ylim(0,10)

