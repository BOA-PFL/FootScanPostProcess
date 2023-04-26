rm(list=ls())
library(tidyverse)
library(readxl)
library(lme4)
library(patchwork)
library(effsize)

#-------------------------------------------------------------------------------
# The purpose of this code is to understand the differences between male and 
# female foot anthroprometrics
#-------------------------------------------------------------------------------
# Loading in the data frame and organizing left and right sides into date frames
subSizes <- read.csv('C:/Users/eric.honert/Boa Technology Inc/PFL Team - General/BigData/FootScan Data/data_management_BOA Technology_2023-03-28 10_33_40.csv')

subSizes$R_pMTP1 <- subSizes$Right.Length.To.First.Met.Head/subSizes$Right.Length*10
subSizes$L_pMTP1 <- subSizes$Left.Length.To.First.Met.Head/subSizes$Left.Length*10
subSizes$R_pMTP5 <- subSizes$Right.Length.To.Fifth.Met.Head/subSizes$Right.Length*10
subSizes$L_pMTP5 <- subSizes$Left.Length.To.Fifth.Met.Head/subSizes$Left.Length*10
subSizes$R_pDorsal <- subSizes$Right.Dorsal.Height/subSizes$Right.Length*10
subSizes$L_pDorsal <- subSizes$Left.Dorsal.Height/subSizes$Left.Length*10



dat <- data.frame(Sex = c(subSizes$Gender,subSizes$Gender),
                  footL = c(subSizes$Right.Length*10,subSizes$Left.Length*10),
                  ffWidth = c(subSizes$Right.Width*10,subSizes$Left.Width*10),
                  heelWidth = c(subSizes$Right.Heel.Width,subSizes$Left.Heel.Width),
                  dorsalHeight = c(subSizes$Right.Dorsal.Height,subSizes$Left.Dorsal.Height))

percSizes <- data.frame(Sex = c(subSizes$Gender,subSizes$Gender),
                      footL = c(subSizes$Right.Length*10,subSizes$Left.Length*10),
                      MTP1 = c(subSizes$Right.Length.To.First.Met.Head,subSizes$Left.Length.To.First.Met.Head),
                      MTP5 = c(subSizes$Right.Length.To.Fifth.Met.Head,subSizes$Left.Length.To.Fifth.Met.Head),
                      pMTP1 = c(subSizes$R_pMTP1,subSizes$L_pMTP1), 
                      pMTP5 = c(subSizes$R_pMTP5,subSizes$L_pMTP5),
                      dDorsalHeight = c(subSizes$R_pDorsal,subSizes$L_pDorsal))

percSizes <- percSizes %>%
  filter(pMTP1 != 'NA')



ggplot(percSizes, aes(x=pMTP1,color=Sex)) + geom_histogram() + xlab('Location of MTP1 (% Foot Length)')

ggplot(percSizes, aes(x=footL,y=MTP1,color=Sex)) + geom_point(size=2) + 
  xlab('Foot Length (mm)') + ylab('Location of MTP1 (mm)') + theme(text = element_text(size = 30))

ggplot(percSizes, aes(x=footL,y=pMTP1,color=Sex)) + geom_point(size=2) + 
  xlab('Foot Length (mm)') + ylab('Location of MTP1 (% Foot Length)') + theme(text = element_text(size = 30))

ggplot(percSizes, aes(x=pMTP5,color=Sex)) + geom_histogram() + xlab('Location of MTP5 (% Foot Length)')


ggplot(percSizes, aes(x=footL,y=MTP5,color=Sex)) + geom_point(size=2) + 
  xlab('Foot Length (mm)') + ylab('Location of MTP5 (mm)') + theme(text = element_text(size = 30))
ggplot(percSizes, aes(x=footL,y=pMTP5,color=Sex)) + geom_point(size=2) + xlab('Foot Length (mm)') + ylab('Location of MTP5 (% Foot Length)')

ggplot(percSizes, aes(x=footL,y=dDorsalHeight,color=Sex)) + geom_point(size=2) + 
  xlab('Foot Length (mm)') + ylab('Instep Height (% Foot Length)') + theme(text = element_text(size = 30))


ggplot(dat, aes(x=footL,y=ffWidth,color=Sex)) + geom_point(size=2) + xlab('Foot Length (mm)') + ylab('Forefoot Width (mm)')

ggplot(dat, aes(x=footL,y=dorsalHeight,color=Sex)) + geom_point(size=2) + 
  xlab('Foot Length (mm)') + ylab('Instep Height (mm)') + theme(text = element_text(size = 30))

ggplot(dat, aes(x=footL,y=heelWidth,color=Sex)) + geom_point(size=2) + xlab('Foot Length (mm)') + ylab('Hee Width (mm)')

