
library(Rtsne)
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(ggplot2)
library(stringr)

# setwd('C:/Users/duongdb/Documents/WS22qOther_08102021/Classify/b4ns448Wl5ss10lr1e-05dp0.2b64ntest1M0.7T0.8+Soft+WS+22q11DS+Control+Normal+Whole+GanInEval+Hard')

setwd('/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr1e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole')

namelist = c('test_on_fold_5_from_fold0.csv') # '22q_Norm0.7_from_fold0.csv', 
df_all = NULL 
for (name in namelist){
  # df = '22q_Norm0.7_from_fold0.csv' # 'test_on_fold_5_from_fold0.csv'
  df = read.csv(name)

  if (name == '22q_Norm0.7_from_fold0.csv') {
    df = df[df['fold']==0, ] # ! our own filtering ? 
  } else { 
    df = df[df['fold']==5, ] # !
  }

  # names(df)
  # dfname = df[ names(df)[1:7] ]

  df['fake'] = 'real'
  df[ grep('seed', t((df['name']))), 'fake' ] = 'fake'

  if (is.null(df_all)){
    df_all = df
  } else {
    df_all = rbind(df_all,df)
  }
  rm(df)
}

z = names(df_all)
for (i in z){
  if (! i %in% names(df)){
    print (i)
  }
}

print (table(df_all['fake']))

dfsub = df_all[paste0('X',0:1791)]
tsne.norm = Rtsne(dfsub, pca = FALSE, check_duplicates = FALSE)

dfname = df_all[ c('fake',names(df_all)[1:7]) ]
# rm (df_all)
dfname %<>% mutate(tsne1 = tsne.norm$Y[, 1], tsne2 = tsne.norm$Y[, 2])


p = ggplot(subset(dfname,dfname$fake=='real'), aes(x = tsne1, y = tsne2, colour = label )) + 
    geom_point(alpha = 0.5) + 
    theme_bw() +
    theme(text = element_text(size=18), legend.position="left")+
    guides(color = guide_legend(override.aes = list(size = 5) ), shape = guide_legend(override.aes = list(size = 5) ) )

png("tsne-real.png",width = 9, height = 8, units = "in", res=300)
print(p)
dev.off()


p = ggplot(dfname, aes(x = tsne1, y = tsne2, colour = label, shape=fake )) + 
    geom_point(alpha = 0.5) + 
    theme_bw() +
    theme(text = element_text(size=18), legend.position="left")+
    guides(color = guide_legend(override.aes = list(size = 5) ), shape = guide_legend(override.aes = list(size = 5) ) )

png("tsne-train.png",width = 9, height = 8, units = "in", res=300)
print(p)
dev.off()


p = ggplot(dfname, aes(x = tsne1, y = tsne2, colour = fake )) + 
    geom_point(alpha = 0.5) + 
    theme_bw() +
    theme(text = element_text(size=18), legend.position="left")+
    guides(color = guide_legend(override.aes = list(size = 5) ), shape = guide_legend(override.aes = list(size = 5) ) )

png("tsne-train-fakehighlight.png",width = 9, height = 8, units = "in", res=300)
print(p)
dev.off()


p = ggplot(subset(dfname,dfname$label %in% c('22q11DS','WS','Controls')), aes(x = tsne1, y = tsne2, colour = fake, shape=label )) + 
    geom_point(alpha = 0.5) + 
    theme_bw() +
    theme(text = element_text(size=18), legend.position="left")+
    guides(color = guide_legend(override.aes = list(size = 5) ), shape = guide_legend(override.aes = list(size = 5) ) )

png("tsne-train-fakehighlight-subset.png",width = 9, height = 8, units = "in", res=300)
print(p)
dev.off()


p = ggplot(subset(dfname,dfname$label %in% c('22q11DS','Controls') & dfname$fold %in% c(0,5)), aes(x = tsne1, y = tsne2, colour = label, shape=fake )) + 
    geom_point(alpha = 0.5,size = 2) + 
    theme_bw() +
    theme(text = element_text(size=18), legend.position="left") +
    guides(color = guide_legend(override.aes = list(size = 5) ), shape = guide_legend(override.aes = list(size = 5) ) )

png("tsne-train-fakehighlight-subset-22qControl.png",width = 9, height = 8, units = "in", res=300)
print(p)
dev.off()

