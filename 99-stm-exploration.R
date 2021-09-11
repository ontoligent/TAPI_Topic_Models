
library(stm)
library(igraph)
library(stmCorrViz)

# INGEST

data <- read.csv("./experimental/data/poliblogs2008.csv") 
# Pre-processed texts by the package authors named 'shortdoc' 
# that was used for their vignette example
load("./experimental/data/VignetteObjects.RData") 

# PREPARE

processed <- textProcessor(data$documents, metadata=data)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
token = out$documents
vocab = out$vocab
meta = out$meta
plotRemoved(processed$documents, lower.thresh=seq(1,200, by=100))

# ESTIMATE

poliblogPrevFit <- stm(out$documents, 
                       out$vocab, 
                       K=20, 
                       prevalence=~rating+s(day), 
                       max.em.its=75,
                       data=out$meta, 
                       init.type="Spectral", 
                       seed=8458159)
plot(poliblogPrevFit, type="summary", xlim=c(0,.4))
plot(poliblogPrevFit, type="labels", topics=c(3,7,20))
plot(poliblogPrevFit, type="hist")
plot(poliblogPrevFit, type="perspectives", topics=c(7,10))

# EVALUATE

poliblogSelect <- selectModel(out$documents, out$vocab, K=20, prevalence=~rating+s(day),
                              max.em.its=75, data=meta, runs=20, seed=8458159)
plotModels(poliblogSelect)
topicQuality(model=poliblogPrevFit, documents=docs)
selectedModel3 <- poliblogSelect$runout[[3]] # Choose model #3
storage <- manyTopics(out$documents, out$vocab, K=c(7:10), prevalence=~rating+s(day),
                      data=meta, runs=10)
storageOutput1 <- storage$out[[1]] # For example, choosing the model with 7 topics
plot(storageOutput1)

kResult <- searchK(out$documents, out$vocab, K=c(7,10), prevalence=~rating+s(day),
                   data=meta)
plot(kResult)

# UNDERSTAND

labelTopicsSel <- labelTopics(poliblogPrevFit, c(3,7,20))
print(sageLabels(poliblogPrevFit))
thoughts3 <- findThoughts(poliblogPrevFit, texts=shortdoc, n=3, topics=3)$docs[[1]]
plotQuote(thoughts3, width=40, main="Topic 3")
out$meta$rating <- as.factor(out$meta$rating)
prep <- estimateEffect(1:20 ~ rating+s(day), poliblogPrevFit, meta=out$meta, 
                       uncertainty="Global")
plot(prep, covariate="rating", topics=c(3, 7, 20), model=poliblogPrevFit, 
     method="difference", cov.value1="Liberal", cov.value2="Conservative",
     xlab="More Conservative ... More Liberal", main="Effect of Liberal vs. Conservative",
     xlim=c(-.15,.15), labeltype ="custom", custom.labels=c('Obama', 'Sarah Palin', 
                                                            'Bush Presidency'))
plot(prep, "day", method="continuous", topics=20, model=z, printlegend=FALSE, xaxt="n", 
     xlab="Time (2008)")
monthseq <- seq(from=as.Date("2008-01-01"), to=as.Date("2008-12-01"), by="month")
monthnames <- months(monthseq)
axis(1, at=as.numeric(monthseq)-min(as.numeric(monthseq)), labels=monthnames)
mod.out.corr <- topicCorr(poliblogPrevFit)
plot(mod.out.corr)

# VISUALIZE

poliblogContent <- stm(out$documents, out$vocab, K=20, prevalence=~rating+s(day), 
                       content=~rating, max.em.its=75, data=out$meta, 
                       init.type="Spectral", seed=8458159)

plot(poliblogContent, type="perspectives", topics=7)

cloud(poliblogContent, topic=7)

poliblogInteraction <- stm(out$documents, out$vocab, K=20, prevalence=~rating*day, 
                           max.em.its=75, data=out$meta, seed=8458159)

prep2 <- estimateEffect(c(20) ~ rating*day, poliblogInteraction, metadata=out$meta, 
                        uncertainty="None")
plot(prep2, covariate="day", model=poliblogInteraction, method="continuous", xlab="Days",
     moderator="rating", moderator.value="Liberal", linecol="blue", ylim=c(0,0.12), 
     printlegend=F)
plot(prep2, covariate="day", model=poliblogInteraction, method="continuous", xlab="Days",
     moderator="rating", moderator.value="Conservative", linecol="red", add=T,
     printlegend=F)
legend(0,0.12, c("Liberal", "Conservative"), lwd=2, col=c("blue", "red"))
plot(poliblogPrevFit$convergence$bound, type="l", ylab="Approximate Objective", 
     main="Convergence")
stmCorrViz(poliblogPrevFit, "stm-interactive-correlation.html", 
           documents_raw=data$documents, documents_matrix=out$documents)



###############################################################

### WINE REVIEWS
corpus = read.csv('./corpora//winereviews-tapi.csv', sep = '|')
edition <- textProcessor(corpus$doc_content, metadata=corpus)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
