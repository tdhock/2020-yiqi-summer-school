url.vec <- c(
  spam="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data", 
  zip.gz="https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz")
for(f in names(url.vec)){
  u <- url.vec[[f]]
  if(!file.exists(f)){
    download.file(u, f)
  }
}

