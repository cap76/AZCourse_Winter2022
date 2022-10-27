

SOX17 <- readDNAStringSet("SOX17.fa")
SOXseq <- paste(SOX17)
allS17  <- array(0, dim=c(length(SOXseq),1000,4))
 
for (i in 1:length(SOXseq)){
         allS17[i,1:1000,1:4] <- t(diag(4)[match(unlist(lapply(SOXseq[i], utf8ToInt)), utf8ToInt("ACGT")), ])
}


PRDM1 <- readDNAStringSet("PRDM1.fa")
PRDMseq <- paste(PRDM1)
allP1  <- array(0, dim=c(length(PRDMseq),1000,4))
> for (i in 1:length(PRDMseq)){
         allP1[i,1:1000,1:4] <- t(diag(4)[match(unlist(lapply(PRDMseq[i], utf8ToInt)), utf8ToInt("ACGT")), ])
}

RANDOM1 <- readDNAStringSet("random.fa")
Rseq <- paste(RANDOM1)
allR1  <- array(0, dim=c(length(Rseq),1000,4))
> for (i in 1:length(Rseq)){
         allR1[i,1:1000,1:4] <- t(diag(4)[match(unlist(lapply(Rseq[i], utf8ToInt)), utf8ToInt("ACGT")), ])
}
