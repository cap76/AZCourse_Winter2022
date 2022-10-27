cat GSM4836893_PRDM1_hPGCLC_r1_peaks.narrowPeak GSM4836894_PRDM1_hPGCLC_r2_peaks.narrowPeak > PRDM1_merged.narrowPeak
cat GSM4836897_SOX17_hPGCLC_r1_peaks.narrowPeak GSM4836898_SOX17_hPGCLC_r3_peaks.narrowPeak > SOX17_merged.narrowPeak

bedtools merge -i PRDM1_merged.s.narrowPeak > PRDM1_merged.m.narrowPeak
bedtools merge -i SOX17_merged.s.narrowPeak > SOX17_merged.m.narrowPeak

awk 'BEGIN {OFS="\t"} { print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2) }' PRDM1_merged.m.narrowPeak > check1.tmp
awk 'BEGIN {OFS="\t"} { print $1 "\t" int(($2+$3)/2) "\t" int(($2+$3)/2) }' SOX17_merged.m.narrowPeak > check2.tmp

bedtools slop -i check1.tmp -b 500 -g /mnt/scratch/gurdon/cap76/Wolfram/RNASeq/chr.size > PRDM1.bed
bedtools slop -i check2.tmp -b 500 -g /mnt/scratch/gurdon/cap76/Wolfram/RNASeq/chr.size > SOX17.bed

bedtools getfasta -fi /mnt/scratch/gurdon/cap76/Wolfram/RNASeq/GRCh38.primary_assembly.genome.fa -bed PRDM1.bed > PRDM1.fa
bedtools getfasta -fi /mnt/scratch/gurdon/cap76/Wolfram/RNASeq/GRCh38.primary_assembly.genome.fa -bed SOX17.bed > SOX17.fa

bedtools shuffle -i PRDM1.bed -g /mnt/scratch/gurdon/cap76/Wolfram/RNASeq/chr.size > random.tmp
bedtools subtract -a random.tmp -b check1.tmp -A > random.tmp2
bedtools subtract -a random.tmp -b check2.tmp -A > random.bed

bedtools getfasta -fi /mnt/scratch/gurdon/cap76/Wolfram/RNASeq/GRCh38.primary_assembly.genome.fa -bed random.bed > random.fa


