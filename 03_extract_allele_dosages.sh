#!/bin/bash
#SBATCH --partition=caslake
#SBATCH --job-name=par_all_freqs
#SBATCH --account=pi-haky
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=10G
#SBATCH --output=allele-par-%j.out
#SBATCH --error=allele-par-%j.err

module load python
cd /beagle3/haky/users/sofia/enformer_retrain/scripts
source activate /scratch/midway3/rkb7263/anaconda/TFXcan-pipeline-tools

for chr in {1..22} X ; 
do
    echo "Processing chr$chr..."
bcftools view -S /beagle3/haky/users/sofia/enformer_retrain/eur_1000G.txt -R /beagle3/haky/users/sofia/enformer_retrain/expanded_validation_intervals.txt -v snps -Ou /project2/haky/Data/1000G/vcf_snps_only/ALL.chr${chr}.shapeit2_integrated_SNPs_v2a_27022019.GRCh38.phased.vcf.gz |
bcftools query -f '%CHROM\t%POS\t%ALT[\t%GT]\n' |
awk '{
    sum = 0;
    count = 0;
    for (i = 3; i <= NF; i++) {
        gsub(/[|\/]/, " ", $i);
        split($i, a, " ");
        for (j in a) {
            if (a[j] ~ /^[0-9]+$/) {
                sum += a[j];
                count++;
            }
        }
    }
    avg = (count > 0) ? sum / count : "NA";
    print $1, $2, $3, avg;
}' > /beagle3/haky/users/sofia/enformer_retrain/EUR_allele_freqs_val/chr${chr}_EUR_allele_freqs.txt &
done
wait