#!/usr/bin/bash
#SBATCH -t 0-5

# Download NIH grant data from https://exporter.nih.gov/ExPORTER_Catalog.aspx?sid=0&index=1

download=false

usage="download and/or process abstract data from the NIH website.
		
		bash [-d] download_files.sh

		-d/--download: download files (by default, the script processes the already downloaded files) "


while [[ $# > 0 ]]; do

	key="$1"

	case $key in
		-h|--help)
		echo $usage
		exit 0
		;;
		-d|--download)
		download=true
		;;
	esac
	shift
done

# Set up directories

if [[ ! -d ./data/sample ]]; then
	mkdir -p ./data/sample
fi

if [[ ! -d ./data/R ]]; then
	mkdir -p ./data/R
fi

# Download sample years

years=(2016 2006 1996 1986)

for year in ${years[@]}; do

	if [[ $download = true ]]; then

		# Download data
		curl -o data/${year}.csv.zip https://exporter.nih.gov/CSVs/final/RePORTER_PRJABS_C_FY${year}.zip
	fi

	# Unzip data and remove any weird (non UTF8) symbols
	# Remove additional text added at the beginning of abstracts in certain years 
	# Also filter for abstracts that have at least 10 contiguous words
	# Also remove the prefix that was added to all abstracts in later years
	# Shuffle the entries
	unzip -c data/${year}.csv.zip | iconv -c -t UTF-8 | sed 's/DESCRIPTION (provided by applicant)//g' | sed 's/unreadable//g' | sed -r 's/^[^"]+//g' | sed -r 's/\"[^A-Za-z]+//g' | sed -r 's/(\")[A-Z ]{2,}/\1/g' | grep -P '([A-Za-z]+[^A-Za-z]+){10,}' | shuf > ./data/${year}.csv

	
	# Split into train and test
	head -10000 ./data/${year}.csv > ./data/sample/${year}_train.csv
	tail -1000 ./data/${year}.csv > ./data/sample/${year}_test.csv

	# Take sample for R analysis
	head -1000 ./data/${year}.csv > ./data/R/${year}.csv

	rm ./data/${year}.csv

done

