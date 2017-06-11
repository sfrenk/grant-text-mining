# Download NIH grant data from https://exporter.nih.gov/ExPORTER_Catalog.aspx?sid=0&index=1


# Set up directories

if [[ ! -d ./data/sample ]]; then
	mkdir -p ./data/sample
fi

# Download sample years

years=(2016 2006 1996 1986)

for year in ${years[@]}; do

	# Download data
	#curl -o ${year}.csv.zip https://exporter.nih.gov/CSVs/final/RePORTER_PRJABS_C_FY${year}.zip

	# Unzip data and remove any weird (non UTF8) symbols
	# Also remove the prefix that was added to all abstracts in later years
	unzip -c ${year}.csv.zip | iconv -c -t UTF-8 | grep -Eo '"[^"]+"' | sed 's/DESCRIPTION (provided by applicant)://g' | sed 's/unreadable//g' > ./data/${year}.csv

	# Random sample
	shuf -n 1000 ./data/${year}.csv > ./data/sample/${year}.csv

done

