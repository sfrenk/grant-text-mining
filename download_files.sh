# Download NIH grant data from https://exporter.nih.gov/ExPORTER_Catalog.aspx?sid=0&index=1

years=(2016 1996 1986)

if [[ -d sample ]]; then
	mkdir sample
fi

for year in ${years[@]}; do
	curl -o ${year}.csv.zip https://exporter.nih.gov/CSVs/final/RePORTER_PRJABS_C_FY${year}.zip
	unzip -c ${year}.csv.zip | iconv -c -t UTF-8 > ${year}.csv
	head -1000 ${year}.csv > sample/${year}_train.csv
	head -2000 ${year}.csv | tail -1000 > sample/${year}_test.csv

done
