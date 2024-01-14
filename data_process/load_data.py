from Bio import Entrez, Medline
from datetime import datetime

Entrez.email = "leepual168@gmail.com"

# Search for documents with the term "intelligence" in the abstract and published between 2013 and 2023
search_term = "intelligence"
search_results = Entrez.read(
    Entrez.esearch(db="pubmed", term=search_term, retmax=1000000, datetype="pdat")
)

pmid_counter = 0

# Fetch and print information for each document
for pmid in search_results['IdList']:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    record = Medline.read(handle)
    # print(record)

    # Check if the abstract contains the term "intelligence"
    if 'AB' in record.keys() and search_term.lower() in record['AB'].lower():
        # Check if all the keys are present in the record
        required_keys = ['PMID', 'TI', 'AU', 'JT', 'DP', 'AB']
        if all(key in record for key in required_keys):
            # Check if the publication date is within the specified range
            if 'DP' in record.keys():
                # print(record['DP'])

                try:
                    # Try parsing with format '%Y %b %d'
                    publication_date = datetime.strptime(record['DP'], "%Y %b %d").date()
                except ValueError:
                    try:
                        # If the first format fails, try parsing with format '%Y %b'
                        publication_date = datetime.strptime(record['DP'], "%Y %b").date()
                    except ValueError:
                        try:
                            # If the second format fails, try parsing with format '%Y'
                            publication_date = datetime.strptime(record['DP'], "%Y").date()
                        except ValueError:
                            print(f"Could not parse the date: {record['DP']}")
                            # If date format is not one of the three formats, then we skip this date
                            publication_date = None

                if publication_date != None and 2013 <= publication_date.year <= 2023:
                    # Save relevant information for the document
                    pmid_counter += 1
                    print(f"Processing document {pmid}\n")

                    # Create a new output text file for every 20 documents
                    if pmid_counter % 20 == 1:
                        file_name = (f"../assets/data/pubmed_medical_intelligence/{pmid_counter // 20}_pubmed_medical_intelligence.txt")
                        with open(file_name, 'w', encoding='utf-8') as output_file:
                            output_file.write(f"PMID: {pmid}\n")
                            output_file.write(f"Title: {record['TI']}\n")
                            output_file.write(f"Author: {record['AU']}\n")
                            output_file.write(f"Journal Title: {record['JT']}\n")
                            output_file.write(f"Publication Date: {record['DP']}\n")
                            output_file.write(f"Abstract: {record['AB']}\n")
                            output_file.write("---------------------------------\n")
                    else:
                        # Append to the current text file
                        with open(file_name, 'a', encoding='utf-8') as output_file:
                            output_file.write(f"PMID: {pmid}\n")
                            output_file.write(f"Title: {record['TI']}\n")
                            output_file.write(f"Author: {record['AU']}\n")
                            output_file.write(f"Journal Title: {record['JT']}\n")
                            output_file.write(f"Publication Date: {record['DP']}\n")
                            output_file.write(f"Abstract: {record['AB']}\n")
                            output_file.write("---------------------------------\n")

    # Close the handle
    handle.close()