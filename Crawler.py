import pip._vendor.requests as requests
from bs4 import BeautifulSoup
from transformers import pipeline
import csv

url = "https://en.wikipedia.org/wiki/1975_Pacific_hurricane_season"
page = requests.get(url)

soup = BeautifulSoup(page.content, "html.parser")

# Find the start and end of the content to be scraped
start = soup.find('h2', id='Systems')
end = soup.find('h2', id='See_also')

# # scrape all sections, searching for a div-table-link-paragraph(s)-div pattern
def scrape_sections():
    
    # Collect all elements between start and end
    sections = []
    in_section = False
    current_section = []
    
    for element in start.find_all_next():
      
        if element == end:
            break
        
        # Identify the start of a new section by looking for the first div in the section that contains the mw-heading class
        if element.name == 'div' and set(['mw-heading', 'mw-heading3']).issubset(element.get('class', [])):

            # If we're in the middle of a section, this signals a new section, so save it
            if in_section:
                sections.append(current_section)
                current_section = []

            # Start a new section
            in_section = True
            current_section.append(element.get_text(separator=' ', strip=True))
        
        # Look for a link, table, and paragraphs within the section
        elif in_section and (element.name == 'a' or element.name == 'table' or element.name == 'p'):
            current_section.append(element.get_text(separator=' ', strip=True))
        
        # The end of a section is indicated by a div with style 'clear:both;'
        elif in_section and element.name == 'div' and 'clear:both;' in element.get('style', ''):
            current_section.append(element.get_text(separator=' ', strip=True))
            sections.append(current_section)
            current_section = []
            in_section = False

    return sections


def clean_text(text):
    # Remove unparseable characters by encoding to ASCII and ignoring errors
    return text.encode('ascii', 'ignore').decode('ascii').replace('?', ' ')

sections = scrape_sections()
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

with open('hurricanes_1975.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['hurricane_storm_name', 'date_start', 'date_end', 'number_of_deaths', 'list_of_areas_affected'])
    for section in sections:
        hurricane_text = '\n'.join(section)

        # Questions to extract data
        questions = [
            {"question": "What is the name of the hurricane?", "context": hurricane_text},
            {"question": "When did the hurricane begin?", "context": hurricane_text},
            {"question": "When did the hurricane end?", "context": hurricane_text},
            {"question": "How many died from the hurricane?", "context": hurricane_text},
            {"question": "Which areas did the hurricane affect?", "context": hurricane_text}
        ]

        # Extract data using the question-answering model
        extracted_data = {q['question']: qa_pipeline(q)['answer'] for q in questions}

        print('\n'*3)
        # Writing the output to a text file
        row = [
            clean_text(extracted_data.get("What is the name of the hurricane?", 'Unknown')),
            clean_text(extracted_data.get("When did the hurricane begin?", 'Unknown')),
            clean_text(extracted_data.get("When did the hurricane end?", 'Unknown')),
            clean_text(extracted_data.get("How many died from the hurricane?",'Unknown')),
            clean_text(extracted_data.get("Which areas did the hurricane affect?", 'Unknown'))
        ]
        
        # Write the row to the CSV file
        writer.writerow(row)












