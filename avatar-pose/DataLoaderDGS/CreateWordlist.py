"""
CreateWordlist.py

Authors: Carolin Gottschalk, Joans D. Stephan
License: Apache License 2.0

Description:
This script extracts a list of words from the SignDict website.
It iterates through letters A-Z, scrapes words from multiple pages,
and saves them to a text file after removing duplicates and sorting them alphabetically.
"""
import requests
from bs4 import BeautifulSoup
import string


class SignDictWordCrawler:
    """
    A web scraper for collecting words from the SignDict dictionary.
    """
    def __init__(self):
        pass

    def get_words(self):
        """
        Scrapes words from SignDict by iterating through all letters from A-Z.
        Removes duplicates and saves the sorted list to a file.
        """
        word_list = list()

        # Iterate through all letters from A to Z
        for letter in string.ascii_uppercase:
            print(f"Aktueller Buchstabe: {letter}")

            for page in range(1, 50):
                tmp_word_list = self.search_letter(letter, page)
                if page == 1:
                    last_word_list = tmp_word_list
                else:
                    if tmp_word_list == last_word_list:
                        break
                    else:
                        last_word_list = tmp_word_list
                word_list.extend(tmp_word_list)

        # Remove duplicates
        word_list = list(set(word_list))

        # Sort words alphabetically, case-insensitive
        word_list.sort(key=str.lower)

        # Print the words
        for word in word_list:
          print(word)

        # Save the words to a file
        with open('word_list.txt', 'w', encoding='utf-8') as file:
            for word in word_list:
                file.write(word + '\n')

    @staticmethod
    def search_letter(letter, page):
        """
        Fetches words from a specific letter and page on SignDict.

        :param letter: The letter to search words for.
        :type letter: str
        :param page: The page number to scrape.
        :type page: int
        :return: A list of words extracted from the page.
        :rtype: list
        """
        # URL of the SignDict page
        url = f"https://signdict.org/entry?letter={letter}&page={page}"

        # Send an HTTP request to the webpage
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract words (adapted to the website's HTML structure)
            words = []
            for word_tag in soup.find_all("a", class_="so-search-result--link"):
                words.append(word_tag.get_text(strip=True))

            # Return the extracted words
            return words
        else:
            print(f"Error: Unable to load page. Status code {response.status_code}")

if __name__ == '__main__':
    sign_dict_word_crawler = SignDictWordCrawler()
    sign_dict_word_crawler.get_words()

