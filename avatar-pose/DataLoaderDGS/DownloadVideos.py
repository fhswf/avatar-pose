"""
DownloadVideos.py

Authors: Carolin Gottschalk, Jonas D. Stephan
License: Apache License 2.0

Description:
This script scrapes SignDict's GraphQL API to retrieve sign language videos.
It organizes videos alphabetically, extracts metadata, converts JSON data to CSV,
and downloads videos for offline use.
"""
import requests
import os
import json
import pandas as pd
from bs4 import BeautifulSoup
import string


class SignDictVideoCrawler:
    """
    A web scraper that collects and organizes sign language videos from SignDict.
    """
    def __init__(self):
        """
        Initializes the SignDictVideoCrawler with API details and processing steps.
        """
        self.url = "https://signdict.org/graphql-api"  # URL von GraphQL API
        self.step_create_json_video_info_list = True
        self.step_convert_json_to_csv = True
        self.step_enrich_csv_data = True
        self.step_download_videos = True

    def run(self):
        """
        Executes the full pipeline: scrape words, extract video metadata, save to CSV, and download videos.
        """
        # Create folders for A-Z
        for letter in string.ascii_uppercase:
            os.makedirs(name=f"videos/{letter}", exist_ok=True)

        # Create folders for special German characters
        additional_letter_dgs=["Ä","Ö","Ü"]
        for letter in additional_letter_dgs:
            os.makedirs(name=f"videos/{letter}", exist_ok=True)

        if self.step_create_json_video_info_list:
            # load and clean word list
            cleaned_word_list = self.load_cleaned_word_list(file_name="word_list.txt")

            # go through all words with graphql queries and save results as json
            self.create_json_video_info_list(word_list=cleaned_word_list)

        if self.step_convert_json_to_csv:
            # convert json to csv
            self.convert_json_to_csv(file_name="signdict_json_list.json")

        if self.step_enrich_csv_data:
            # enrich csv data with description and combine text with description
            self.enrich_csv_data(file_name="signdict_table_data.csv")

        if self.step_download_videos:
            # download videos
            self.download_videos(file_name="signdict_table_data_with_descr.csv")

    @staticmethod
    def load_cleaned_word_list(file_name):
        """
        Loads a word list from a file, removing unnecessary spaces and newlines.

        :param file_name: The filename containing the words.
        :type file_name: str
        :return: A list of cleaned words.
        :rtype: list
        """
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        return lines

    def create_json_video_info_list(self, word_list):
        """
        Queries the GraphQL API for each word to retrieve video metadata.

        :param word_list: List of words to search for videos.
        :type word_list: list
        """
        # Initialize an empty list to store video entries
        current_video_info_list = []

        counter = 0
        for word in word_list:
            if counter % 100 == 0:
                print(f"Word {counter} to JSON.")
            graphql_query = self.build_graphql_query(word)

            # Send request to API
            response = requests.post(self.url, json={"query": graphql_query})

            # Check if request was successful
            if response.status_code == 200:
                data = response.json()

                # Extract video data from response
                for entry in data['data']['search']:
                    current_video_info = entry.get('currentVideo', {})

                    if current_video_info:
                        current_video_info_list.append({
                            'id': entry['id'],
                            'text': entry['text'],
                            'type': entry['type'],
                            'currentVideo': current_video_info
                        })
                counter = counter + 1
            else:
                print("Error:", response.status_code, response.text)

        # Save extracted video information to JSON file
        with open('signdict_json_list.json', 'w', encoding='utf-8') as f:
            json.dump(current_video_info_list, f, ensure_ascii=False, indent=4)

    @staticmethod
    def build_graphql_query(word):
        """
        Constructs a GraphQL query for a given word.

        :param word: The word to query.
        :type word: str
        :return: A GraphQL query string.
        :rtype: str
        """
        query = f"""
        {{
          search(word: "{word}") {{
            id
            text
            type
            currentVideo {{
              videoUrl
              license
              copyright
              originalHref
              user {{
                name
              }}
            }}
          }}
        }}
        """
        return query

    @staticmethod
    def convert_json_to_csv(file_name):
        """
        Converts video metadata stored in a JSON file to a structured CSV file.

        :param file_name: The name of the JSON file containing video metadata.
        :type file_name: str
        """

        with open(file_name, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        df_data = []
        for item in loaded_data:
            current_video = item['currentVideo']
            df_data.append({
                'id': item['id'],
                'text': item['text'],
                'type': item['type'],
                'copyright': current_video['copyright'],
                'license': current_video['license'],
                'originalHref': current_video['originalHref'],
                'user': current_video['user']['name'],
                'videoUrl': current_video['videoUrl']
            })

        df = pd.DataFrame(df_data)
        df = df.drop_duplicates(subset='id', keep='first')
        df.to_csv("signdict_table_data.csv", index=False)

    @staticmethod
    def enrich_csv_data(file_name):
        """
        Enriches the CSV data by scraping additional descriptions from SignDict.

        :param file_name: The name of the CSV file to enrich.
        :type file_name: str
        """
        df = pd.read_csv(filepath_or_buffer=file_name)

        # Insert columns for description and merged text-description field
        df.insert(loc=2, column="description", value=None)
        df.insert(loc=3, column="word_with_descr", value=None)

        for index, row in df.iterrows():
            if index % 100 == 0:
                print(f"Index Dataframe: {index}")
            url = f"https://signdict.org/entry/{row['id']}"

            # Send an HTTP request to the webpage
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.content, "html.parser")

                # Navigate to the element with class "so-video-details--headline"
                headline_tag = soup.find('h1', class_='so-video-details--headline')

                # Find the next <p> tag and extract the text
                next_p_tag = headline_tag.find_next('p')
                description = next_p_tag.get_text()

                # Ignore descriptions containing irrelevant phrases
                if "Interessante Gebärde" in description:
                    description = None

                df.at[index, "description"] = description
                if description:
                    df.at[index, "word_with_descr"] = f"{row['text']} ({description})"

        df.to_csv(path_or_buf="signdict_table_data_with_descr.csv", index=False)

    @staticmethod
    def download_videos(file_name):
        """
        Downloads sign language videos from SignDict and saves them in categorized folders.

        :param file_name: The name of the CSV file containing video URLs.
        :type file_name: str
        """
        df = pd.read_csv(filepath_or_buffer=file_name)

        for index, row in df.iterrows():
            tmp_video_link = row["videoUrl"]
            tmp_video_title = f"{row['id']}"
            tmp_first_letter = row["text"][0].upper()
            tmp_file_path = f"videos/{tmp_first_letter}/{tmp_video_title}.mp4"

            print(f"Downloading video from {tmp_video_link}...")
            response = requests.get(tmp_video_link, stream=True)

            if response.status_code == 200:
                with open(tmp_file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Video successfully downloaded: {tmp_file_path}")
            else:
                print(f"Error downloading video: {response.status_code}")

if __name__ == '__main__':
    sign_dict_video_crawler = SignDictVideoCrawler()
    sign_dict_video_crawler.run()

