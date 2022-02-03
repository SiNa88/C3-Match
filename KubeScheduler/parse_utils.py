import json
from typing import List
from matplotlib.font_manager import json_dump
import pandas as pd

EQUAL_DELIMINATOR: str = '='
MINUS_DELIMINATOR: str = '-'


class ParseTextOutput():

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._dataframe: pd.DataFrame = self.__load_file()

    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def get_json(self) -> str:
        parsed_df = self._dataframe.iloc[0].to_json(orient='columns')
        json_representation = json.loads(parsed_df)
        json_representation = json.dumps(json_representation, indent=4)
        return json_representation

    def __load_file(self):
        with open(self.file_path) as file:
            file_contents: List[str] = self.__load_file_contents(file)
            file_contents = self.__parse_file_contents(file_contents)
            return self.__convert_to_dataframe(file_contents)

    def __load_file_contents(self, file) -> List[str]:
        file_contents = []
        finished = False

        while not finished:
            next_line = file.readline()
            if EQUAL_DELIMINATOR in next_line or MINUS_DELIMINATOR in next_line:
                # Nothing to do here
                continue
            if len(next_line) == 0:
                # EOF
                finished = True
            else:
                file_contents.append(next_line)

        return file_contents

    def __parse_file_contents(self, file_contents: List[str]) -> List[str]:
        line_content = []
        # iterate through all lines except the last one
        for content in file_contents[:-1]:
            content_split = content.split(' ')
            line_content.append([self.__clean_string(elem)
                                for elem in content_split if len(elem) > 0])

        return line_content

    def __clean_string(self, content: str) -> str:
        return content.replace('\n', '').replace('\t', '')

    def __convert_to_dataframe(self, file_contents: List[str]) -> pd.DataFrame:
        dataframe_columns = file_contents[0]
        dataframe_content = file_contents[1:]
        return pd.DataFrame(dataframe_content, columns=dataframe_columns)

