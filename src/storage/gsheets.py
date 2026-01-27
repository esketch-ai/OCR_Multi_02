import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from src.config import Config

class GoogleSheetClient:
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets']
        self.creds = None
        self.service = None
        self.sheet_id = Config.GOOGLE_SHEET_ID
        
        self.authenticate()

    def authenticate(self):
        try:
            if os.path.exists(Config.GOOGLE_APPLICATION_CREDENTIALS):
                self.creds = service_account.Credentials.from_service_account_file(
                    Config.GOOGLE_APPLICATION_CREDENTIALS, scopes=self.scopes)
                self.service = build('sheets', 'v4', credentials=self.creds)
            else:
                print("Service account file not found. Google Sheets integration disabled.")
        except Exception as e:
            print(f"Authentication failed: {e}")

    def get_all_values(self):
        """Fetches all values from the first sheet."""
        if not self.service:
            return []
        try:
            sheet_name = self.get_first_sheet_name()
            # Quote sheet name to handle spaces/special chars safely
            range_name = f"'{sheet_name}'!A:Z" 
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.sheet_id, range=range_name).execute()
            return result.get('values', [])
        except Exception as e:
            print(f"Failed to fetch values: {e}")
            return []

    def get_first_sheet_name(self):
        """Fetch the title of the first sheet in the spreadsheet."""
        try:
            sheet_metadata = self.service.spreadsheets().get(spreadsheetId=self.sheet_id).execute()
            sheets = sheet_metadata.get('sheets', [])
            if sheets:
                return sheets[0]['properties']['title']
            return 'Sheet1'
        except Exception as e:
            print(f"Failed to fetch sheet name: {e}")
            return 'Sheet1'

    def clear_first_sheet(self):
        """Clears all content from the first sheet."""
        if not self.service:
            print("Google Sheet service not initialized.")
            return False
            
        try:
            sheet_name = self.get_first_sheet_name()
            range_name = f"'{sheet_name}'!A:Z"
            self.service.spreadsheets().values().clear(
                spreadsheetId=self.sheet_id, range=range_name, body={}
            ).execute()
            print(f"Cleared sheet: {sheet_name}")
            return True
        except Exception as e:
            print(f"Failed to clear sheet: {e}")
            return False

    def append_row(self, data):
        """
        Appends a list of values as a row to the configured Google Sheet.
        """
        if not self.service:
            print("Google Sheet service not initialized.")
            return False

        try:
            # Dynamically get the first sheet name
            sheet_name = self.get_first_sheet_name()
            range_name = f"'{sheet_name}'!A1"
            
            body = {
                'values': [data]
            }
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.sheet_id,
                range=range_name, # Appends to the first sheet dynamically
                valueInputOption='RAW',
                body=body
            ).execute()
            print(f"{result.get('updates').get('updatedCells')} cells appended.")
            return True
        except Exception as e:
            print(f"Failed to append row: {e}")
            return False
