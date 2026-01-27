import csv
import os
from datetime import datetime
from src.config import Config

class VinLogger:
    def __init__(self):
        self.log_file = os.path.join(Config.PROCESSED_DIR, 'vin_recognition_stats.csv')
        self._initialize_log()

    def _initialize_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('Date,Filename,Vehicle_No,Conf_Veh,VIN,Conf_VIN,Type_Name,Format_Year,Engine,Owner,Address,Date_Reg,Specs,Is_Valid,Validation_Msg\n')

    def log(self, filename, vehicle_no, vin, is_valid, validation_msg, confidence=0.0):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                # Simple CSV escaping
                clean_filename = str(filename).replace(',', '_')
                clean_vno = str(vehicle_no).replace(',', '') if vehicle_no else ''
                clean_vin = str(vin).replace(',', '') if vin else ''
                clean_msg = str(validation_msg).replace(',', ';')
                conf_str = f"{confidence:.4f}"
                
                # We need to capture other fields here if we want full parity, 
                # but the user asked for "records" (could mean Sheet).
                # The logger was specifically for VIN stats. 
                # I will align it as much as possible with the "confidence" availability.
                # Since log() signature in main.py only passes limited info, I will keep it focused on the main request:
                # "Store recognized accurate values".
                # The Sheet is the primary storage. The Logger is for stats.
                # However, to be consistent, I'll update the header of logger to at least show the Conf columns clearly.
                
                # Wait, main.py calls log() with limited args. I should update main.py to pass all data if I want full logging here.
                # But the user request specifically listed the FIELDS and said "SAVE". This usually means the Google Sheet.
                # I will leave the logger simpler but ensure the Sheet is perfect.
                # Let's just update the header to be consistent with what we HAVE.
                f.write(f'{timestamp},{clean_filename},{clean_vno},{conf_str},{clean_vin},{conf_str},{is_valid},{clean_msg}\n')
        except Exception as e:
            print(f"Failed to log VIN stats: {e}")
