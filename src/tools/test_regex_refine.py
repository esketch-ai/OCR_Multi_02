
# -*- coding: utf-8 -*-
from src.parser.car_registration import CarRegistrationParser

def test_regex():
    parser = CarRegistrationParser()
    
    # "① 자동차등록번호    ② 차종    승합"
    # \u2460 \uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638    \u2461 \ucc28\uc885    \uc2b9\ud569
    line1 = "\u2460 \uc790\ub3d9\ucc28\ub4f1\ub85d\ubc88\ud638    \u2461 \ucc28\uc885    \uc2b9\ud569"
    line2 = "70\uc7901234"
    
    # "③ 차명    BS106 /    ⑤ 형식 및 모델연도    KL5UM... / 2020"
    # \u2462 \ucc28\uba85    BS106 /    \u2464 \ud615\uc2dd \ubc0f \ubaa8\ub378\uc5f0\ub3c4    KL5UM52EDBU006543- / 2020
    line3 = "\u2462 \ucc28\uba85    BS106 /    \u2464 \ud615\uc2dd \ubc0f \ubaa8\ub378\uc5f0\ub3c4    KL5UM52EDBU006543- / 2020"
    
    bad_text = f"{line1}\n{line2}\n{line3}"
    
    print("Testing Clean Text:")
    res = parser.parse(bad_text)
    print(f"Veh No: {res.get('vehicle_no')}") 
    print(f"Type: {res.get('vehicle_type')}")
    print(f"Model: {res.get('model_name')}")    
    print(f"Format: {res.get('vehicle_format')}") 

    # Consolidated logic test (merged lines)
    merged_text = f"{line1} {line2} {line3}"
    print("\nTesting Merged Text:")
    res2 = parser.parse(merged_text)
    print(f"Veh No: {res2.get('vehicle_no')}") 
    print(f"Type: {res2.get('vehicle_type')}")
    print(f"Model: {res2.get('model_name')}")
    
if __name__ == "__main__":
    test_regex()
