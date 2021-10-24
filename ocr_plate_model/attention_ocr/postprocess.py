# -*- coding: utf-8 -*-
import re
# Military plate license
# AA     | 12-34
# prefix | suffix
military_prefix = ['AA', 'AB', 'AC', 'AD', 'AV', 'AT', 'AN', 'AP',
                    'BBB', 'BC', 'BH', 'BK', 'BL', 'BT', 'BP', 'BS', 'BV',
                    'HA', 'HB', 'HC', 'HD', 'HE', 'HT', 'HQ', 'HN', 'HH',
                    'KA', 'KB', 'KC', 'KD', 'KV', 'KP', 'KK', 'KT', 'KN',
                    'PA', 'PP', 'PM', 'PK', 'PT', 'PY', 'PQ', 'PX', 'PC', 'HL',
                    'QA', 'QB', 'QH',
                    'TC', 'TH', 'TK', 'TT', 'TM', 'TN', 'DB', 'ND', 'CH', 'VB', 'VK', 'CV', 'CA', 'CP', 'CM', 'CC',
                    'VT', 'CB']

# Civil plate license
# 26     | LD       | 88678
# prefix | postfix  | suffix
civil_prefix = range(10, 100) # 10, 11, ..., 99
special_postfix = ['LD', 'DA', 'R', 'T', 'MK', 'CD']

# Biển số xe 50cc
# XX-xx / XXX.XX
# seri biển số sử dụng lần lượt một trong 20 chữ cái sau đây: A, B, C, D, E, F, G, H, K, L, M, N, P, S, T, U, V, X, Y, Z kết hợp với một trong các chữ cái sau: A, B, C, D, E, F, H, K, L, M, N, P, R, S, T, U, V, X, Y, Z
regex_50cc_moto = '^[1-9][0-9][ABCDEFGHKLMNPSTUVXYZ][ABCDEFHKLMNPRSTUVXYZ][0-9]{4,5}$'

# Biển số xe 100cc - 150cc
# XX-Xn / XXX.XX
# seri biển số sử dụng một trong 19 chữ cái B, C, D, E, F, G, H, K, L, M, N, P, S, T, U, V, X, Y, Z kết hợp với một chữ số tự nhiên từ 1 đến 9
regex_100cc_moto = '^[1-9][0-9][BCDEFGHKLMNPSTUVXYZ][1-9][0-9]{4,5}$'

# Biển số xe Phân khối lớn
# XX-Xn / XXX.XX
# seri sử dụng chữ cái A kết hợp với 1 chữ số tự nhiên từ 1 đến 9
regex_powerful_moto = '^[1-9][0-9]A[1-9][0-9]{4,5}$'

# Biển số xe Ô tô - tải
# XXn/nn- - XXX.XX
#  sêri biển số sử dụng lần lượt một trong 20 chữ cái sau đây: A, B, C, D, E, F, G, H, K, L, M, N, P, S, T, U, V, X, Y, Z
regex_oto = '^[1-9][0-9][ABCDEFGHKLMNPSTUVXYZ][0-9]{4,5}$'

# Biển số đặc biệt
regex_specical = '^[1-9][0-9](LD|DA|R|T|MK|CD|T#)[0-9]{4,5}$'

# Military plate license
# AA     | 12-34
# prefix | suffix
regex_military = '^(AA|AB|AC|AD|AV|AT|AN|AP|BBB|BC|BH|BK|BL|BT|BP|BS|BV|HA|HB|HC|HD|HE|HT|HQ|HN|HH|\
                   KA|KB|KC|KD|KV|KP|KK|KT|KN|PA|PP|PM|PK|PT|PY|PQ|PX|PC|HL|QA|QB|QH|TC|TH|TK|TT|\
                   TM|TN|DB|ND|CH|VB|VK|CV|CA|CP|CM|CC|VT|CB)[0-9]{4}$'

                    
# Electric motorbike
# XX-MĐx / XXX.XX
#regex_electric_moto = r'^[1-9][0-9]M[#D][1-9][0-9]{4,5}$'
regex_electric_moto = r'^[1-9][0-9]M[#D][1-9][0-9]{5}$'

# Biển số tạm thời
regex_temp = r'^T[1-9][0-9][0-9]{4,5}$'

# Foreign plate license
# 41-291-NG-01
regex_foreign = r'^[1-9][0-9][0-9]{3}(NN|NG|CV|QT)[0-9]{2}$'

pattern_general_list = [regex_electric_moto, regex_military, regex_specical, regex_powerful_moto, regex_100cc_moto, regex_50cc_moto, regex_temp, regex_foreign, regex_oto]

pattern_car_list = [None, regex_military, regex_specical, None, None, None, regex_temp, regex_foreign, regex_oto]

def matching_plate(plate, label=None):
    if label is None:
        pattern_list = pattern_general_list
    elif label == 'car' or label == 'van' or label == 'truck':
        pattern_list = pattern_car_list
        
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        if pattern is not None:
            prog = re.compile(pattern)
            result = prog.match(plate)
            #print(pattern, result, plate)
            if result is not None:
                return i
    return -1
        
def postprocess(plate, probs):
    thresh = 0.6
    plate = list(plate)
    
    if len(plate) <= 5:
        return ""
    num_alpha = 0
    for c in plate:
        if c.isalpha() or c == '#':
            num_alpha += 1

#     if num_alpha == 0:
#         if plate[2] == '6':
#             plate[2] = 'G'
#         elif plate[2] == '8':
#             plate[2] = 'B'
#         elif plate[2] == '2':
#             plate[2] = 'Z'
#         elif plate[2] == '0':
#             plate[2] = 'D'
    
    
    plate = ''.join(plate)
    #print(plate, probs)
    for i in range(len(plate)):
        if probs[i] < thresh:
            return ""
    pattern_index = matching_plate(plate)
    if pattern_index == -1:
        return ""
    else:
        return plate
    
if __name__ == '__main__':
    #Biển số xe 50cc
    plate_test_0 = {"26AA0021": 'T', "26AA00212": 'T', "26AO00212": 'F', "26AB002125": 'F', "26AB002": 'F', "6AB0021": 'F'}
    for plate in plate_test_0.keys():
        pattern_index = matching_plate(plate)
        if pattern_index == -1:
            result = "F"
        else:
            result = "T"
        if result != plate_test_0[plate]:
            print("Test {} is failed.".format(plate))
    # Biển số xe 100cc - 150cc
    plate_test_1 = {"26B10034": 'T', "26B100345": 'T'}
    for plate in plate_test_1.keys():
        pattern_index = matching_plate(plate)
        if pattern_index == -1:
            result = "F"
        else:
            result = "T"
        if result != plate_test_1[plate]:
            print("Test {} is failed.".format(plate), pattern_index)
